import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from collections import deque
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import pickle
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from torch.nn.init import trunc_normal_
from sklearn.metrics import f1_score
from pri_ReMemory import PrioritizedReplayMemory, SumTree, Transition
import tempfile
from sklearn.metrics import precision_score, recall_score, f1_score


class JointMambaPolicyNetwork(nn.Module):
    def __init__(self, mamba_classifier, feature_dim=1024, hidden_dim=512, num_classes=5, dropout_prob=0.3):
        super(JointMambaPolicyNetwork, self).__init__()

        self.mamba_classifier = mamba_classifier
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=hidden_dim,
                                       dropout=dropout_prob, batch_first=True), num_layers=2)
        
        self.projection_layer = nn.Linear(feature_dim, hidden_dim)

        self.feature_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )
        self.global_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )

        # Decision network for policy 
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1),
        )

        self._initialize_weights()

    def forward(self, patches_features, history_seq=None, available_mask=None, mode='both'):
        """
        Args:
            patches_features: [Batchsize, seq_len, feature_dim] 
            history_seq: [Batchsize, history_seq_len, feature_dim]
            available_mask: [Batchsize, seq_len] boolean mask for available patches
            mode: 'both', 'policy', or 'classification'
        """
        classification_output = None
        if mode in ['both', 'classification'] and history_seq is not None:
            classification_output = self.mamba_classifier(history_seq)

        q_values = None
        if mode in ['both', 'policy']:
            batch_size = patches_features.size(0)
            patches_per_sample = patches_features.size(1)
            feature_dim = patches_features.size(2)

            if history_seq is None or history_seq.size(1) == 0:
                history_seq = torch.zeros(batch_size, 1, feature_dim, device=patches_features.device)
            history_seq = history_seq.permute(1, 0, 2)                
            transformer_out = self.transformer_encoder(history_seq)  
            hidden_state = transformer_out[-1, :, :]                 
            hidden_state = self.projection_layer(hidden_state)        
            history_embeds = hidden_state.unsqueeze(1).expand(-1, patches_per_sample, -1)   

            patch_embed = self.feature_network(patches_features.view(-1, feature_dim))   
            patch_embed = patch_embed.view(batch_size, patches_per_sample, -1).contiguous()  

            global_states = patches_features.mean(dim=1)  
            global_embeds = self.global_network(global_states)  
            global_embeds = global_embeds.unsqueeze(1).expand(-1, patches_per_sample, -1)  

            combined_out = torch.cat([patch_embed, global_embeds, history_embeds], dim=-1)  

            q_values = self.decision_network(combined_out.view(-1, combined_out.size(-1))).contiguous()
            q_values = q_values.view(batch_size, patches_per_sample).contiguous()

            # Mask unavailable patches
            if available_mask is not None:
                q_values = q_values.masked_fill(~available_mask, float('-inf'))

        if mode == 'both':
            return q_values, classification_output
        elif mode == 'policy':
            return q_values
        else:  # mode == 'classification'
            return classification_output

    def _initialize_weights(self):
        def init_fn(m):
            if any(m is layer for layer in self.mamba_classifier.modules()):
                return
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='sigmoid')
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='sigmoid')
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        for module in self.modules():
            if module not in self.mamba_classifier.modules():
                init_fn(module)


class JointTrainingAgent:
    def __init__(self, mamba_classifier, feature_dim=1024, hidden_dim=512,
                memory_size=10000, gamma=0.90, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.950,
                classification_weight=0.5, **kwargs):
        self.args = kwargs.get("args", None)
        self.device = self.args.device
        self.num_classes = self.args.num_classes
        self.classification_weight = classification_weight
        self.args = kwargs.get("args", None)

        self.policy_net = JointMambaPolicyNetwork(
            mamba_classifier=mamba_classifier,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
        ).to(self.device)

        import copy
        target_mamba_classifier = copy.deepcopy(mamba_classifier)

        self.target_net = JointMambaPolicyNetwork(
            mamba_classifier=target_mamba_classifier,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict(), strict=False)

        self.memory = PrioritizedReplayMemory(memory_size)
        self.batch_size = self.args.batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = MultiStepLR(
                                    self.optimizer,
                                    milestones=self.args.scheduler_milestones,  
                                    gamma=self.args.scheduler_gamma 
                                    )

        # PER parameters
        self.beta = 0.4
        self.class_loss = nn.CrossEntropyLoss()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        total_policy_loss = 0.0
        total_classification_loss = 0.0
        self.optimizer.zero_grad()

        try:
            transitions, indices, weights = self.memory.sample(self.batch_size, self.beta)
            td_errors = []

            for i in range(self.batch_size):
                transition = transitions[i]
                weight = torch.tensor([weights[i]], device=self.device, dtype=torch.float16)

                # Extract transitions
                patches_features = transition.patches_features.unsqueeze(0).to(self.device)
                history_seq = transition.history_seq.unsqueeze(0).to(
                    self.device) if transition.history_seq is not None else None
                available_mask = transition.available_mask.unsqueeze(0).to(self.device)
                action = transition.action.unsqueeze(0).to(self.device)
                reward = transition.reward.to(self.device, dtype=torch.float16)
                label = transition.label.unsqueeze(0).to(self.device)

                non_final = transition.next_history_seq is not None
                next_history_seq = transition.next_history_seq.unsqueeze(0).to(self.device) if non_final else None
                next_available_mask = transition.next_available_mask.unsqueeze(0).to(self.device) if non_final else None

                # Compute current Q values and classification output
                q_values, classification_output = self.policy_net(
                    patches_features,
                    history_seq,
                    available_mask,
                    mode='both'
                )

                state_action_value = q_values.gather(1, action.unsqueeze(1))

                cls_loss = self.class_loss(classification_output, label)
                total_classification_loss += cls_loss.item()

                next_state_value = torch.zeros(1, device=self.device, dtype=torch.float16)
                if non_final:
                    with torch.no_grad():
                        # Double DQN approach
                        q_next_policy = self.policy_net(patches_features, next_history_seq, next_available_mask,
                                                            mode='policy')
                        best_next_action = q_next_policy.argmax(1, keepdim=True)

                        q_next_target = self.target_net(patches_features, next_history_seq, next_available_mask,
                                                            mode='policy')
                        next_state_value[0] = q_next_target.gather(1, best_next_action).squeeze(1)

                expected_state_action_value = (next_state_value * self.gamma) + reward
                expected_state_action_value = expected_state_action_value.unsqueeze(1).to(state_action_value.dtype)

                # Compute TD error
                td_error = torch.abs(state_action_value - expected_state_action_value).detach().cpu().numpy()
                td_errors.append(td_error[0][0])

                # Compute policy loss
                policy_loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)
                total_policy_loss += policy_loss.item()

                combined_loss = policy_loss + self.classification_weight * cls_loss
                weighted_loss = weight * combined_loss

                scaled_loss = weighted_loss / self.batch_size
                scaled_loss.backward()

                del patches_features, history_seq, available_mask, action, reward
                del next_history_seq, next_available_mask, q_values, state_action_value
                if non_final:
                    del q_next_policy, q_next_target, best_next_action

            # Update priorities
            self.memory.update_priorities(indices, np.array(td_errors))

            self.beta = min(0.8, self.beta + 0.001)

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()  

            avg_policy_loss = total_policy_loss / self.batch_size
            avg_classification_loss = total_classification_loss / self.batch_size

            torch.cuda.empty_cache()  
            return avg_policy_loss, avg_classification_loss

        except Exception as e:
            print(f"Error during optimize_model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict(), strict=False)


class JointTrainer:
    def __init__(self, agent, device, num_classes):
        self.agent = agent
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def compute_similarity_reward(self, selected_feature, new_confidence, old_confidence, step, max_step,
                                  history_features=None, labels=None):
        batch_size = selected_feature.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        conf_diff = new_confidence - old_confidence
        conf_reward = torch.tanh(conf_diff * 5.0)  
        abs_conf_reward = torch.tanh(new_confidence * 2.0) * 0.2
        class_reward = torch.zeros_like(rewards)
        if step > 0:  
            classification_output = self.agent.policy_net(
                patches_features=None,
                history_seq=history_features,
                available_mask=None,
                mode='classification'
            )
            _, predicted_class = torch.max(torch.nn.Softmax(dim=1)(classification_output), dim=1)

            correct_prediction = (predicted_class == labels).float()
            class_reward = correct_prediction * 0.5
        early_bonus = (1.0 - step / max_step) * 0.2
        final_reward = conf_reward + abs_conf_reward + class_reward + early_bonus
        final_reward = torch.clamp(final_reward, min=-1.0, max=1.0)

        return final_reward

    def train_joint_epoch(self, train_loader, budget):
        """
        Train the joint network for one epoch.
        Args:
            train_loader: DataLoader containing the training data
            budget: Number of patches to sample per sample
        """
        print("Starting joint network training phase...")
        self.agent.policy_net.train() 

        epoch_joint_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_rewards = 0.0
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            patches_features = batch["patches_features"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = patches_features.size(0)
            patches_per_sample = patches_features.size(1)
            feature_dim = patches_features.size(2)

            available_mask = torch.ones(batch_size, patches_per_sample, device=self.device, dtype=torch.bool)
            history_seq = None  

            episode_reward = 0
            # Patch sampling phase
            for step in range(budget):
                with torch.no_grad():
                    if history_seq is None:
                        q_values = self.agent.policy_net(
                            patches_features, history_seq, available_mask, mode='policy'
                        )
                        old_confidence = torch.zeros(batch_size, device=self.device)
                    else:
                        q_values, classification_output = self.agent.policy_net(
                            patches_features, history_seq, available_mask, mode='both'
                        )
                        classification_probs = torch.nn.Softmax(dim=1)(classification_output)
                        old_confidence = classification_probs[torch.arange(batch_size), labels]

                if random.random() < self.agent.epsilon:
                    actions_random = []
                    # Random action
                    for i in range(batch_size):
                        valid_actions = torch.where(available_mask[i])[0]
                        if len(valid_actions) > 0:
                            random_index = random.randrange(len(valid_actions))
                            actions_random.append(valid_actions[random_index])
                        else:
                            break
                    action = torch.tensor(actions_random).to(self.device)
                else:
                    action = q_values.argmax(dim=1)  

                action_expanded = action.unsqueeze(1).unsqueeze(-1).expand(-1, -1, feature_dim)
                selected_feature = patches_features.gather(1, action_expanded)

                # Update history sequence 
                if history_seq is None:
                    history_seq = selected_feature.clone()
                else:
                    history_seq = torch.cat([history_seq, selected_feature], dim=1) 

                with torch.no_grad():
                    new_classification_output = self.agent.policy_net(history_seq, history_seq, None,
                                                                      mode='classification')
                    new_probs = torch.nn.Softmax(dim=1)(new_classification_output)
                    new_confidence = new_probs[torch.arange(batch_size), labels]

                reward = self.compute_similarity_reward(
                    selected_feature.detach(), new_confidence, old_confidence,
                    step, budget, history_seq.detach(), labels
                )  
                episode_reward += reward.mean().item() 

                next_available_mask = available_mask.clone()
                next_available_mask[torch.arange(batch_size), action] = False
                next_history_seq = history_seq.clone()

                is_terminal = (step == budget - 1)

                # Store transition
                for i in range(batch_size):
                    self.agent.memory.push(
                        patches_features[i],     
                        labels[i],
                        history_seq[i] if history_seq is not None else None,
                        available_mask[i],
                        action[i],
                        reward[i],
                        None if is_terminal else next_history_seq[i],
                        next_available_mask[i]
                    )

                available_mask = next_available_mask

                del selected_feature, reward, action, q_values
                torch.cuda.empty_cache()

            epoch_rewards += episode_reward

            # Use experience replay to update the joint network
            losses = self.agent.optimize_model()
            if losses is not None:
                policy_loss, classification_loss = losses
                epoch_policy_loss += policy_loss
                epoch_classification_loss += classification_loss
                epoch_joint_loss += policy_loss + self.agent.classification_weight * classification_loss

            if batch_idx % 20 == 0:  
                self.agent.update_target_network()

                if losses is not None:
                    if (batch_idx + 1) % 50 == 0:
                        print(f"Batch {batch_idx + 1}/{total_batches}, "
                              f"Policy Loss: {policy_loss:.4f}, "
                              f"Classification Loss: {classification_loss:.4f}, "
                              f"Epsilon: {self.agent.epsilon:.4f}")

        self.agent.update_epsilon()
        self.agent.scheduler.step()
        avg_policy_loss = epoch_policy_loss / max(1, total_batches)
        avg_classification_loss = epoch_classification_loss / max(1, total_batches)
        avg_joint_loss = epoch_joint_loss / max(1, total_batches)
        avg_epoch_rewards = (epoch_rewards / max(1, total_batches)) / budget

        return {
            'policy_loss': avg_policy_loss,
            'classification_loss': avg_classification_loss,
            'joint_loss': avg_joint_loss,
            'rewards': avg_epoch_rewards
        }

    def evaluate(self, val_loader, budget, save_path=None):
        self.agent.policy_net.eval() 

        val_loss = 0
        all_labels_list = []
        all_predictions_list = []
        all_probabilities_list = []
        results = []  

        try:
            with torch.no_grad():
                for batch in val_loader:
                    patches_features = batch["patches_features"].to(self.device)
                    slide_label = torch.tensor(batch["label"]) 
                    label = torch.tensor(batch["label"]).to(self.device)
                    slice_name = batch["slide_name"]  

                    batch_size = patches_features.size(0)
                    patches_per_sample = patches_features.size(1)
                    feature_dim = patches_features.size(2)

                    available_mask = torch.ones(batch_size, patches_per_sample, device=self.device, dtype=torch.bool)
                    history_seq = None  

                    for step in range(budget):
                        if history_seq is None:
                            q_values = self.agent.policy_net(
                                patches_features, history_seq, available_mask, mode='policy'
                            )
                        else:
                            q_values, _ = self.agent.policy_net(
                                patches_features, history_seq, available_mask, mode='both'
                            )

                        action = q_values.argmax(dim=1)

                        action_expanded = action.unsqueeze(1).unsqueeze(-1).expand(-1, -1, feature_dim)
                        selected_feature = patches_features.gather(1, action_expanded)

                        if history_seq is None:
                            history_seq = selected_feature.clone()  
                        else:
                            history_seq = torch.cat([history_seq, selected_feature],
                                                    dim=1)  

                        next_available_mask = available_mask.clone()
                        next_available_mask[torch.arange(batch_size), action] = False
                        available_mask = next_available_mask

                    if history_seq is not None:
                        outputs = self.agent.policy_net(None, history_seq, None, mode='classification')
                        label = label.expand(outputs.size(0))
                        loss = self.criterion(outputs, label)

                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        slide_probability = probabilities.mean(dim=0)  
                        slide_prediction = torch.argmax(slide_probability) 
                        slide_confidence = slide_probability[slide_prediction]  

                        val_loss += loss.item()

                        results.append({
                            "slice_name": slice_name,
                            "true_label": slide_label.item(),
                            "predicted_label": slide_prediction.item(),
                            "confidence": slide_confidence.item()
                        })

                        all_labels_list.append(slide_label.unsqueeze(0)) 
                        all_predictions_list.append(slide_prediction.unsqueeze(0))  
                        all_probabilities_list.append(slide_probability.unsqueeze(0))  

                    del history_seq, patches_features
                    torch.cuda.empty_cache()

                all_labels = torch.cat(all_labels_list, dim=0)  
                #print("True labels:", all_labels)
                all_predictions = torch.cat(all_predictions_list, dim=0)  
                #print("Predicted labels:", all_predictions)
                all_probabilities = torch.cat(all_probabilities_list, dim=0)  

                metrics = self.calculate_metrics(all_predictions, all_labels, all_probabilities)
                metrics['loss'] = val_loss / len(val_loader)

                if save_path is not None:
                    save_path = Path(save_path)
                    save_path.mkdir(parents=True, exist_ok=True)

                    results_path = save_path / 'evaluation_results.json'
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)

                    print("\nEvaluation Results:")
                    print("-" * 50)
                    for metric, value in metrics.items():
                        if isinstance(value, (float, int, np.float32, np.float64)):
                            print(f"{metric}: {value:.4f}")

                    self.plot_evaluation_metrics(metrics, save_path)

                return metrics, all_labels, all_predictions

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()

    def calculate_metrics(self, predictions, labels, probabilities):
        """
        Calculate metrics for multi-class
        """
        metrics = {}

        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities.cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))   
 
        micro_f1 = f1_score(y_true, y_pred, average='micro')              
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')       
        metrics['accuracy'] = (y_true == y_pred).mean()                 

        class_metrics = []         
        for i in range(self.num_classes):
            class_true = (y_true == i)        
            class_pred = (y_pred == i)        
            class_prob = y_prob[:, i]        

            precision, recall, _ = precision_recall_curve(class_true, class_prob)
            pr_auc = auc(recall, precision)    

            fpr, tpr, _ = roc_curve(class_true, class_prob)
            roc_auc = auc(fpr, tpr)           

            class_metrics.append({
                'precision': precision,
                'recall': recall,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc
            })

        metrics['class_metrics'] = class_metrics
        macro_precision = np.mean([np.mean(m['precision']) for m in class_metrics])
        macro_recall = np.mean([np.mean(m['recall']) for m in class_metrics])
        macro_pr_auc = np.mean([m['pr_auc'] for m in class_metrics])
        macro_roc_auc = np.mean([m['roc_auc'] for m in class_metrics])
        metrics['macro_avg'] = {
                'precision': macro_precision,
                'recall': macro_recall,
                'pr_auc': macro_pr_auc,
                'roc_auc': macro_roc_auc
                }                               

        micro_precision = precision_score(y_true, y_pred, average='micro')
        micro_recall = recall_score(y_true, y_pred, average='micro')

        weighted_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')

        metrics['micro_avg'] = {
            'f1': micro_f1,
            'precision': micro_precision,
            'recall': micro_recall,
        }
        metrics['weighted_avg'] = {
            'f1': weighted_f1,
            'precision': weighted_precision,
            'recall': weighted_recall,
        }

        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm          

        return metrics

    def plot_evaluation_metrics(self, metrics, save_path):
        """
        Plot and save evaluation metrics
        """
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()

        # Plot PR curves
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            class_metric = metrics['class_metrics'][i]
            plt.plot(class_metric['recall'], class_metric['precision'],
                     label=f'Class {i} (AUC = {class_metric["pr_auc"]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'pr_curves.png'))
        plt.close()

        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            class_metric = metrics['class_metrics'][i]
            plt.plot(class_metric['fpr'], class_metric['tpr'],
                     label=f'Class {i} (AUC = {class_metric["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'roc_curves.png'))
        plt.close()
