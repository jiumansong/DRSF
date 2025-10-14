import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse


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
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, patches_features, history_seq=None, available_mask=None, mode='both'):
        """
        Args:
            patches_features: [Batchsize, seq_len, feature_dim] 
            history_seq: [Batchsize, current_sampling_len, feature_dim]
            available_mask: [Batchsize, seq_len] boolean mask for available patches
            mode: 'both', 'policy', or 'classification'
        """
        classification_output = None
        if mode in ['both', 'classification'] and history_seq is not None:
            # Process through mamba_classifier
            classification_output = self.mamba_classifier(history_seq)

        # Policy output
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

            if available_mask is not None:
                q_values = q_values.masked_fill(~available_mask, float('-inf'))

        if mode == 'both':
            return q_values, classification_output
        elif mode == 'policy':
            return q_values
        else: 
            return classification_output


class ModelValidator:

    def __init__(self, model_path, mamba_classifier, device, num_classes, feature_dim=1024, hidden_dim=512):
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

        self.model = JointMambaPolicyNetwork(
            mamba_classifier=mamba_classifier,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        ).to(self.device)

        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load model parameters
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if 'policy_net_state_dict' in checkpoint:
                    state_dict = checkpoint['policy_net_state_dict']
                else:
                    print("No checkpoint")
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=True)
            print(f"Successfully load model from {model_path}")

            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f"Model is saved at epoch: {checkpoint['epoch']}")
                if 'best_accuracy' in checkpoint:
                    print(f"Best accuracy: {checkpoint['best_accuracy']:.4f}")

        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            raise

    def evaluate(self, val_loader, budget, save_path=None):
        """
        Args:
            val_loader: DataLoader containing validation/test data
            budget: Number of patches to sample per image
            save_path: Path to save evaluation results
        """
        self.model.eval() 
        self.model.mamba_classifier.eval()

        val_loss = 0
        all_labels_list = []
        all_predictions_list = []
        all_probabilities_list = []
        results = []  

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    patches_features = batch["patches_features"].to(self.device)
                    slide_label = torch.tensor(batch["label"])  
                    label = torch.tensor(batch["label"]).to(self.device)
                    slice_name = batch["slide_name"]  

                    batch_size = patches_features.size(0)
                    seq_len = patches_features.size(1)
                    feature_dim = patches_features.size(2)

                    actual_budget = min(budget, seq_len)
                    if actual_budget < budget:
                        print(
                            f"Warning: Budget {budget} exceeds available patches {seq_len} for slide {slice_name}. Using {actual_budget} patches.")

                    # Initialize
                    available_mask = torch.ones(batch_size, seq_len, device=self.device, dtype=torch.bool)
                    history_seq = None  
                    # Patch sampling phase
                    for step in range(actual_budget):
                        if history_seq is None:
                            q_values = self.model(
                                patches_features, history_seq, available_mask, mode='policy'
                            )
                        else:
                            q_values, _ = self.model(
                                patches_features, history_seq, available_mask, mode='both'
                            )

                        action = q_values.argmax(dim=1)

                        action_expanded = action.unsqueeze(1).unsqueeze(-1).expand(-1, -1, feature_dim)
                        selected_feature = patches_features.gather(1, action_expanded)

                        if history_seq is None:
                            history_seq = selected_feature.clone() 
                        else:
                            history_seq = torch.cat([history_seq, selected_feature], dim=1) 

                        next_available_mask = available_mask.clone()
                        next_available_mask[torch.arange(batch_size), action] = False
                        available_mask = next_available_mask

                    # Classification phase 
                    if history_seq is not None:
                        outputs = self.model(None, history_seq, None, mode='classification')
                        label = label.expand(outputs.size(0))
                        loss = self.criterion(outputs, label)

                        slide_probability = torch.nn.functional.softmax(outputs, dim=1)  
                        slide_probability = slide_probability.mean(dim=0)   # For ensemble aggregation strategy evaluation
                        #slide_probability = slide_probability.squeeze()   # For WSI evaluation
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

                    if (batch_idx + 1) % 50 == 0:
                        print(f"Processed {batch_idx + 1}/{len(val_loader)} batches")

                all_labels = torch.cat(all_labels_list, dim=0)  
                all_predictions = torch.cat(all_predictions_list, dim=0)  
                all_probabilities = torch.cat(all_probabilities_list, dim=0)  

                metrics = self.calculate_metrics(all_predictions, all_labels, all_probabilities)
                metrics['loss'] = val_loss / len(val_loader)

                # Save and plot results
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

                return metrics, all_labels, all_predictions, results

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()

    def calculate_metrics(self, predictions, labels, probabilities):
        metrics = {}

        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities.cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        micro_f1 = f1_score(y_true, y_pred, average='micro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        metrics['accuracy'] = (y_true == y_pred).mean() * 100

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

        # Calculate macro averages
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

        # Calculate micro and weighted averages
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

        # Calculate confusion matrix
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
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(save_path, 'pr_curves.png'), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

