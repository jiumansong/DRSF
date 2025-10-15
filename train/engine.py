from PolicyAndMamba import JointTrainer, JointTrainingAgent
import torch
import matplotlib.pyplot as plt
import os
import sys

def log_message(message, log_file):
    print(message)  
    with open(log_file, 'a') as f:  
        f.write(message + '\n')

def train_joint_model(agent, train_loader, val_loader,
                      num_epochs, device, num_classes, patch_budget,
                      png_save_path="training_metrics", model_save_path="saved_models"):
    """
    Training loop for the joint training
    """
    save_path = png_save_path
    model_save_path = model_save_path
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    log_file = os.path.join(model_save_path, 'train_log.txt')

    trainer = JointTrainer(agent, device, num_classes)

    train_metrics = {
        'policy_losses': [],
        'classification_losses': [],
        'joint_losses': [],
        'episode_rewards': [],
        'accuracies': []
    }
    val_metrics = {
        'losses': [],
        'accuracies': [],
        'best_accuracy': 0,
        'best_epoch': 0
    }

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")

        joint_metrics = trainer.train_joint_epoch(train_loader, patch_budget)
        avg_policy_loss = joint_metrics['policy_loss']
        avg_classification_loss = joint_metrics['classification_loss']
        avg_joint_loss = joint_metrics['joint_loss']
        avg_epoch_rewards = joint_metrics['rewards']

        train_metrics['policy_losses'].append(avg_policy_loss)
        train_metrics['classification_losses'].append(avg_classification_loss)
        train_metrics['joint_losses'].append(avg_joint_loss)
        train_metrics['episode_rewards'].append(avg_epoch_rewards)

        log_message(f"Joint Training Epoch {epoch +1}:", log_file)
        log_message(f"  Avg Policy Loss = {avg_policy_loss:.4f}", log_file)
        log_message(f"  Avg Classification Loss = {avg_classification_loss:.4f}", log_file)
        log_message(f"  Avg Joint Loss = {avg_joint_loss:.4f}", log_file)
        log_message(f"  Avg Rewards = {avg_epoch_rewards:.4f}", log_file)
        train_metrics['accuracies'].append(None)

        val_metrics_epoch, all_labels, all_predictions = trainer.evaluate(val_loader,
                                                                        patch_budget, 
                                                                        save_path if epoch == num_epochs - 1 else None)

        val_metrics['losses'].append(val_metrics_epoch['loss'])
        val_metrics['accuracies'].append(val_metrics_epoch['accuracy'])

        if val_metrics_epoch['accuracy'] > val_metrics['best_accuracy']:
            val_metrics['best_accuracy'] = val_metrics_epoch['accuracy']
            val_metrics['best_epoch'] = epoch
            # Save policy network and target network
            torch.save({
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
            }, os.path.join(model_save_path, 'best_joint_model.pth'))
            log_message("Saving the model --------------------------------------------", log_file)
            
        log_message(f"Validation Epoch {epoch}:", log_file)
        log_message(f"  Validation Loss = {val_metrics_epoch['loss']:.6f}", log_file)
        log_message(f"  Validation Accuracy = {val_metrics_epoch['accuracy']:.6f}", log_file)

        log_message(f"  Micro f1 = {val_metrics_epoch['micro_avg']['f1']:.6f}", log_file)
        log_message(f"  Weighted f1 = {val_metrics_epoch['weighted_avg']['f1']:.6f}", log_file)

        log_message(f"  Macro Precision: = {val_metrics_epoch['macro_avg']['precision']:.6f}", log_file)
        log_message(f"  Micro Precision: = {val_metrics_epoch['micro_avg']['precision']:.6f}", log_file)
        log_message(f"  Weighted Precision: = {val_metrics_epoch['weighted_avg']['precision']:.6f}", log_file)

        log_message(f"  Macro recall: = {val_metrics_epoch['macro_avg']['recall']:.6f}", log_file)
        log_message(f"  Micro recall: = {val_metrics_epoch['micro_avg']['recall']:.6f}", log_file)
        log_message(f"  Weighted recall: = {val_metrics_epoch['weighted_avg']['recall']:.6f}", log_file)

        log_message(f"  Macro roc_auc: = {val_metrics_epoch['macro_avg']['roc_auc']:.6f}", log_file)
        log_message(f"  Macro pr_auc: = {val_metrics_epoch['macro_avg']['pr_auc']:.6f}", log_file)

        log_message(f"  confusion_matrix:\n{val_metrics_epoch['confusion_matrix']}", log_file) 


        log_message(f"True labels:, {all_labels}", log_file)
        log_message(f"Predicted labels:, {all_predictions}", log_file)

    log_message(f"\n Validation accuracy of the saved model: {val_metrics['best_accuracy']:.2f}% at epoch {val_metrics['best_epoch']}", log_file)

    return train_metrics, val_metrics