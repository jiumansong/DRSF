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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix, recall_score, precision_score
)
import seaborn as sns
from sklearn.preprocessing import label_binarize
from torch.nn.init import trunc_normal_
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
import tempfile



class JointTrainer:
    def __init__(self, mamba_classifier, mamba_optimizer, mamba_scheduler, device, num_classes, save_dir='./training_cache'):
        self.mamba_classifier = mamba_classifier
        self.mamba_optimizer = mamba_optimizer     
        self.mamba_scheduler = mamba_scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes             

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.mamba_scaler = GradScaler()


    def pre_train_classifier(self, train_loader):
        print("Starting classifier training phase...")
        self.mamba_classifier.train()

        total_batches = len(train_loader)
        classification_loss_sum = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, batch in enumerate(train_loader):
            patches_features = batch["patches_features"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_size = patches_features.size(0)
            patches_per_sample = patches_features.size(1)
            feature_dim = patches_features.size(2)

            with autocast():              
                predictions = self.mamba_classifier(patches_features)
                classification_loss = self.criterion(predictions, labels)

            self.mamba_optimizer.zero_grad()
            self.mamba_scaler.scale(classification_loss).backward()
            self.mamba_scaler.unscale_(self.mamba_optimizer)
            torch.nn.utils.clip_grad_norm_(self.mamba_classifier.parameters(), max_norm=1.0)
            self.mamba_scaler.step(self.mamba_optimizer)
            self.mamba_scaler.update()

            classification_loss_sum += classification_loss.item()
            pred_labels = predictions.argmax(dim=1)
            correct_predictions += (pred_labels == labels).sum().item()
            total_samples += batch_size
                
            if (batch_idx + 1) % 50 == 0:
                accuracy = (correct_predictions / total_samples) * 100
                print(f"Batch {batch_idx + 1}/{total_batches}, "
                      f"Loss: {classification_loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        self.mamba_scheduler.step()
        epoch_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
        avg_classification_loss = classification_loss_sum / total_batches if total_batches > 0 else 0.0

        return epoch_accuracy, avg_classification_loss

    def pre_evaluate_classifier(self, val_loader, save_path=None):

        self.mamba_classifier.eval()

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
                    png_num = patches_features.size(0)
                    patches_per_sample = patches_features.size(1)
                    feature_dim = patches_features.size(2)

                    outputs = self.mamba_classifier(patches_features)

                    label = label.expand(outputs.size(0))
                    loss = self.criterion(outputs, label)

                    probabilities = F.softmax(outputs, dim=1)
                    slide_probability = probabilities.mean(dim=0)  # For ensemble aggregation strategy evaluation
                    slide_prediction = torch.argmax(slide_probability)      
                    slide_confidence = slide_probability[slide_prediction]   
                    val_loss += loss.item()
                    # Store slice results
                    results.append({
                        "slice_name": slice_name,
                        "true_label": slide_label.item(),
                        "predicted_label": slide_prediction.item(),
                        "confidence": slide_confidence.item()
                    })
                    all_labels_list.append(slide_label.unsqueeze(0))  
                    all_predictions_list.append(slide_prediction.unsqueeze(0))  
                    all_probabilities_list.append(slide_probability.unsqueeze(0))  

                all_labels = torch.cat(all_labels_list, dim=0)  
                all_predictions = torch.cat(all_predictions_list, dim=0) 
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

                return metrics

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()



    def calculate_metrics(self, predictions, labels, probabilities):
        """
        Calculate and plot evaluation metrics
        """
        metrics = {}

        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities.cpu().numpy()

        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))  
 
        micro_f1 = f1_score(y_true, y_pred, average='micro')             
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')        
        metrics['accuracy'] = ((y_true == y_pred).mean())*100                 

        # multi-class
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
