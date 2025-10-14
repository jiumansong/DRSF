from trainer import JointTrainer
import torch
import matplotlib.pyplot as plt
import os
from torchinfo import summary        
from thop import profile             
import psutil  #
import time


def log_message(message, log_file):
    print(message)  
    with open(log_file, 'a') as f:  
        f.write(f"{message}\n")


def train_pre_mamba(mamba_classifier, mamba_optimizer, mamba_scheduler, train_loader, val_loader,
                      num_epochs, device, num_classes, png_save_path="training_metrics", model_save_path="saved_models"):

    save_path = png_save_path
    model_save_path = model_save_path
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    log_file = os.path.join(model_save_path, 'train_log.txt')

    log_message("==== Mamba Classifier Structure ====", log_file)
    log_message(str(mamba_classifier), log_file)
    with open(log_file, "a") as f:
        model_summary = summary(
            mamba_classifier,
            input_size=(64, 100, 1024), 
            verbose=2,
        )
        f.write(str(model_summary) + "\n")

    trainer = JointTrainer(mamba_classifier, mamba_optimizer, mamba_scheduler, device, num_classes)

    train_metrics = {
        'classification_losses': [],
        'accuracies': []
    }
    val_metrics = {
        'losses': [],
        'accuracies': [],
        'best_accuracy': 0,
        'best_epoch': 0
    }

    training_times = []

    for epoch in range(num_epochs):
        log_message(f"\nEpoch {epoch +1}/{num_epochs}", log_file)

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9  
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9  
            log_message(f"Memory allocated: {memory_allocated:.4f} GB", log_file)
            log_message(f"Memory reserved: {memory_reserved:.4f} GB", log_file)
        else:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            log_message(f"Memory used (RAM): {memory_info.rss / 1e6:.4f} MB", log_file)

        start_epoch_time = time.time()
        # Train for one epoch
        epoch_accuracy, avg_classification_loss = trainer.pre_train_classifier(train_loader)
        train_metrics['classification_losses'].append(avg_classification_loss)
        train_metrics['accuracies'].append(epoch_accuracy)
        training_time = time.time() - start_epoch_time
        training_times.append(training_time)

        log_message(f"Classification Training Epoch {epoch +1}:", log_file)
        log_message(f"  Avg Classification Loss = {avg_classification_loss:.6f}", log_file)
        log_message(f"  Training Accuracy = {epoch_accuracy:.6f}%", log_file)

        start_inference_time = time.time()
        val_metrics_epoch = trainer.pre_evaluate_classifier(val_loader, save_path=None)
        inference_time = time.time() - start_inference_time
        avg_inference_time_per_sample = inference_time/len(val_loader)
        log_message(f"Average inference time per sample: {avg_inference_time_per_sample:.6f} seconds", log_file)


        val_metrics['losses'].append(val_metrics_epoch['loss'])
        val_metrics['accuracies'].append(val_metrics_epoch['accuracy'])
        if val_metrics_epoch['accuracy'] > val_metrics['best_accuracy']:
            val_metrics['best_accuracy'] = val_metrics_epoch['accuracy']
            val_metrics['best_epoch'] = epoch
            torch.save({
                'mamba_classifier_state_dict': mamba_classifier.state_dict(),
            }, os.path.join(model_save_path, 'best_pre_trained_mamba_model.pth'))
            log_message("Saving the model --------------------------------------------", log_file)
        log_message(f"Validation Epoch {epoch +1}:", log_file)
        log_message(f"  Validation Loss = {val_metrics_epoch['loss']:.6f}", log_file)
        log_message(f"  Validation Accuracy = {val_metrics_epoch['accuracy']:.6f}%", log_file)

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

    avg_training_time = sum(training_times) / len(training_times)
    log_message(f"Average epoch training time: {avg_training_time:.2f} seconds", log_file)
    avg_training_time_in_minutes = avg_training_time / 60 
    log_message(f"Average epoch training time: {avg_training_time_in_minutes:.2f} minutes", log_file)

    log_message(f"\nBest validation accuracy: {val_metrics['best_accuracy']:.6f}% at epoch {val_metrics['best_epoch']}", log_file)

    return 0


