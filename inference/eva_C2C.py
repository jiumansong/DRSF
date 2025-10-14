import argparse
from C2C_PolicyAndMamba_eval import ModelValidator
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.models_mamba_inference import VisionMamba
import torch
from dataloader_features import load_test_data
# from dataloader_features_WSIeva import load_test_data         


def log_message(message, log_file):
    print(message)  
    with open(log_file, 'a') as f:  
        f.write(message + '\n')


def get_args_parser():
    parser = argparse.ArgumentParser('RL_Mamba classification', add_help=False)
    parser.add_argument('--num_classes', default=5, type=int)   
    parser.add_argument('--feature_dim', default=1024, type=int)  
    parser.add_argument('--hidden_dim', default=512, type=int)  
    parser.add_argument('--class_cate', default=['0', '1', '2', '3', '4'], type=list)  
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')  
    parser.add_argument('--patch_budget', default=20, type=int, help='patch sampling numple in one input')
    parser.add_argument('--mamba_depth', default=32, type=int)
    parser.add_argument('--model_path', type=str,
                        default='saved_models/best_joint_model.pth',
                        help='Path to the saved model checkpoint')
    parser.add_argument('--save_path', type=str, default='validation_results',
                        help='Path to save validation results')

    return parser


def main(args):
    print("Namespace parameters: ", args)
    device = torch.device(args.device)

    mamba_classifier = VisionMamba(
        embed_dim=1024, depth=args.mamba_depth, num_classes=args.num_classes, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True).to(device)

    test_loader = load_test_data()

    # Initialize validator
    validator = ModelValidator(
        model_path=args.model_path,
        mamba_classifier=mamba_classifier,
        device=device,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim
    )
    for epoch in range(1):
        val_metrics, all_labels, all_predictions, results = validator.evaluate(
            val_loader=test_loader,
            budget=args.patch_budget,
            save_path=args.save_path
        )
        log_file = os.path.join(args.save_path, 'eval_log.txt')

        log_message(f"Validation :", log_file)
        log_message(f"  Validation Loss = {val_metrics['loss']:.6f}", log_file)
        log_message(f"  Validation Accuracy = {val_metrics['accuracy']:.6f}", log_file)

        log_message(f"  Micro f1 = {val_metrics['micro_avg']['f1']:.6f}", log_file)
        log_message(f"  Weighted f1 = {val_metrics['weighted_avg']['f1']:.6f}", log_file)

        log_message(f"  Macro Precision: = {val_metrics['macro_avg']['precision']:.6f}", log_file)
        log_message(f"  Micro Precision: = {val_metrics['micro_avg']['precision']:.6f}", log_file)
        log_message(f"  Weighted Precision: = {val_metrics['weighted_avg']['precision']:.6f}", log_file)

        log_message(f"  Macro recall: = {val_metrics['macro_avg']['recall']:.6f}", log_file)
        log_message(f"  Micro recall: = {val_metrics['micro_avg']['recall']:.6f}", log_file)
        log_message(f"  Weighted recall: = {val_metrics['weighted_avg']['recall']:.6f}", log_file)

        log_message(f"  Macro roc_auc: = {val_metrics['macro_avg']['roc_auc']:.6f}", log_file)
        log_message(f"  Macro pr_auc: = {val_metrics['macro_avg']['pr_auc']:.6f}", log_file)

        log_message(f"  confusion_matrix:\n{val_metrics['confusion_matrix']}", log_file)

        log_message(f"True labels:, {all_labels}", log_file)
        log_message(f"Predicted labels:, {all_predictions}", log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RL_Mamba classification', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
