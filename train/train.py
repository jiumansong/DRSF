import argparse
from PolicyAndMamba import JointTrainingAgent
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.models_mamba import VisionMamba
import torch
from dataloader_features import load_train_data, load_test_data, load_policy_test_data
from engine import train_joint_model
import random
import numpy as np
import pandas as pd


def save_metrics_to_excel(train_metrics, val_metrics, save_path='training_metrics.xlsx'):
    df_train = pd.DataFrame(train_metrics)
    df_val = pd.DataFrame(val_metrics)

    with pd.ExcelWriter(save_path) as writer:
        df_train.to_excel(writer, sheet_name='Train Metrics', index_label='Epoch')
        df_val.to_excel(writer, sheet_name='Val Metrics', index_label='Epoch')


def set_seed(seed):
    random.seed(seed)                 
    np.random.seed(seed)              
    torch.manual_seed(seed)           
    torch.cuda.manual_seed(seed)      
    torch.cuda.manual_seed_all(seed)  


def get_args_parser():
    parser = argparse.ArgumentParser('RL-Mamba Joint Training', add_help=False)
    parser.add_argument('--num_classes', default=5, type=int)  
    parser.add_argument('--class_cate', default=['0', '1', '2', '3', '4'], type=list) 
    parser.add_argument('--batch_size', default=64, type=int)  
    parser.add_argument('--epochs', default=150, type=int)  
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing') 
    parser.add_argument('--patch_budget', default=20, type=int, help='patch sampling numple in one bag')
    parser.add_argument('--mamba_depth', default=32, type=int)
    parser.add_argument('--lr', type=float, default=1e-5, help='sampling model learning rate (default: 1e-4)')
    parser.add_argument('--scheduler_milestones', default='30,60,90,120', type=lambda x: [int(i) for i in x.split(',')],
                        help='List of milestones for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')  
    parser.add_argument('--picture_save_path', default="metrics_statistics", type=str)  
    parser.add_argument('--model_save_path', default="saved_models", type=str)

    return parser


def main(args):
    set_seed(42)
    print("Namespace parameters: ", args)
    device = torch.device(args.device)

    mamba_classifier = VisionMamba(
        embed_dim=1024, depth=args.mamba_depth, num_classes=args.num_classes, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True).to(device)
    print(mamba_classifier)
    ## Load model parameters obtained from Vim as initialization for the classification module of RL-Mamba
    checkpoint = torch.load('path/initialization_mamba.pth', map_location=device) 
    result = mamba_classifier.load_state_dict(checkpoint['mamba_classifier_state_dict'], strict=False)
    total_params = sum(p.numel() for p in mamba_classifier.parameters())
    trainable_params = sum(p.numel() for p in mamba_classifier.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    num_classes = args.num_classes
    train_loader = load_train_data(batch_size=args.batch_size)
    test_loader = load_test_data()

    agent = JointTrainingAgent(
        mamba_classifier=mamba_classifier,
        feature_dim=1024,
        args=args
    )

    train_metrics, val_metrics = train_joint_model(
        agent=agent,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=args.epochs,
        device=device,
        num_classes=num_classes,  
        patch_budget=args.patch_budget,  
        png_save_path=args.picture_save_path,
        model_save_path=args.model_save_path
    )
    matrics_save_filename = 'joint_training_results.xlsx'
    matrics_save_path = os.path.join(args.picture_save_path, matrics_save_filename)
    save_metrics_to_excel(train_metrics, val_metrics, save_path=matrics_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RL-Mamba Joint Training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.picture_save_path:
        Path(args.picture_save_path).mkdir(parents=True, exist_ok=True)
    if args.model_save_path:
        Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    main(args)
