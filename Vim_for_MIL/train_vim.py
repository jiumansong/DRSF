import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.models_mamba import VisionMamba
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataloader_features import load_train_data, load_test_data
from engine import train_pre_mamba
from torchvision import transforms
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('Vim classification', add_help=False)
    parser.add_argument('--num_classes', default=5, type=int)   

    parser.add_argument('--class_cate', default=['0', '1', '2', '3', '4'], type=list)  
    parser.add_argument('--batch_size', default=64, type=int) 
    parser.add_argument('--epochs', default=200, type=int)  
    parser.add_argument('--device', default='cuda:2', help='device to use for training / testing') 

    parser.add_argument('--mamba_depth', default=32, type=int)
    parser.add_argument('--mamba_lr', type=float, default=1e-6, help='mamba learning rate')
    parser.add_argument('--mamba_scheduler_milestones', default='30,60,90,120', type=lambda x: [int(i) for i in x.split(',')],
                        help='List of milestones for learning rate scheduler')
    parser.add_argument('--mamba_scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')  
    parser.add_argument('--picture_save_path', default="metrics_statistics", type=str)  
    parser.add_argument('--model_save_path', default="saved_models", type=str)

    return parser


def main(args):

    device = torch.device(args.device)

    mamba_classifier = VisionMamba(
        embed_dim=1024, depth=args.mamba_depth, num_classes=args.num_classes, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True).to(device)

    mamba_optimizer = optim.Adam(mamba_classifier.parameters(), lr=args.mamba_lr, weight_decay=args.weight_decay)
    mamba_scheduler = MultiStepLR(
        mamba_optimizer,
        milestones=args.mamba_scheduler_milestones, 
        gamma=args.mamba_scheduler_gamma  
    )

    num_classes = args.num_classes
    train_loader = load_train_data(batch_size=args.batch_size)
    test_loader = load_test_data()


    train_pre_mamba(
                    mamba_classifier=mamba_classifier,
                    mamba_optimizer=mamba_optimizer,
                    mamba_scheduler=mamba_scheduler,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    num_epochs=args.epochs,
                    device=device,
                    num_classes=num_classes,
                    png_save_path=args.picture_save_path,
                    model_save_path=args.model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vim classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.picture_save_path:
        Path(args.picture_save_path).mkdir(parents=True, exist_ok=True)
    if args.model_save_path:
        Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    main(args)
