import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataloader_MocoV2 import WSIPatchDataset
from MocoV2 import *


def get_mocov2_augmentations(size=224):
    """
    MoCo v2 augmentations with added Gaussian blur
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),  # MoCo v2 uses grayscale
        transforms.GaussianBlur(kernel_size=max(1, int(0.1 * size) // 2 * 2 + 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def train_mocov2(model, train_loader, optimizer, scheduler, epochs=100, save_path='mocov2_model_pretrained_mocov2.pth'):
    """
    Training loop for MoCo v2
    """
    model.train()
    best_loss = float('inf')
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}') as progress_bar:
            for images in progress_bar:
                optimizer.zero_grad()
                batch_img1, batch_img2 = images
                im_q = batch_img1.cuda()
                im_k = batch_img2.cuda()

                logits, labels = model(im_q, im_k)

                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'Model saved at epoch {epoch + 1} with loss {best_loss:.4f}')

    return model


def extract_features(model, data_loader):
    """
    Extract features from a trained MoCo v2 model
    """
    model.eval()
    features = []

    with torch.no_grad():
        for images in tqdm(data_loader, desc="Extracting features"):
            # For inference, we only need one view
            if isinstance(images, list) or isinstance(images, tuple):
                img1 = images[0]

            img1 = img1.cuda()
            # Use forward_features instead of the standard forward
            batch_features = model.forward_features(img1)
            features.append(batch_features.cpu().numpy())

    return np.vstack(features)


def visualize_features(features, labels=None, perplexity=30):
    """
    Visualize features using t-SNE
    """
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(10, 8))
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=f'Class {label}')
        plt.legend()
    else:
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1])

    plt.title('t-SNE visualization of MoCo v2 features')
    plt.savefig('tsne_mocov2_features.png')
    plt.close()


def main(patches_dir, scheduler_milestones, scheduler_gamma,
         batch_size=128, epochs=100, learning_rate=0.0003, save_dir='./model_outputs',
         queue_size=4096, momentum=0.999, temperature=0.07, device=0):
    """
    Main function to train MoCo v2 model and extract features
    """
    torch.cuda.set_device(device)
    print(f"Using CUDA device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    transform = get_mocov2_augmentations()
    transform = TwoCropsTransform(transform)

    dataset = WSIPatchDataset(patches_dir=patches_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Create MoCo v2 model
    model = MoCoV2FeatureExtractor(dim=128, K=queue_size, m=momentum, T=temperature).cuda()

    optimizer = torch.optim.AdamW(model.encoder_q.parameters(), lr=learning_rate, weight_decay=1e-6)
    optimizer_model_scheduler = MultiStepLR(
        optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma
    )

    # Train model
    model = train_mocov2(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        scheduler=optimizer_model_scheduler,
        epochs=epochs,
        save_path=os.path.join(save_dir, 'mocov2_model_pretrained_mocov2.pth')
    )
    """
    inference_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    """

    #inference_dataset = WSIPatchDataset(patches_dir=patches_dir, transform=inference_transform)
    #inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    #print("Extracting features...")
    #features = extract_features(model, inference_loader)

    #features_path = os.path.join(save_dir, 'patch_features_mocov2_pretrained_mocov2.npy')
    #np.save(features_path, features)
    #print(f"Features saved to {features_path}")

    #visualize_features(features)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MoCo v2 on IHC WSI patches')
    parser.add_argument('--patches_dir', type=str, default='path/MGMT_data/patches_overlap0',
                        help='Directory containing WSI patches')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./model_outputs', help='Directory to save outputs')
    parser.add_argument('--scheduler_milestones', default='100', type=lambda x: [int(i) for i in x.split(',')],
                        help='List of milestones for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    # MoCo v2 parameters
    parser.add_argument('--queue_size', type=int, default=4096, help='Queue size for MoCo v2')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum coefficient for key encoder update')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use')

    args = parser.parse_args()

    main(
        patches_dir=args.patches_dir,
        scheduler_milestones=args.scheduler_milestones,
        scheduler_gamma=args.scheduler_gamma,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        device=args.cuda,
    )
