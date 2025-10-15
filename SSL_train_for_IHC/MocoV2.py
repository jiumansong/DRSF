import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet50_pretrainedTCGA_output1024 import PatchEmbed_ResNet50_PretrainedTCGA


class MoCoV2FeatureExtractor(nn.Module):
    """
    MoCo v2 Feature Extractor using the first 3 layers of ResNet50
    Implementation based on "Improved Baselines with Momentum Contrastive Learning"
    """

    def __init__(self, dim=128, K=4096, m=0.999, T=0.07):
        """
        dim: feature dimension 
        K: queue size 
        m: momentum for updating key encoder 
        T: softmax temperature 
        """
        super(MoCoV2FeatureExtractor, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = nn.Sequential(
            PatchEmbed_ResNet50_PretrainedTCGA(),  
            nn.Linear(1024, 512),  
            nn.ReLU(inplace=True),
            nn.Linear(512, dim)
        )

        self.encoder_k = nn.Sequential(
            PatchEmbed_ResNet50_PretrainedTCGA(),  
            nn.Linear(1024, 512),  
            nn.ReLU(inplace=True),
            nn.Linear(512, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward_features(self, x):
        """Extract 1024-dim features without projection head"""
        return self.encoder_q[0](x)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:, :remaining].T
            self.queue[:, :batch_size - remaining] = keys[:, remaining:].T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images 
        Output:
            logits, targets, query features
        """
        if im_k is None:
            return self.forward_features(im_q)

        q = self.encoder_q(im_q) 
        q = F.normalize(q, dim=1)

        with torch.no_grad():  
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)  
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCoV2Loss(nn.Module):
    def __init__(self):
        super(MoCoV2Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)


def train_one_epoch(model, data_loader, optimizer, epoch):
    model.train()

    for images, _ in data_loader:
        im_q, im_k = get_two_augmentations(images)  

        im_q, im_k = im_q.cuda(), im_k.cuda()

        logits, labels = model(im_q, im_k)

        loss = MoCoV2Loss()(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.cuda()
            batch_features = model.forward_features(images)
            features.append(batch_features.cpu())
            labels.append(targets)

    return torch.cat(features), torch.cat(labels)


def get_two_augmentations(images):
    import torchvision.transforms as transforms

    transform_q = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),  # MoCo v2 adds a Gaussian blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_k = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_q(images), transform_k(images)
