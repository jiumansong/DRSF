import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class WSIPatchDataset(Dataset):
    """
    Dataset for loading WSI patches for self-supervised learning.
    """

    def __init__(self, patches_dir, transform=None):
        """
        Args:
            patches_dir (str): Directory containing WSI patches
            transform: Optional transform to be applied to patches
        """
        self.patches_dir = patches_dir
        self.transform = transform
        self.patch_files = [f for f in os.listdir(patches_dir) if f.endswith(('.png', '.jpg', '.tif'))]

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.patches_dir, self.patch_files[idx])
        image = Image.open(img_path).convert('RGB')  

        if self.transform:
            img1, img2 = self.transform(image) 

        return img1, img2