import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re


class PTFeatureDataset(Dataset):
    def __init__(self, root_dir):
        """
        Input is a large-size tissue image 
        Args:
            root_dir (str): The root directory containing patch features
        features/                 
        ├── train/                
        │   ├── class_0/          
        │   │   ├── xxx.pt        
        │   │   ├── ...           
        │   ├── class_1/          
        │   │   ├── yyy.pt
        │   │   ├── ...
        """
        self.root_dir = root_dir
        self.samples = []

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith(".pt"):
                        file_path = os.path.join(class_path, file)
                        label = int(class_folder.split("_")[-1]) 
                        slide_name = os.path.splitext(file)[0]
                        self.samples.append((file_path, label, slide_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, slide_name = self.samples[idx]
        data = torch.load(file_path)
        patches_features = data["patches_features"].squeeze(0)

        return {
            "patches_features": patches_features,  
            "label": label,  
            "slide_name": slide_name 
        }


def load_train_data(batch_size):
    train_feature_root = 'xxx/features/train'
    train_dataset = PTFeatureDataset(train_feature_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)    

    return train_loader


class SlideBasedPTFeatureDataset(Dataset):
    def __init__(self, root_dir):
        """
        Input is a WSI containing one/multiple large-size images (png) for WSI evaluation
        Args:
            root_dir (str): The root directory containing patch features
        features/                 
        ├── test/                
        │   ├── class_0/          
        │   │   ├── xxx_files_1.pt        
        │   │   ├── xxx_files_2.pt    
        │   │   ├── ... 
        │   ├── class_1/          
        │   │   ├── yyy.pt
        │   │   ├── ...
        """
        self.root_dir = root_dir
        self.slides = []  
        self.slide_to_files = defaultdict(list)  

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith(".pt"):
                        file_path = os.path.join(class_path, file)
                        label = int(class_folder.split("_")[-1])  
                        match = re.search(r"\['(.*?)'\]\.pt", file)
                        if match:
                            clean_filename = match.group(1)  
                        else:
                            clean_filename = file  
                        slide_name = os.path.splitext(clean_filename)[0].split("_files")[0]
                        self.slide_to_files[slide_name].append((file_path, label))

        for slide_name, files in self.slide_to_files.items():
            if files:  
                label = files[0][1]
                self.slides.append((slide_name, files, label))

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        slide_name, files, label = self.slides[idx]

        all_patch_features = []
        file_names = []

        for file_path, _ in files:
            data = torch.load(file_path, weights_only=False)
            patches_features = data["patches_features"].squeeze(0) 
            for patch_idx in range(patches_features.shape[0]):
                all_patch_features.append(patches_features[patch_idx])
            file_names.append(os.path.basename(file_path))

        concatenated_features  = torch.stack(all_patch_features)
        concatenated_features = concatenated_features.unsqueeze(0)

        return {
            "patches_features": concatenated_features,  #  (1, patch_num, feature_dim)
            "label": label, 
            "slide_name": slide_name,  
            "file_names": file_names,  
            "seq_len": len(all_patch_features)
        }


def collate_slides(batch):
    return batch[0]


def load_test_data():
    test_feature_root = 'xxx/features/test'
    test_dataset = SlideBasedPTFeatureDataset(test_feature_root)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_slides
    )

    return test_loader


