import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re


class PTFeatureDataset(Dataset):
    def __init__(self, root_dir):
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
    train_feature_root = 'xxx/train'
    train_dataset = PTFeatureDataset(train_feature_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)     #num_workers=0是单进程
    return train_loader


class SlideBasedPTFeatureDataset(Dataset):
    def __init__(self, root_dir):
        """
        For ensemble aggregation strategy evaluation
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

        all_patches_features = []
        file_names = []

        for file_path, _ in files:
            data = torch.load(file_path, weights_only=False)
            patches_features = data["patches_features"].squeeze(0)  
            all_patches_features.append(patches_features)
            file_names.append(os.path.basename(file_path))

        stacked_features = torch.stack(all_patches_features)

        return {
            "patches_features": stacked_features,  
            "label": label,  
            "slide_name": slide_name,  
            "file_names": file_names 
        }


def collate_slides(batch):
    return batch[0]


def load_test_data():
    test_feature_root = 'xxx/test'
    test_dataset = SlideBasedPTFeatureDataset(test_feature_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_slides
    )

    return test_loader


