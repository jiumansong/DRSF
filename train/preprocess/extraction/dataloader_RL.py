import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import cv2
import csv

def label_int(_label, _class_num, _class_cate):
    for i in range(_class_num):  
        if str(_label) == list(_class_cate)[i]:
            conv_label = i
            return conv_label


class PngDataset(Dataset):
    def __init__(self, csv_file, transform=None, class_cate=None, class_num=None, patch_size=224):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self._class_cate = class_cate
        self._class_number = class_num
        self.patch_size = patch_size 
        self.h5_cache = {}  

    def __len__(self):
        return len(self.data)

    def _cut_image_into_patches(self, img):
        height, width, _ = img.shape
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        patches = []
        relative_positions = []  

        for i in range(num_patches_x):
            for j in range(num_patches_y):

                patch = img[j * self.patch_size:(j + 1) * self.patch_size,
                        i * self.patch_size:(i + 1) * self.patch_size]
                patches.append(patch)
                relative_positions.append((i, j))
        patch_num = len(patches)

        return np.stack(patches), np.array(relative_positions), patch_num

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]

        slide_Name = f"{img_path.split(os.path.sep)[-2]}_{img_path.split(os.path.sep)[-1]}"
        label = self.data.iloc[idx, 1]
        _label = label_int(label, self._class_number, self._class_cate)

        h5_file_path = img_path.replace('.png', '.h5')  
        if not os.path.exists(h5_file_path):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  
                if img.shape[2] == 4:  
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)                         
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")

            patches_np, relative_positions_np, patch_num = self._cut_image_into_patches(img)

            with h5py.File(h5_file_path, 'w') as h5f:
                h5f.create_dataset('patches', data=patches_np)                         
                h5f.create_dataset('relative_positions', data=relative_positions_np)  
                h5f.attrs['patch_num'] = patch_num

        else:
            with h5py.File(h5_file_path, 'r') as h5f:
                patches_np = h5f['patches'][:]                                     
                relative_positions_np = h5f['relative_positions'][:] 
                patch_num = h5f.attrs['patch_num']

        if self.transform:
            patches_tensor = [self.transform(patch) for patch in patches_np]

            patches_tensor = torch.stack(patches_tensor)
        else:
            patches_tensor = torch.tensor(patches_np) 

        relative_positions_tensor = torch.tensor(relative_positions_np, dtype=torch.float32)  

        return {"patches": patches_tensor,
                "_label": _label,
                "slice_name": slide_Name,
                #"relative_positions": relative_positions_tensor,
                #"patch_num": patch_num
                }



def load_train_data(_class_cate, _class_number, batch_size):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_csv_path = 'path/train_path_label.csv'  # A CSV file was saved containing the paths and corresponding labels of tissue area images (png).
    train_dataset = PngDataset(csv_file=train_csv_path, transform=data_transforms, class_cate=_class_cate, class_num=_class_number)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)     #num_workers=0是单进程

    test_csv_path = 'path/test_path_label.csv'                     
    test_dataset = PngDataset(csv_file=test_csv_path, transform=data_transforms, class_cate=_class_cate, class_num=_class_number)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader



