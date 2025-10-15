import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader_RL import load_train_data  
from extraction.resnet50_pretrainedTCGA_output1024 import PatchEmbed_ResNet50_PretrainedTCGA 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

batch_size = 1  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_root = "oot directory where features are saved" 

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trainloader, testloader = load_train_data(_class_cate=['0', '1', '2', '3', '4'], _class_number=5, batch_size=batch_size)

model = PatchEmbed_ResNet50_PretrainedTCGA().to(device)
model.eval()  

for sample in testloader:
    patches = sample["patches"].to(device)  
    label = sample["_label"].item()  
    slice_name = sample["slice_name"]  

    with torch.no_grad():
        patches_features = model(patches)  

    class_folder = os.path.join(save_root, f"class_{label}")  
    os.makedirs(class_folder, exist_ok=True)
    save_path = os.path.join(class_folder, f"{slice_name}.pt")

    torch.save({
        "patches_features": patches_features.cpu(),
        "label": label
    }, save_path)

    print(f"Saved: {save_path}")

print("Feature extraction complete.")
