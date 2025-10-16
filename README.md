# RL-Mamba for WSI Classification
# Reinforcement Learning-based Whole Slide Image Discriminative Region Sampling and Classification Framework
## Installation
create a conda env and install these packages:
```
conda create -n rl-mamba python=3.10 -y
conda activate rl-mamba
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
Install causal_conv1d and mamba respectively:
```
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```
Suppose you have downloaded and extracted the source package:
```
python setup.py build install
```
| package | version | package | version |
|:------------:|:------------:|:------------:|:------------:|
| openslide-python |4.7.0 | numpy | 1.26.2 |
| setuptools | 75.1.0 | seaborn | 0.13.2 |
| matplotlib | 3.8.2 | opencv-python | 4.8.1 |
| timm | 0.4.12 | pandas | 2.1.3 | 
| scikit-learn | 1.3.2 | h5py | 3.12.1 |
| pillow | 10.1.0 | ninja | 1.11.1 |
| scipy |1.11.4 | tqdm | 4.66.1 |

## Dataset 
The dataset of the TCGA-SARC project can be download at <https://portal.gdc.cancer.gov/>.

Internal datasets involve patient privacy and are not considered for public release for the time being.

## Data preprocessing
Handle whole-slide images (WSIs) in SVS or MRXS format, determine the downsampling rate:
```
cd preprocess/handle_svs
python tile.py
```

## Feature extraction
```
cd preprocess/extraction
python extraction.py 
```

## Training and WSI prediction
```
python train.py --epochs EPOCH --lr LR --num_classes NUM --batch_size BATCH_SIZE --device DEVICE  --patch_budget BUDGET mamba_depth --DEPTH 
```
Code for visualizing intermediate results will be available soon.
