# H2R-Aug
Code repository for the paper **H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos**

## Installation

**Python 3.8+** and **CUDA 11.8**. Then follow the steps below:

### Core Dependencies
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.6.13
pip install pytorch-lightning==2.0.0
pip install mmcv==1.3.9
```
### HaMeR
```bash
cd third_party
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
```
### Inpainting Modules
```bash
cd third_party
git clone https://github.com/geekyutao/Inpaint-Anything.git
pip install -e segment_anything
pip install -r lama/requirements.txt
```
### Sapien Simulator & System Libraries(for Linux) 
```bash
pip install easydict==1.9.0 scikit-image albumentations==0.5.2 kornia==0.5.0 sapien==2.2.2 draccus
sudo apt install libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa libvulkan1 vulkan-utils
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=:1
```
## Data Augmentation
```bash
python h2r.py
```
## Pretrain
- mae: https://github.com/facebookresearch/mae

- r3m: https://github.com/facebookresearch/r3m
## Evaluation
- Robomimic: https://github.com/ARISE-Initiative/robomimic
- DP: https://github.com/real-stanford/diffusion_policy
