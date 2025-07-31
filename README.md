# H2R-Aug
Code repository for the paper **H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos**

This repository provides the official implementation of H2R, a human-to-robot data augmentation pipeline for bridging the visual domain gap in robot learning. H2R converts egocentric human videos (e.g., Ego4D, SSv2) into robot-centric data.


## Installation


This code has been tested and validated in the linux environment (Ubuntu 20.04, Python 3.10, CUDA 11.8). It is recommended to use a virtual environment (e.g., conda or venv) to avoid conflicts with other packages.

When cloning this repository, ensure submodules are initialized:
```bash
git clone --recurse-submodules git@github.com:log2r/H2R-Aug.git
```

If you already cloned without submodules, run:
```bash
git submodule update --init --recursive
```

Before installing Python dependencies, ensure the following system libraries are installed:

```bash
sudo apt update
sudo apt install libgl1-mesa-glx libglib2.0-0 libegl1-mesa libgles2-mesa libvulkan1 vulkan-utils
echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json" >> ~/.bashrc
```

Create a virtual environment and install the required Python packages:

```bash
conda create -n h2r python=3.10
conda activate h2r
pip install -r requirements.txt
```

Install third-party packages:
```bash
# Install HAMER and ViTPose
cd third_party/hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
cd ../..

# Install Inpaint-Anything
cd third_party/Inpaint-Anything/
pip install -e segment_anything
pip install -r lama/requirements.txt
cd ../..
```

## Prepare the Model Weights
First, download the hamer weights using the following bash command.

```bash
cd third_party/hamer
bash fetch_demo_data.sh
cd _DATA
wget https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
```
Besides these files, you also need to download the MANO model. Please visit the MANO website and register to get access to the downloads section. We only require the right hand model. You need to put MANO_RIGHT.pkl under the `third_party/hamer/_DATA/data/mano` folder.

Next, you can refer to the excerpt from the Inpaint-Anything README below and place the related Inpaint-Anything weights in: `/share/project/lyx/H2R-Aug/third_party/Inpaint-Anything/pretrained_models`

> Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [LaMa](./lama/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)), and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.





## Data Augmentation

You can directly use the `h2r.py` file to augment a single egocentric (first-person) image containing human hands:  
```bash
python h2r.py path/to/your/image.jpg
```

## Pretrain
- mae: https://github.com/facebookresearch/mae

- r3m: https://github.com/facebookresearch/r3m
## Evaluation
- Robomimic: https://github.com/ARISE-Initiative/robomimic
- DP: https://github.com/real-stanford/diffusion_policy
