# Heyoh Object Detection / Tracking pipeline
Heyoh is an open source virtual camera based on PyTorch. It is a training and inference code for [heyoh-camera](https://github.com/heyoh-app/heyoh-camera) app. 

Contributors:
Marko Kostiv, Danylo Bondar, Oleh Sehelin, Ksenia Demska

PyTorch Annual Hackathon 2021

# Table of Contents
1. [Prerequisites](#prerequisites)
2. [Getting started](#getting-started)
3. [Training](#training)
4. [Usage](#usage-updated-for-apple-m1-support)
5. [Supplementary materials](#supplementary-materials)

## Prerequisites
- Linux or macOS
- CUDA 11.1
- [git-lfs](https://git-lfs.github.com/)

### Getting started
1. Create and activate conda environment:
```
conda create -n heyoh python=3.8 -y
conda activate heyoh
```
2. Install requirements:
```
pip3 install -r multitask_lightning/requirements.txt
```

### Training
1. Navigate to `multitask_lightning` directory
```
cd multitask_lightning/
```
2. Modify config file if needed:
```
nano configs/train_config.yaml
```
3. Run training:
```
python train.py configs/train_config.yaml
```

### Usage UPDATED FOR APPLE M1 SUPPORT
We developed a python debug application which processes frames similarly to [heyoh-camera](https://github.com/heyoh-app/heyoh-camera) app in Python. It is extremely useful for a faster prototyping and hyperparameters tuning.  
#### [HOW TO TEST](multitask_lightning/inference_test/README.md)
Guide on how to convert `pytorch_lightning` checkpoint to TorchScript and CoreML models and run the debug application.

### Supplementary materials
- [Sample dataset description](multitask_lightning/data/README.md)
- [Pre-trained checkpoint](multitask_lightning/checkpoints/epoch=126_val_loss=0_374_val_mAP=0_363.ckpt)
