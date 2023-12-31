# PolyPooling: An Accurate Polyp Segmentation from Colonoscopy Images
> **Authors:** 
> Hoang Long Nguyen, 
> Dinh Cong Nguyen

This repository contains the official Pytorch implementation of training & evaluation code for PolyPooling.

## Overview

![Results](images/architecture.png "Results")

## Environment

- Requirement `CUDA 11.1` and `pytorch 1.7.1`

### Dataset

Downloading necessary data:

- Download testing dataset and move it into `./data/TestDataset/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view).
- Download training dataset and move it into `./data/TrainDataset/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view).

### Training

Download PoolFormer's pretrained `weights`
(
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)
) on ImageNet-1K, and put them in a folder `checkpoints/`.
Config hyper-parameters in `configs/custom.yaml` and run `train.py` for training. For example:

```
python train.py
```

### Evaluation

For evaluation, specific your backbone version, weight's path and dataset and run `val.py`. For example:

```
python val.py
```
### Results

![Results](images/results.png "Results")