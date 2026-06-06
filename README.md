# Image_Classification_With_LSNet
This repository implements LSNet (Large-Small Network) for image classification. LSNet is a lightweight vision architecture inspired by the human visual system's "See Large, Focus Small" strategy, combining large-kernel perception and small-kernel aggregation for efficient feature extraction and accurate classification.

## Overview

LSNet is a lightweight vision architecture inspired by the human visual system's **"See Large, Focus Small"** mechanism. It introduces **LS Convolution**, which combines:
- **Large-Kernel Perception (LKP):** Captures broad contextual information using large-kernel depthwise convolutions.
- **Small-Kernel Aggregation (SKA):** Dynamically aggregates local features using small-kernel convolutions.

This design enables efficient feature extraction while maintaining strong classification performance.

## Objectives

- Implement LSNet variants for image classification.
- Train and evaluate LSNet on CIFAR-10 and EMNIST datasets.
- Compare LSNet performance with ResNet-18.
- Analyze training behavior using accuracy and loss curves.
- Evaluate pretrained and custom-trained models.

## Features

- Implementation of LSNet architectures:
  - LSNet-T (Tiny)
  - LSNet-S (Small)
  - LSNet-B (Base)
- Baseline Model - ResNet-18
- CIFAR-10 training and evaluation pipeline
- EMNIST training and evaluation pipeline
- Early stopping support
- Automatic model checkpoint saving (`.pth`)
- Training log generation
- Training & validation loss visualization
- GPU acceleration with PyTorch
- Reproducible training setup

## Dataset

This project supports image classification on the following datasets:

### CIFAR-10
- 60,000 RGB images
- 10 object classes
- Image size: 32 × 32
- Standard benchmark for object classification

### EMNIST (Balanced)
- Extended version of the MNIST dataset
- Handwritten digits and letters
- Grayscale images of size 28 × 28
- Suitable for handwritten character recognition tasks

Datasets are automatically downloaded using Torchvision.

## Results

### Available Evaluation Outputs

| Dataset / Model | Evaluation Files |
|----------------|------------------|
| LSNet on CIFAR-10 | TXT + CSV |
| LSNet on EMNIST | TXT + CSV |
| Pretrained LSNet | TXT + CSV |
| ResNet-18 on CIFAR-10 | TXT + CSV |

### Training Curves Included

- LSNet-T Loss Curves (CIFAR-10 & EMNIST)
- LSNet-S Loss Curves (CIFAR-10 & EMNIST)
- LSNet-B Loss Curves (CIFAR-10 & EMNIST)
- ResNet-18 Accuracy Curve
- ResNet-18 Loss Curve

## Repository Structure

```
Curves/
├── LSNet_CIFAR_10/
├── LSNet_EMNIST/
└── ResNet_18_CIFAR_10/

evaluation_result_CIFAR_10/
evaluation_results_EMNIST/
evaluation_results_Pretrained_Images/
evaluation_results_ResNet18_CIFAR_10/

Training Scripts
Testing Scripts
Presentation
README.md
requirements.txt
```
## Performance Comparison

| Model | Dataset | Top-1 Accuracy (%) | Parameters (M) | FLOPs (G) |
|---------|---------|------------------:|---------------:|----------:|
| ResNet-18 | CIFAR-10 | 84.43 | 11.18 | 0.037 |
| LSNet-T | CIFAR-10 | 84.98 | 11.41 | 0.321 |
| LSNet-S | CIFAR-10 | 85.92 | 16.14 | 0.546 |
| **LSNet-B** | CIFAR-10 | **87.82** | 23.31 | 1.263 |
| LSNet-T | EMNIST | 87.32 | 15.75 | 0.035 |
| LSNet-S | EMNIST | 87.75 | 32.86 | 0.059 |
| **LSNet-B** | EMNIST | **88.44** | 96.59 | 0.157 |

## Performance Comparison

![Accuracy Comparison](Performance_Comparasion.png)

### Key Observations

- LSNet-B achieved the highest classification accuracy on both CIFAR-10 and EMNIST datasets.
- LSNet-T provides a lightweight alternative with competitive accuracy and lower computational requirements.
- The results demonstrate the effectiveness of LSNet's "See Large, Focus Small" strategy for efficient image classification.
  
## Reference

LSNet: See Large, Focus Small
Ao Wang, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding
CVPR 2025

Paper:
https://arxiv.org/abs/2503.23135

Official Repository:
https://github.com/jameslahm/lsnet
