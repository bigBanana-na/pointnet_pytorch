# PointNet PyTorch

This repository contains a PyTorch implementation of PointNet, a neural network architecture for point cloud classification and segmentation. The project is a kind of private learning project studied from [fxia22's PointNet PyTorch implementation](https://github.com/fxia22/pointnet.pytorch).

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Introduction
PointNet is a neural network that directly consumes point clouds, which are sets of 3D points. This implementation focuses on both classification and part segmentation tasks using the ShapeNet dataset.

## Project Structure
```plaintext
pointnet_pytorch/
├── dataset.py                # Data loading and processing
├── model.py                  # PointNet model definitions
├── visualization.py          # Script for visualizing results
├── train_segmentation.py     # Script for training all classes for segmentation
├── XXX.pth                   # Pre-trained model of different class for segmentation
├── README.md                 # Project documentation
```

## Acknowledgements
Have learned a lot from the PointNet PyTorch implementation by fxia22(https://github.com/fxia22/pointnet.pytorch). Special thanks to the authors for their great work and contributions to the community.
