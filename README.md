# Xray_Classification
Chest X-ray Classification using Deep Learning (Custom CNN &amp; Transfer Learning)
# Overview
This repository contains two deep learning approaches for classifying chest X-ray images:

- A **Custom Convolutional Neural Network (CNN)** built from scratch
- A **Transfer Learning model** using pretrained architectures like **VGG19** and **DenseNet201**

Both models aim to detect anomalies in X-ray scans, a crucial application in medical diagnostics using AI. The goal is to compare performance and training behavior between a handcrafted model and advanced pretrained networks.
## 🔍 Notebooks

- `notebooks/Xray_CNN.ipynb`: Defines and trains a custom CNN architecture for chest X-ray classification.
- `notebooks/Pre_Trained.ipynb`: Uses pretrained VGG19 and DenseNet201 architectures via transfer learning for the same task.

## 📁 Datasets
Due to size limitations, datasets are not included. You may download the X-ray dataset used from:
[Insert dataset source or link]

The structure should follow:
datasets/ ├── train/ │ NORMAL/ │ PNEUMONIA/ ├── test/ │ NORMAL/ │ PNEUMONIA/
## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/Xray-Classification.git
cd Xray-Classification
