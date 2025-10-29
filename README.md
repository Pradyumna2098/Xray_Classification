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

## 📁 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset introduced by Kermany *et al.* (2018).

- Official dataset page: <https://data.mendeley.com/datasets/rscbjbr9sj/3>
- License and usage: released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Any use of the
  data must credit the original authors and follow the citation guidelines outlined on the dataset page.

### Directory structure

The training assets should be arranged as follows after extraction:

```
datasets/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 📥 Download helper script

Use the `scripts/download_data.py` utility to download the official archive and unpack it into the `datasets/` directory:

```bash
python scripts/download_data.py
```

Key options:

- `--output-dir`: customise where the dataset is extracted (defaults to `datasets/`).
- `--url`: override the download URL if you have mirrored the archive elsewhere.
- `--sha256`: provide a known hash to verify downloads (recommended when sharing archives).
- `--force`: re-download and overwrite an existing installation.

If the script detects a partial installation it aborts with a helpful message so that you can decide whether to rerun with
`--force` or clean the directory manually.
## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/Xray-Classification.git
cd Xray-Classification
```

2. (Optional) Download the dataset into `datasets/`:
```bash
python scripts/download_data.py
```
