# Quishing Detection

This repository contains a deep learning project aimed at detecting malicious QR codes (Quishing - QR Phishing) using a Convolutional Neural Network (CNN) built with PyTorch.

## Overview

Quishing is a form of phishing where attackers use QR codes to direct victims to malicious websites or trick them into downloading malware. This project trains a binary classifier to distinguish between benign (safe) and malicious (phishing) QR codes based on their visual patterns.

## Dataset

The dataset consists of 9,987 grayscale QR code images (69x69 pixels).
- **Benign (Class 0)**: 5,005 samples
- **Malicious (Class 1)**: 4,982 samples

The pixel values are binary (0 or 1), and the dataset is well-balanced.

## Project Structure

- `sujjal.ipynb`: The main Jupyter Notebook containing the data loading, preprocessing, model definition, training loop, and evaluation visualizations.
- `description.md`: A detailed document explaining the architectural and design choices for the PyTorch CNN model.
- `requirements.txt`: Python dependencies required to run the project.
- `QuishingDataset/`: Directory containing the dataset as pickle files (`qr_codes_29.pickle` and `qr_codes_29_labels.pickle`).

## Setup Instructions

1. **Activate your virtual environment** (if not already activated):
   ```bash
   source .venv/bin/activate
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook sujjal.ipynb
   ```
   *Alternatively, you can open and run the notebook directly within your IDE (e.g., VS Code or Antigravity).*

## Model Architecture

The model is a custom PyTorch Convolutional Neural Network consisting of:
- **3 Convolutional Blocks**: Each block contains a `Conv2d` layer (3x3), `BatchNorm2d`, `ReLU` activation, and `MaxPool2d`. The channel depths progress from 32 -> 64 -> 128.
- **Classification Head**: Flattening followed by a dense layer with 128 neurons, Dropout (50%) for regularization, and a final linear layer outputting 1 logit.
- **Loss Function**: `BCEWithLogitsLoss` for stable binary classification.

For a comprehensive breakdown of the design decisions, refer to `description.md`.
