# Quishing Detection CNN — Model Architecture & Design Choices

## Overview

This document describes all design decisions made for the **PyTorch CNN** model that classifies QR code images as **Benign (0)** or **Malicious/Phishing (1)**.

---

## 1. Dataset Summary

| Property | Value |
|---|---|
| Total samples | 9,987 |
| Image dimensions | 69 × 69 pixels, grayscale |
| Pixel value range | 0 or 1 (binary) |
| Class 0 (Benign) | 5,005 samples (~50.1%) |
| Class 1 (Malicious) | 4,982 samples (~49.9%) |
| Format | Pickle files (`qr_codes_29.pickle`, `qr_codes_29_labels.pickle`) |

The dataset is **nearly perfectly balanced**, so no class-weighting or oversampling is needed.

---

## 2. Data Preprocessing

- **Normalization**: Pixel values are already binary (0 or 1), so they are simply cast to `float32`. No `/255` scaling is needed.
- **Channel dimension**: Images are expanded from `(H, W)` to `(1, H, W)` to provide the single grayscale channel that PyTorch Conv2d expects (channels-first format).
- **Label type**: Labels are converted to `float32` for use with `BCEWithLogitsLoss`.

---

## 3. Train / Validation / Test Split

| Split | Percentage | Approximate Size |
|---|---|---|
| Training | 70% | ~6,990 |
| Validation | 15% | ~1,498 |
| Test | 15% | ~1,499 |

- **Stratified splitting** is used to maintain the ~50/50 class balance in every split.
- `random_state=42` ensures reproducibility.
- The validation set is used for monitoring during training (early stopping, learning rate scheduling). The test set is held out and only used for final evaluation.

---

## 4. CNN Architecture

The model uses **3 convolutional blocks** followed by **2 fully connected layers**.

### Convolutional Blocks

| Block | Filters | Kernel | Padding | Activation | Pooling | BatchNorm |
|---|---|---|---|---|---|---|
| Conv Block 1 | 32 | 3×3 | same | ReLU | MaxPool 2×2 | Yes |
| Conv Block 2 | 64 | 3×3 | same | ReLU | MaxPool 2×2 | Yes |
| Conv Block 3 | 128 | 3×3 | same | ReLU | MaxPool 2×2 | Yes |

### Why 3 convolutional blocks?
- QR codes are 69×69 — relatively small images with high-frequency binary patterns. Three blocks progressively reduce spatial dimensions (69→34→17→8) while increasing the receptive field enough to capture meaningful patterns without over-reducing the feature maps.
- Each block doubles the filter count (32→64→128), allowing the network to learn increasingly complex features.

### Why 3×3 kernels with `same` padding?
- 3×3 is the standard modern choice (inspired by VGGNet). It provides a good balance between receptive field size and parameter efficiency. Two stacked 3×3 convolutions cover the same receptive field as a single 5×5 but with fewer parameters and more non-linearities.
- `same` padding preserves spatial dimensions before pooling, making architecture reasoning easier.

### Why Batch Normalization?
- BatchNorm normalizes activations between layers, which:
  - Stabilizes and accelerates training
  - Acts as a mild regularizer
  - Allows using higher learning rates
- It is placed **after** the convolution and **before** ReLU activation, which is a common and effective pattern.

### Why MaxPool 2×2?
- MaxPooling with stride 2 halves the spatial dimensions, reducing computation and providing translation invariance. After 3 pooling operations: 69→34→17→8.

### Classification Head

| Layer | Neurons | Activation | Dropout |
|---|---|---|---|
| Flatten | — | — | 30% |
| Dense 1 | 128 | ReLU | 50% |
| Dense 2 (output) | 1 | Sigmoid (via BCEWithLogitsLoss) | — |

### Why 128 neurons in the dense layer?
- 128 provides sufficient capacity to learn the decision boundary for binary classification without being so large that it overfits. The flattened feature vector from the conv blocks is 128×8×8 = 8,192 dimensions, so 128 neurons represents a ~64× compression, which forces the network to learn a compact representation.

### Why Dropout (30% after conv, 50% in FC)?
- **30% after the final conv block**: Provides regularization in the feature extraction stage without being too aggressive (we want the conv filters to learn stable features).
- **50% in the FC layer**: Standard for fully connected layers. The FC layer has the most parameters (8,192×128 = 1,048,576), making it the most prone to overfitting. 50% dropout is a well-established default that works well in practice.

### Why single output neuron with BCEWithLogitsLoss?
- For binary classification, a single output with sigmoid activation is more parameter-efficient than two outputs with softmax. `BCEWithLogitsLoss` combines sigmoid + binary cross-entropy in a numerically stable way.

---

## 5. Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Adaptive learning rates per-parameter; works well out-of-the-box |
| Initial learning rate | 0.001 | Standard starting point for Adam |
| Loss function | BCEWithLogitsLoss | Numerically stable binary cross-entropy |
| Batch size | 32 | Good balance between gradient noise (helps generalization) and training speed |
| Max epochs | 50 | Upper bound; early stopping will typically halt training earlier |
| Early stopping patience | 7 | Allows the model enough epochs to recover from temporary validation loss increases while preventing significant overfitting |
| LR scheduler | ReduceLROnPlateau | Reduces LR by factor 0.5 when validation loss plateaus for 3 epochs; min LR = 1e-7 |

### Why Adam optimizer?
- Adam combines the benefits of momentum (SGD with momentum) and adaptive learning rates (RMSProp). It converges faster than vanilla SGD and requires less hyperparameter tuning.

### Why ReduceLROnPlateau?
- Rather than using a fixed schedule, this adapts to the actual training dynamics. When validation loss stops improving, the learning rate is halved, allowing the optimizer to fine-tune in a smaller neighborhood.

---

## 6. Evaluation Metrics

The model is evaluated on the **held-out test set** using:

1. **Accuracy** — Overall correct predictions / total predictions
2. **Precision** — Of all predicted positives, how many are truly positive
3. **Recall** — Of all actual positives, how many were correctly identified
4. **F1-Score** — Harmonic mean of precision and recall
5. **Confusion Matrix** — Shows true/false positives/negatives
6. **ROC Curve & AUC** — Plots true positive rate vs false positive rate across all thresholds; AUC summarizes overall discriminative ability
7. **Training/Validation Loss & Accuracy Curves** — Show learning progress and detect overfitting

---

## 7. Visualizations

| Visualization | Purpose |
|---|---|
| Sample QR code images with labels | Understand what the data looks like |
| Class distribution bar chart | Verify dataset balance |
| Training vs Validation Accuracy plot | Detect overfitting (diverging curves) |
| Training vs Validation Loss plot | Monitor convergence |
| Confusion Matrix heatmap | See per-class performance |
| ROC Curve with AUC score | Evaluate threshold-independent performance |

---

## 8. Total Model Parameters

Approximate parameter count:

| Component | Parameters |
|---|---|
| Conv Block 1 (32 filters) | ~320 + 64 (BN) |
| Conv Block 2 (64 filters) | ~18,496 + 128 (BN) |
| Conv Block 3 (128 filters) | ~73,856 + 256 (BN) |
| FC Layer 1 (8192→128) | ~1,048,704 |
| FC Layer 2 (128→1) | ~129 |
| **Total** | **~1,141,953** |

The model is compact enough to train on CPU in reasonable time (~2-5 minutes) while having sufficient capacity for the task.

---

## 9. Framework Choice: PyTorch

PyTorch was chosen because:
- **Dynamic computation graphs** make debugging easier
- **Explicit training loop** gives full control over the training process
- **Industry standard** for research and increasingly for production
- **Rich ecosystem** (torchvision, tensorboard integration, etc.)

---

## 10. Expected Performance

Based on the reference TensorFlow implementation achieving ~82% accuracy, the PyTorch CNN with BatchNorm improvements is expected to achieve **80-85% accuracy** on the test set. QR codes are inherently challenging for pixel-level CNNs because malicious vs benign QR codes may differ only in the encoded URL, which affects only a subset of the binary pattern.
