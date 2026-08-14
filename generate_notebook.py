import json

cells = []

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip().split("\n")})

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().split("\n")})

# --- Cell 1: Title ---
md("# Quishing Detection using PyTorch CNN\n\nThis notebook builds a CNN model using PyTorch to classify QR code images as **Benign (0)** or **Malicious (1)**.")

# --- Cell 2: Imports ---
code("""import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Import Colab drive module
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
""")

# --- Cell 3: Load Data ---
md("## 1. Mount Google Drive and Load Dataset")
code("""# Mount Google Drive if running in Google Colab
if IN_COLAB:
    drive.mount('/content/drive')
    # UPDATE THIS PATH IF YOUR REPOSITORY IS IN A DIFFERENT LOCATION IN YOUR DRIVE
    BASE_PATH = '/content/drive/MyDrive/Quishing_Detection/'
else:
    BASE_PATH = './'

# Load QR codes and labels from pickle files
with open(BASE_PATH + 'QuishingDataset/qr_codes_29.pickle', 'rb') as f:
    qr_codes = pickle.load(f)

with open(BASE_PATH + 'QuishingDataset/qr_codes_29_labels.pickle', 'rb') as f:
    qr_labels = pickle.load(f)

print("QR codes shape:", qr_codes.shape)
print("QR codes dtype:", qr_codes.dtype)
print("Labels shape:", qr_labels.shape)
print("Labels dtype:", qr_labels.dtype)""")

# --- Cell 4: Explore Data ---
md("## 2. Explore the Data")
code("""# Display 10 random QR code samples
indices = np.random.choice(len(qr_codes), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(qr_codes[indices[i]], cmap="gray")
    label_text = "Malicious" if qr_labels[indices[i]] == 1 else "Benign"
    ax.set_title(f"Label: {label_text}")
    ax.axis("off")
plt.tight_layout()
plt.show()""")

# --- Cell 5: Class Distribution ---
code("""# Check class distribution
unique, counts = np.unique(qr_labels, return_counts=True)
print("Class distribution:")
for label, count in zip(unique, counts):
    name = "Malicious" if label == 1 else "Benign"
    print(f"  Class {label} ({name}): {count} samples ({count / len(qr_labels) * 100:.2f}%)")

# Bar chart
plt.figure(figsize=(6, 4))
plt.bar(["Benign (0)", "Malicious (1)"], counts, color=["#2196F3", "#f44336"])
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()""")

# --- Cell 6: Preprocess ---
md("## 3. Preprocess the Data")
code("""# Convert to float32 numpy arrays
X = np.asarray(qr_codes, dtype="float32")
y = np.asarray(qr_labels, dtype="float32")

# Normalize if needed (pixels are already 0 or 1)
if X.max() > 1:
    X = X / 255.0

print(f"X shape: {X.shape}, range: [{X.min()}, {X.max()}]")
print(f"y shape: {y.shape}, unique: {np.unique(y)}")

# Verify binary labels
assert set(np.unique(y)).issubset({0, 1}), "Labels must be binary (0 and 1)"
print("Labels verified as binary.")""")

# --- Cell 7: Split ---
md("## 4. Split into Train / Validation / Test Sets")
code("""# 70% train, 15% validation, 15% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training:   {X_train.shape} | {y_train.shape}")
print(f"Validation: {X_val.shape}  | {y_val.shape}")
print(f"Testing:    {X_test.shape}  | {y_test.shape}")

# Verify stratification
for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    u, c = np.unique(labels, return_counts=True)
    print(f"  {name}: {dict(zip(u.astype(int), c))}")""")

# --- Cell 8: PyTorch Datasets ---
md("## 5. Create PyTorch DataLoaders")
code("""# Add channel dimension: (N, H, W) -> (N, 1, H, W) for PyTorch Conv2d
X_train_t = torch.tensor(X_train).unsqueeze(1)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val).unsqueeze(1)
y_val_t = torch.tensor(y_val).unsqueeze(1)
X_test_t = torch.tensor(X_test).unsqueeze(1)
y_test_t = torch.tensor(y_test).unsqueeze(1)

BATCH_SIZE = 32

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")
print(f"Input shape:   {X_train_t.shape}")""")

# --- Cell 9: Define Model ---
md("## 6. Define the CNN Model")
code("""class QRCodeCNN(nn.Module):
    def __init__(self):
        super(QRCodeCNN, self).__init__()
        
        # Conv Block 1: 1 -> 32 filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 2: 32 -> 64 filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Conv Block 3: 64 -> 128 filters
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
model = QRCodeCNN().to(device)
print(model)
print(f"\\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")""")

# --- Cell 10: Training Setup ---
md("## 7. Train the Model")
code("""# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
# Added weight_decay (L2 regularization) to penalize large weights and reduce overfitting
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
)

# Training configuration
NUM_EPOCHS = 50
PATIENCE = 7  # Early stopping patience

# History tracking
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_precision': [], 'val_precision': [],
    'train_recall': [], 'val_recall': [],
    'lr': []
}""")

# --- Cell 11: Training Loop ---
code("""def compute_metrics(y_true, y_pred):
    \"\"\"Compute accuracy, precision, recall from binary predictions.\"\"\"
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    total = len(y_true)
    correct = (y_pred == y_true).sum().item()
    
    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return acc, precision, recall

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    all_train_preds, all_train_labels = [], []
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_train_preds.append(preds.cpu())
        all_train_labels.append(y_batch.cpu())
    
    train_loss /= len(train_ds)
    all_train_preds = torch.cat(all_train_preds).squeeze()
    all_train_labels = torch.cat(all_train_labels).squeeze()
    t_acc, t_prec, t_rec = compute_metrics(all_train_labels, all_train_preds)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_val_preds, all_val_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_val_preds.append(preds.cpu())
            all_val_labels.append(y_batch.cpu())
    
    val_loss /= len(val_ds)
    all_val_preds = torch.cat(all_val_preds).squeeze()
    all_val_labels = torch.cat(all_val_labels).squeeze()
    v_acc, v_prec, v_rec = compute_metrics(all_val_labels, all_val_preds)
    
    # Record history
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)
    history['train_precision'].append(t_prec)
    history['val_precision'].append(v_prec)
    history['train_recall'].append(t_rec)
    history['val_recall'].append(v_rec)
    history['lr'].append(current_lr)
    
    # Step scheduler
    scheduler.step(val_loss)
    
    # Print progress
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {t_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {v_acc:.4f} | "
          f"LR: {current_lr:.1e}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\\nEarly stopping at epoch {epoch+1}!")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best model (val_loss: {best_val_loss:.4f})")""")

# --- Cell 12: Visualizations header ---
md("## 8. Training Visualizations")

# --- Cell 13: Accuracy plot ---
code("""# Training vs Validation Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Training vs Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

# --- Cell 14: Precision/Recall plot ---
code("""# Precision and Recall over epochs
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(history['train_precision'], label='Train Precision')
axes[0].plot(history['val_precision'], label='Val Precision')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision over Epochs')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_recall'], label='Train Recall')
axes[1].plot(history['val_recall'], label='Val Recall')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall over Epochs')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

# --- Cell 15: LR plot ---
code("""# Learning rate schedule
plt.figure(figsize=(8, 4))
plt.plot(history['lr'], marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()""")

# --- Cell 16: Evaluate on Test Set header ---
md("## 9. Evaluate on Test Set")

# --- Cell 17: Test evaluation ---
code("""# Get predictions on the test set
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    test_loss = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu())
        all_labels.append(y_batch.cpu())

test_loss /= len(test_ds)
y_prob = torch.cat(all_probs).squeeze().numpy()
y_true = torch.cat(all_labels).squeeze().numpy()
y_pred = (y_prob >= 0.5).astype(int)

test_acc = (y_pred == y_true).mean()
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")""")

# --- Cell 18: Classification report ---
code("""# Detailed classification report
print(classification_report(
    y_true.astype(int), y_pred,
    target_names=["Benign", "Malicious"],
    digits=4
))""")

# --- Cell 19: Confusion matrix ---
code("""# Confusion Matrix
cm = confusion_matrix(y_true.astype(int), y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title("CNN Confusion Matrix (Test Set)")
plt.show()""")

# --- Cell 20: ROC curve ---
code("""# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"CNN ROC-AUC = {roc_auc:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color='gray', alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN ROC Curve (Test Set)")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\\nROC-AUC Score: {roc_auc:.4f}")""")

# --- Cell 21: Sample predictions ---
md("## 10. Sample Predictions")
code("""# Show some test predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
indices = np.random.choice(len(X_test), 10, replace=False)

for i, ax in enumerate(axes.flat):
    idx = indices[i]
    ax.imshow(X_test[idx], cmap="gray")
    true_label = "Malicious" if int(y_true[idx]) == 1 else "Benign"
    pred_label = "Malicious" if y_pred[idx] == 1 else "Benign"
    prob = y_prob[idx]
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"True: {true_label}\\nPred: {pred_label} ({prob:.2f})",
                 color=color, fontsize=9)
    ax.axis("off")

plt.suptitle("Sample Test Predictions (Green=Correct, Red=Wrong)", fontsize=13)
plt.tight_layout()
plt.show()""")

# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("sujjal.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully: sujjal.ipynb")
