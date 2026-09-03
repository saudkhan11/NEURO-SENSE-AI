"""
VGG16 Transfer Learning for Alzheimer's MRI 4-Class Classification (PyTorch)
==============================================================================
Classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented

IMPORTANT NOTE ON THIS DATASET:
This "balanced" version was created by duplicating/augmenting images so every
class has 1000 samples. ModerateDemented has only ~64 unique source scans
(copied ~15x) and MildDemented has ~896 unique scans. A plain random
train/test split will leak near-duplicate copies of the same scan into both
train and test, inflating accuracy artificially. This script splits by
ORIGINAL IMAGE (grouped split) so all copies of one scan stay on the same
side of the split, giving a trustworthy accuracy number.

Hardware target: Core Ultra 7, 16GB RAM, RTX 4060 8GB VRAM.
Two-phase transfer learning:
  Phase 1: freeze VGG16 conv base, train new classifier head.
  Phase 2: unfreeze last conv block, fine-tune at low LR.

Usage:
    python train_vgg16_alzheimer_torch.py --data_dir ./Alzheimer_Balanced
"""

import os
import re
import copy
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")  # safe for headless / non-interactive runs
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SEED = 42

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 35
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
PATIENCE = 8  # early stopping (raised since phase 2 now trains more layers/epochs)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. Leakage-safe file list + grouped train/val/test split
# ---------------------------------------------------------------------------
def base_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return re.sub(r"_copy_\d+$", "", stem)


def collect_files(data_dir):
    paths, labels, groups = [], [], []
    for class_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            paths.append(os.path.join(cls_dir, fname))
            labels.append(class_idx)
            groups.append(f"{cls}__{base_id(fname)}")
    return np.array(paths), np.array(labels), np.array(groups)


def grouped_split(paths, labels, groups, val_frac=0.15, test_frac=0.15, seed=SEED):
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test:n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val:])

    def mask_for(group_set):
        return np.array([g in group_set for g in groups])

    train_mask = mask_for(train_groups)
    val_mask = mask_for(val_groups)
    test_mask = mask_for(test_groups)

    return (
        (paths[train_mask], labels[train_mask]),
        (paths[val_mask], labels[val_mask]),
        (paths[test_mask], labels[test_mask]),
    )


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class AlzheimerDataset(Dataset):
    def __init__(self, paths, labels, training):
        self.paths = paths
        self.labels = labels
        if training:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.08, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")  # replicate grayscale -> 3ch
        img = self.tf(img)
        label = int(self.labels[idx])
        return img, label


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
def build_model(num_classes=len(CLASSES)):
    base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for p in base.features.parameters():
        p.requires_grad = False  # phase 1: frozen

    base.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return base


def set_block5_trainable(model):
    """Unfreeze block4 AND block5 of VGG16 (features indices 17 onward) for
    fine-tuning. block4 alone (index 17-23) plus block5 (24-30) gives the
    model more capacity to separate NonDemented vs VeryMildDemented, which
    is where nearly all the remaining errors come from."""
    for i, layer in enumerate(model.features):
        if i >= 17:  # block4_conv1 onward
            for p in layer.parameters():
                p.requires_grad = True


# ---------------------------------------------------------------------------
# 4. Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(training)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def plot_history(history, tag):
    """Save training/validation loss + accuracy curves as PNGs (like the
    Colab plots) into OUTPUT_DIR."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history["train_loss"], label="Training Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{tag} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_loss.png"))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{tag} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{tag}_accuracy.png"))
    plt.close()

    # Raw numbers, so you can re-plot later without retraining
    with open(os.path.join(OUTPUT_DIR, f"{tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2)


def train_phase(model, train_loader, val_loader, epochs, lr, device, tag):
    # label_smoothing=0.05 softens the targets slightly -- keeps the model
    # from getting overconfident, which helps most on the ambiguous
    # NonDemented/VeryMildDemented boundary.
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS_TENSOR.to(device), label_smoothing=0.05)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{tag}] Epoch {epoch}/{epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, os.path.join(OUTPUT_DIR, f"best_{tag}.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[{tag}] Early stopping at epoch {epoch}")
                break

    plot_history(history, tag)
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
CLASS_WEIGHTS_TENSOR = None  # set inside main()


def main(data_dir):
    global CLASS_WEIGHTS_TENSOR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    paths, labels, groups = collect_files(data_dir)
    print(f"Total images: {len(paths)} | unique groups (real scans): {len(np.unique(groups))}")

    (train_p, train_y), (val_p, val_y), (test_p, test_y) = grouped_split(paths, labels, groups)
    print(f"Train: {len(train_p)}  Val: {len(val_p)}  Test: {len(test_p)}")

    train_ds = AlzheimerDataset(train_p, train_y, training=True)
    val_ds = AlzheimerDataset(val_p, val_y, training=False)
    test_ds = AlzheimerDataset(test_p, test_y, training=False)

    num_workers = 4 if os.name != "nt" else 0  # Windows multiprocessing w/ DataLoader can be finicky; keep 0 there
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    classes_present = np.unique(train_y)
    weights = compute_class_weight(class_weight="balanced", classes=classes_present, y=train_y)
    CLASS_WEIGHTS_TENSOR = torch.zeros(len(CLASSES))
    for c, w in zip(classes_present, weights):
        CLASS_WEIGHTS_TENSOR[c] = w
    print("Class weights (train set only):", CLASS_WEIGHTS_TENSOR.tolist())

    model = build_model().to(device)

    print("\n===== PHASE 1: training classifier head (base frozen) =====")
    model = train_phase(model, train_loader, val_loader, PHASE1_EPOCHS, PHASE1_LR, device, tag="phase1")

    print("\n===== PHASE 2: fine-tuning block5 =====")
    set_block5_trainable(model)
    model = train_phase(model, train_loader, val_loader, PHASE2_EPOCHS, PHASE2_LR, device, tag="finetuned")

    print("\n===== FINAL EVALUATION ON TEST SET (leakage-free, TTA) =====")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            # Test-time augmentation: average softmax probs of the image
            # and its horizontal flip. Usually a free +1-2% accuracy since
            # MRI orientation doesn't change the diagnosis.
            probs = torch.softmax(model(imgs), dim=1)
            probs_flipped = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            avg_probs = (probs + probs_flipped) / 2
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            y_true.extend(lbls.numpy())
            y_pred.extend(preds)

    test_acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test accuracy: {test_acc:.4f}")
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    cm = confusion_matrix(y_true, y_pred)
    print(report)
    print("Confusion matrix:\n", cm)

    # Save confusion matrix as a heatmap image
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # Save the text report too
    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vgg16_alzheimer_final.pt"))
    print(f"\nSaved final model + plots + report to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to folder containing NonDemented/, VeryMildDemented/, "
                              "MildDemented/, ModerateDemented/ subfolders")
    args = parser.parse_args()
    main(args.data_dir)
