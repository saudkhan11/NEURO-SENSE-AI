"""
Evaluate the saved VGG16 Alzheimer model and print accuracy.
Loads the trained weights from outputs/vgg16_alzheimer_final.pt and re-runs
the same leakage-safe test split used during training.

Usage:
    python check_accuracy.py --data_dir .\Alzheimer_Balanced
"""

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
SEED = 42
MODEL_PATH = os.path.join("outputs", "vgg16_alzheimer_final.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def base_id(filename):
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

    def mask_for(gs):
        return np.array([g in gs for g in groups])

    return (
        (paths[mask_for(train_groups)], labels[mask_for(train_groups)]),
        (paths[mask_for(val_groups)], labels[mask_for(val_groups)]),
        (paths[mask_for(test_groups)], labels[mask_for(test_groups)]),
    )


class AlzheimerDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), int(self.labels[idx])


def build_model(num_classes=len(CLASSES)):
    base = models.vgg16(weights=None)  # no need to redownload ImageNet weights, we load our own
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


def main(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No saved model found at {MODEL_PATH}. Run training first.")

    paths, labels, groups = collect_files(data_dir)
    _, _, (test_p, test_y) = grouped_split(paths, labels, groups)
    print(f"Test set size: {len(test_p)}")

    test_ds = AlzheimerDataset(test_p, test_y)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(lbls.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir)
