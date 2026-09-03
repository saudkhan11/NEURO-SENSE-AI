"""
Regenerates the 3 result graphs (Phase 1 loss/accuracy are optional extras;
the required 3 are: fine-tuned loss, fine-tuned accuracy, confusion matrix)
from files already saved by train_vgg16_alzheimer_torch.py - no retraining
needed.

Reads:
    outputs/finetuned_history.json   (saved automatically during training)
    outputs/test_report.txt          (saved automatically after evaluation)

Usage:
    python plot_results.py
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


def plot_loss_accuracy(history, tag):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history["train_loss"], label="Training Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{tag} - Loss")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{tag}_loss_regenerated.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{tag} - Accuracy")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{tag}_accuracy_regenerated.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def parse_confusion_matrix_from_report(report_path):
    """Pulls the confusion matrix array back out of the saved test_report.txt
    (written via np.array2string), so we can replot it without rerunning
    evaluation."""
    with open(report_path) as f:
        content = f.read()
    match = re.search(r"Confusion matrix:\s*\[\[(.*?)\]\]", content, re.DOTALL)
    if not match:
        raise ValueError("Could not find confusion matrix in test_report.txt")
    rows_text = match.group(1).replace("]", "").replace("[", "").strip()
    rows = [r.split() for r in rows_text.split("\n") if r.strip()]
    cm = np.array(rows, dtype=int)
    return cm


def plot_confusion_matrix(cm):
    test_acc = np.trace(cm) / cm.sum()
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (Test Set) - Accuracy = {test_acc*100:.1f}%")
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
    out = os.path.join(OUTPUT_DIR, "confusion_matrix_regenerated.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    # --- Fine-tuned phase loss + accuracy ---
    ft_history_path = os.path.join(OUTPUT_DIR, "finetuned_history.json")
    if os.path.exists(ft_history_path):
        with open(ft_history_path) as f:
            ft_history = json.load(f)
        plot_loss_accuracy(ft_history, "finetuned")
    else:
        print(f"Missing {ft_history_path} - run training first.")

    # --- Confusion matrix ---
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")
    if os.path.exists(report_path):
        cm = parse_confusion_matrix_from_report(report_path)
        plot_confusion_matrix(cm)
    else:
        print(f"Missing {report_path} - run training first.")


if __name__ == "__main__":
    main()

