"""
Plots fine-tuned Training vs Validation Loss AND Accuracy together in a
single combined figure (two panels side by side), instead of two separate
images - easier to drop into a paper as one figure.

Reads:
    outputs/finetuned_history.json

Usage:
    python plot_combined.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def main():
    history_path = os.path.join(OUTPUT_DIR, "finetuned_history.json")
    if not os.path.exists(history_path):
        print(f"Missing {history_path} - run training first.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_acc"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Loss panel ---
    ax1.plot(epochs, history["train_loss"], label="Training Loss")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Fine-tuned - Loss")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # --- Accuracy panel ---
    ax2.plot(epochs, history["train_acc"], label="Training Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Fine-tuned - Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.suptitle("Fine-tuned Phase: Training vs Validation", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "finetuned_combined.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
