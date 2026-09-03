"""
Combines Phase 1 (frozen base) and Phase 2/fine-tuned history into ONE
continuous graph spanning all epochs (e.g. 15 + 35 = 50), with a dashed line
marking where Phase 2 begins - instead of two separate per-phase graphs.

Reads:
    outputs/phase1_history.json
    outputs/finetuned_history.json

Usage:
    python plot_full_training.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def main():
    p1_path = os.path.join(OUTPUT_DIR, "phase1_history.json")
    p2_path = os.path.join(OUTPUT_DIR, "finetuned_history.json")

    for p in (p1_path, p2_path):
        if not os.path.exists(p):
            print(f"Missing {p} - run training first.")
            return

    with open(p1_path) as f:
        h1 = json.load(f)
    with open(p2_path) as f:
        h2 = json.load(f)

    # Concatenate both phases into one continuous series
    train_loss = h1["train_loss"] + h2["train_loss"]
    val_loss = h1["val_loss"] + h2["val_loss"]
    train_acc = h1["train_acc"] + h2["train_acc"]
    val_acc = h1["val_acc"] + h2["val_acc"]

    total_epochs = len(train_loss)
    epochs = list(range(1, total_epochs + 1))
    phase_boundary = len(h1["train_loss"])  # last epoch of phase 1

    print(f"Phase 1 epochs: {len(h1['train_loss'])}")
    print(f"Phase 2 (fine-tuned) epochs: {len(h2['train_loss'])}")
    print(f"Total combined epochs: {total_epochs}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Loss panel ---
    ax1.plot(epochs, train_loss, label="Training Loss")
    ax1.plot(epochs, val_loss, label="Validation Loss")
    ax1.axvline(phase_boundary, color="gray", linestyle="--", linewidth=1,
                label="Phase 1 -> Phase 2")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (Full Training Run)")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # --- Accuracy panel ---
    ax2.plot(epochs, train_acc, label="Training Accuracy")
    ax2.plot(epochs, val_acc, label="Validation Accuracy")
    ax2.axvline(phase_boundary, color="gray", linestyle="--", linewidth=1,
                label="Phase 1 -> Phase 2")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy (Full Training Run)")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.suptitle(f"Training vs Validation - Full {total_epochs}-Epoch Run", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "full_training_combined.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
