"""
Plots the fine-tuned Training vs Validation accuracy curve, PLUS a marker
showing where the final test accuracy (with TTA) lands - so it's visually
clear that the reported 91%+ test result sits above the validation curve,
rather than looking inconsistent with it.

Reads:
    outputs/finetuned_history.json
    outputs/test_report.txt

Usage:
    python plot_accuracy_with_test_marker.py
"""

import os
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def get_test_accuracy(report_path):
    with open(report_path) as f:
        content = f.read()
    match = re.search(r"Test accuracy:\s*([\d.]+)", content)
    if not match:
        raise ValueError("Could not find 'Test accuracy:' in test_report.txt")
    return float(match.group(1))


def main():
    history_path = os.path.join(OUTPUT_DIR, "finetuned_history.json")
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")

    if not os.path.exists(history_path):
        print(f"Missing {history_path} - run training first.")
        return
    if not os.path.exists(report_path):
        print(f"Missing {report_path} - run training first.")
        return

    with open(history_path) as f:
        history = json.load(f)
    test_acc = get_test_accuracy(report_path)

    epochs = list(range(1, len(history["train_acc"]) + 1))
    final_epoch = epochs[-1]

    plt.figure(figsize=(8.5, 6))
    plt.plot(epochs, history["train_acc"], label="Training Accuracy", color="#1f77b4")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", color="#ff7f0e")

    # Marker for final TTA test accuracy - placed just past the last epoch
    marker_x = final_epoch + 1
    plt.scatter([marker_x], [test_acc], color="green", s=110, zorder=5,
                marker="*", label=f"Final Test Accuracy (TTA) = {test_acc*100:.1f}%")
    plt.axhline(test_acc, color="green", linestyle=":", linewidth=1, alpha=0.6)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Fine-tuned Phase: Training vs Validation Accuracy\n(with Final Held-Out Test Accuracy)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "accuracy_with_test_marker.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
    print(f"Final validation accuracy: {history['val_acc'][-1]*100:.2f}%")
    print(f"Final test accuracy (TTA): {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()