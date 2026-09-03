"""
NeuroAI - an assistant built on top of your VGG16 dementia-stage classifier.

Flow:
  1. Predicts the class + generates Grad-CAM + retrieves RAG context (once).
  2. Prints a full initial report.
  3. Shows a menu of quick follow-up options (clinical presentation,
     etiology, management, explain the prediction, etc).
  4. Picking "Something else" drops into free-form chat, grounded in the
     same prediction + knowledge base context, with full conversation memory.

The CNN's prediction is always treated as fixed ground truth - NeuroAI
explains and discusses it, it never re-diagnoses or contradicts the
classifier.

Requires Ollama running locally with a model pulled, e.g.:
    ollama pull llama3.2:3b

Usage:
    python neuroai_assistant.py --image "Alzheimer_Balanced\\MildDemented\\26 (19).jpg"
    python neuroai_assistant.py --image "..." --llm_model llama3.1:8b
"""

import os
import json
import random
import argparse
import urllib.request
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, just saves to file
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm

from rag_knowledge_base import SimpleRAG

IMG_SIZE = 224
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
MODEL_PATH = os.path.join("outputs", "vgg16_alzheimer_final.pt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

BANNER = r"""
  _   _                    ___    ___
 | \ | | ___ _   _ _ __ ___/ _ \  |_ _|
 |  \| |/ _ \ | | | '__/ _ \ | | |  | |
 | |\  |  __/ |_| | | | (_) |_| |  | |
 |_| \_|\___|\__,_|_|  \___/\___/ |___|

  Neuroimaging-Based Dementia Stage Assistant (research tool - not a diagnosis)
"""

MENU_OPTIONS = {
    "1": "Explain the classification basis (Grad-CAM attention/activation mapping)",
    "2": "Clinical presentation associated with this stage",
    "3": "Associated risk factors and etiology for this stage",
    "4": "Management and risk-reduction considerations for this stage",
    "5": "Regenerate the full structured clinical report",
    "6": "Ask a custom clinical question",
    "7": "Try another random image",
    "8": "Upload / browse for your own image",
    "9": "Exit",
}


# ---------------------------------------------------------------------------
# Model + Grad-CAM
# ---------------------------------------------------------------------------
def build_model(num_classes=len(CLASSES)):
    base = models.vgg16(weights=None)
    for module in base.features:
        if isinstance(module, nn.ReLU):
            module.inplace = False  # required for Grad-CAM backward hooks

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


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.features[28]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_img, cam, save_path):
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(original_img.size)
    blended = Image.blend(original_img.convert("RGB"), heatmap_img, alpha=0.4)
    blended.save(save_path)
    return save_path


# ---------------------------------------------------------------------------
# Ollama chat
# ---------------------------------------------------------------------------
def call_ollama_chat(messages, model_name):
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["message"]["content"].strip()
    except Exception as e:
        return (f"[Could not reach Ollama at {OLLAMA_CHAT_URL} - is it running? "
                f"Try 'ollama serve' in another terminal. Error: {e}]")


# ---------------------------------------------------------------------------
# Prediction + Grad-CAM + RAG setup
# ---------------------------------------------------------------------------
def run_prediction(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No saved model found at {MODEL_PATH}. Run training first.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    original_img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = tf(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    predicted_idx = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    probs_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    gradcam = GradCAM(model)
    cam = gradcam.generate(x, predicted_idx)
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    heatmap_path = os.path.join(out_dir, "gradcam_" + os.path.basename(image_path).replace(" ", "_"))
    overlay_heatmap(original_img, cam, heatmap_path)

    return predicted_class, probs_dict, heatmap_path


def build_system_prompt(predicted_class, probs_dict, rag_snippets):
    prob_lines = "\n".join(f"- {cls}: {p*100:.1f}%" for cls, p in probs_dict.items())
    context_lines = "\n".join(f"- {s['text']}" for s in rag_snippets)

    return f"""You are NeuroAI, a research assistant discussing the output of an MRI-based
dementia-stage classifier. This is a research/screening tool, NOT a diagnostic
system - never claim to diagnose, and note that a real diagnosis requires a
qualified clinician when relevant.

Fixed context for this conversation (do not re-predict or contradict this):
  CNN predicted class: {predicted_class}
  Confidence breakdown:
{prob_lines}

Reference knowledge base:
{context_lines}

Respond in a concise, clinically-worded manner (3-6 sentences unless asked
for more detail) suitable for a physician audience, grounded in the context
and reference knowledge above. If asked something clearly outside this
scope, answer normally as a helpful assistant. If asked for a diagnosis or
specific treatment decision, redirect to a qualified clinician."""


def build_full_report_prompt(predicted_class, probs_dict, rag_snippets):
    prob_lines = "\n".join(f"- {cls}: {p*100:.1f}%" for cls, p in probs_dict.items())
    context_lines = "\n".join(f"- {s['text']}" for s in rag_snippets)
    return f"""Using the fixed prediction and reference knowledge already established in
this conversation, write a structured clinical report with these sections:
(1) Result summary, (2) Clinical correlate of this stage (CDR/MMSE range),
(3) Associated risk factors / etiology, (4) Observed Grad-CAM attention
region, (5) Management / risk-reduction considerations, (6) A one-line
disclaimer that this isn't a medical diagnosis. Keep it under 300 words,
plain text, no markdown headers, professional clinical tone."""


def pick_random_image(data_dir="Alzheimer_Balanced"):
    """Pick a random image from a random class folder inside data_dir."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    all_images = []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(cls_dir, fname))

    if not all_images:
        raise FileNotFoundError(f"No images found under {data_dir}")

    return random.choice(all_images)


def pick_file_dialog():
    """Opens a native Windows file picker and returns the selected path,
    or None if the user cancels."""
    root = tk.Tk()
    root.withdraw()          # hide the empty tkinter window
    root.attributes("-topmost", True)  # bring the dialog to the front
    path = filedialog.askopenfilename(
        title="Select an MRI image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


def generate_severity_graph(predicted_class, probs_dict, image_path):
    """Saves a two-panel figure: a bar chart of class confidence, and a
    horizontal severity scale (like a pH scale) showing where the predicted
    stage falls on the Non -> VeryMild -> Mild -> Moderate spectrum."""
    stage_order = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    stage_labels = ["Non-\nDemented", "Very Mild\nDemented", "Mild\nDemented", "Moderate\nDemented"]
    stage_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]  # green -> red, like a pH/severity scale
    predicted_idx = stage_order.index(predicted_class)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})

    # ---- Panel 1: confidence bar chart ----
    values = [probs_dict[cls] * 100 for cls in stage_order]
    bars = ax1.bar(stage_labels, values, color=stage_colors, edgecolor="black", linewidth=0.5)
    bars[predicted_idx].set_edgecolor("black")
    bars[predicted_idx].set_linewidth(2.5)
    ax1.set_ylabel("Confidence (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Prediction Confidence by Class")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%",
                  ha="center", va="bottom", fontsize=9)

    # ---- Panel 2: severity scale gauge ----
    for i, color in enumerate(stage_colors):
        ax2.axvspan(i, i + 1, color=color, alpha=0.85)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax2.set_xticklabels(["Non-\nDemented", "Very Mild", "Mild", "Moderate"], fontsize=9)
    ax2.set_title("Severity Scale - Predicted Stage Marker")

    # Marker (triangle) pointing at the predicted stage's position
    marker_x = predicted_idx + 0.5
    ax2.plot(marker_x, 0.5, marker="v", markersize=22, color="black", zorder=5)
    ax2.plot(marker_x, 0.5, marker="v", markersize=16, color="white", zorder=6)

    fig.suptitle(f"NeuroAI Prediction: {predicted_class}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "severity_graph_" + os.path.basename(image_path).replace(" ", "_") + ".png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    return graph_path


# ---------------------------------------------------------------------------
# Menu-driven interaction
# ---------------------------------------------------------------------------
def print_menu():
    print("\n" + "-" * 60)
    print("What would you like to do?")
    for key, label in MENU_OPTIONS.items():
        print(f"  [{key}] {label}")
    print("-" * 60)


def run_session(image_path, llm_model, data_dir):
    """Runs one full prediction + report + menu session. Returns
    (result, next_image_path) where result is 'restart' or 'exit'."""
    if image_path is None:
        print("No image specified. [R]andom image or [B]rowse for your own?")
        pick = input("Choose (R/B): ").strip().lower()
        if pick == "b":
            picked = pick_file_dialog()
            if picked:
                image_path = picked
            else:
                print("No file selected, falling back to a random image.")
                image_path = pick_random_image(data_dir)
        else:
            image_path = pick_random_image(data_dir)
        print(f"\nUsing image: {image_path}\n")

    print("Running prediction + Grad-CAM + retrieving reference context...\n")
    predicted_class, probs_dict, heatmap_path = run_prediction(image_path)

    print(f"Image: {image_path}")
    print(f"CNN predicted class: {predicted_class}")
    for cls, p in sorted(probs_dict.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:20s}: {p*100:.2f}%")
    print(f"Grad-CAM heatmap saved to: {heatmap_path}")

    graph_path = generate_severity_graph(predicted_class, probs_dict, image_path)
    print(f"Severity graph saved to: {graph_path}")
    try:
        os.startfile(os.path.abspath(graph_path))  # Windows-only: opens in default image viewer
    except Exception:
        pass  # non-Windows or no default viewer - user can still open the file manually

    rag = SimpleRAG()
    query = f"{predicted_class} Alzheimer's dementia stage brain scan"
    snippets = rag.retrieve(query, predicted_class, top_k=4)

    system_prompt = build_system_prompt(predicted_class, probs_dict, snippets)
    messages = [{"role": "system", "content": system_prompt}]

    print("\nGenerating initial report...")
    messages.append({"role": "user", "content": build_full_report_prompt(predicted_class, probs_dict, snippets)})
    report = call_ollama_chat(messages, llm_model)
    messages.append({"role": "assistant", "content": report})

    print("\n" + "=" * 60)
    print("NEUROAI CLINICAL REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)

    while True:
        print_menu()
        choice = input("Choose an option (1-9): ").strip()

        if choice == "9":
            print("\nNeuroAI: Take care. Remember to consult a qualified clinician for any real concerns.")
            return "exit", None

        elif choice == "7":
            return "restart", None

        elif choice == "8":
            print("\nOpening file picker window (check your taskbar if you don't see it)...")
            picked = pick_file_dialog()
            if picked:
                return "restart", picked
            else:
                print("No file selected, returning to menu.")
                continue

        elif choice == "6":
            print("\nFree chat mode - type your question, or 'menu' to go back, 'exit' to quit.")
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "menu":
                    break
                if user_input.lower() in ("exit", "quit"):
                    print("\nNeuroAI: Take care. Remember to consult a qualified clinician for any real concerns.")
                    return "exit", None
                if not user_input:
                    continue
                messages.append({"role": "user", "content": user_input})
                print("NeuroAI: ", end="", flush=True)
                reply = call_ollama_chat(messages, llm_model)
                print(reply)
                messages.append({"role": "assistant", "content": reply})

        elif choice in MENU_OPTIONS:
            question = MENU_OPTIONS[choice]
            messages.append({"role": "user", "content": question})
            print(f"\nNeuroAI: ", end="", flush=True)
            reply = call_ollama_chat(messages, llm_model)
            print(reply)
            messages.append({"role": "assistant", "content": reply})

        else:
            print("\nNot a valid option, please choose a number from the menu.")


def main(image_path, llm_model, data_dir):
    print(BANNER)
    result, picked_path = run_session(image_path, llm_model, data_dir)
    # "restart" with a picked_path means the user browsed for a specific
    # file; "restart" with picked_path=None means "try another random image"
    while result == "restart":
        print("\n" + "#" * 60 + "\n")
        result, picked_path = run_session(picked_path, llm_model, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a specific MRI image. If omitted, a random one is picked.")
    parser.add_argument("--data_dir", type=str, default="Alzheimer_Balanced",
                         help="Dataset folder to pick a random image from (used when --image is omitted)")
    parser.add_argument("--llm_model", type=str, default="llama3.2:3b",
                         help="Ollama model name, e.g. llama3.2:3b or llama3.1:8b or llava:latest")
    args = parser.parse_args()
    main(args.image, args.llm_model, args.data_dir)
