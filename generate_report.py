"""
Full pipeline: CNN prediction -> Grad-CAM -> RAG retrieval -> LLM report.

Architecture (important): the CNN is the validated classifier (88.72% test
accuracy, measured on a leakage-free split). The LLM does NOT re-diagnose or
override the CNN's prediction - it explains and contextualizes that specific
prediction using retrieved reference context, and gives general, non-medical
next-step suggestions. This keeps your one measured accuracy number
meaningful and avoids presenting an unvalidated LLM opinion as a diagnosis.

LLM provider: Ollama (local), vision-capable via llava.

Setup:
    Install Ollama: https://ollama.com/download
    Pull a vision model:
        ollama pull llava
    Make sure Ollama is running (it starts automatically after install,
    or run `ollama serve` manually). No API key needed - fully local.

Usage:
    python generate_report.py --image "Alzheimer_Balanced\\MildDemented\\26 (19).jpg"
"""

import os
import argparse
import base64

import requests
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

OLLAMA_MODEL = "llava"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# Model + Grad-CAM (same as before)
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


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
def call_ollama(prompt, image_path):
    """Calls a local Ollama server running a vision-capable model (e.g. llava)."""
    b64_image = image_to_base64(image_path)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        result = resp.json()
        return result["response"].strip(), None
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to Ollama. Is it running? Try: ollama serve"
    except Exception as e:
        detail = resp.text if "resp" in dir() else str(e)
        return None, f"Ollama error: {detail}"


def generate_report(prompt, image_path):
    """Ollama-only (local, vision-capable)."""
    text, err = call_ollama(prompt, image_path)
    if text:
        return text, "Ollama"
    return f"[Ollama failed: {err}]", "none"


# ---------------------------------------------------------------------------
# Prompt construction (with RAG context)
# ---------------------------------------------------------------------------
def build_prompt(predicted_class, probs_dict, rag_snippets):
    prob_lines = "\n".join(f"- {cls}: {p*100:.1f}%" for cls, p in probs_dict.items())
    context_lines = "\n".join(f"- {s['text']}" for s in rag_snippets)

    return f"""You are helping generate a plain-language research report from an MRI
classification model's output. This model is a research/screening aid, NOT a
diagnostic tool - a real diagnosis requires a qualified clinician.

CNN prediction (already computed, do not second-guess or re-predict this):
  Predicted class: {predicted_class}
  Confidence breakdown:
{prob_lines}

An image is attached: a Grad-CAM heatmap overlay showing which regions of the
scan most influenced the model's prediction (warm/red = higher influence).

Reference context (from a curated knowledge base on dementia staging):
{context_lines}

Write a short, structured report with these sections:
1. Result summary (2-3 sentences, plain language, state the predicted class and
   confidence)
2. What this stage generally means (2-3 sentences on typical symptoms/characteristics
   of this stage, using the reference context above)
3. Contributing factors (1-2 sentences on general risk factors associated with this
   stage, from the reference context - framed as general risk factors, not causes
   specific to this individual)
4. Observed attention region (1-2 sentences on what the Grad-CAM image shows)
5. General risk-reduction / care suggestions (2-3 sentences, using the reference
   context's prevention/management guidance - general and non-prescriptive, NOT
   specific medical advice or medication suggestions)
6. A clear one-line disclaimer that this is a research tool output, not a
   medical diagnosis, and a real diagnosis requires a qualified clinician.

Keep the whole report under 320 words. Plain text, no markdown headers -
just clearly separated short paragraphs."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(image_path):
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

    # ---- CNN prediction (this is the trusted, measured-accuracy step) ----
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    predicted_idx = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    probs_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    print(f"\nImage: {image_path}")
    print(f"CNN predicted class: {predicted_class}  (confidence: {probs[predicted_idx]*100:.2f}%)")
    for cls, p in sorted(probs_dict.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:20s}: {p*100:.2f}%")

    # ---- Grad-CAM ----
    gradcam = GradCAM(model)
    cam = gradcam.generate(x, predicted_idx)
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    heatmap_path = os.path.join(out_dir, "gradcam_" + os.path.basename(image_path).replace(" ", "_"))
    overlay_heatmap(original_img, cam, heatmap_path)
    print(f"Grad-CAM heatmap saved to: {heatmap_path}")

    # ---- RAG retrieval ----
    rag = SimpleRAG()
    query = f"{predicted_class} Alzheimer's dementia stage brain scan"
    snippets = rag.retrieve(query, predicted_class, top_k=4)
    print(f"\nRetrieved {len(snippets)} reference snippets for context.")

    # ---- LLM report ----
    prompt = build_prompt(predicted_class, probs_dict, snippets)
    print("\nGenerating report...")
    report, provider = generate_report(prompt, heatmap_path)

    print("\n" + "=" * 60)
    print(f"PROFESSIONAL REPORT (generated via {provider})")
    print("=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    main(args.image)
 