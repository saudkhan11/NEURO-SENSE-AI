"""
Predict on an MRI image, generate a Grad-CAM heatmap showing WHERE the model
focused, and use a local LLM (via Ollama) to turn the result into a
readable, plain-language report.

Requires Ollama running locally (ollama.com) with a model pulled, e.g.:
    ollama pull llama3.2:3b

Usage:
    python explain_prediction.py --image path\to\scan.jpg
    python explain_prediction.py --image path\to\scan.jpg --llm_model llama3.1:8b
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

IMG_SIZE = 224
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
MODEL_PATH = os.path.join("outputs", "vgg16_alzheimer_final.pt")

# NVIDIA API catalog (build.nvidia.com) - vision-language model.
# Get a free key at build.nvidia.com, then set it as an environment variable
# (never hardcode it in the script):
#   PowerShell:  $env:NVIDIA_API_KEY = "nvapi-your-key-here"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"  # or meta/llama-3.2-11b-vision-instruct for speed


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Short, class-specific clinical context. This is general background info
# only, NOT a diagnosis - the LLM prompt reinforces this too.
CLASS_CONTEXT = {
    "NonDemented": "no signs of cognitive impairment on this scan",
    "VeryMildDemented": "very early, subtle signs sometimes associated with the earliest stage of cognitive decline",
    "MildDemented": "moderate structural changes sometimes associated with mild-stage cognitive decline",
    "ModerateDemented": "more pronounced structural changes sometimes associated with moderate-stage cognitive decline",
}


def build_model(num_classes=len(CLASSES)):
    base = models.vgg16(weights=None)
    # Disable inplace ReLU in the conv features - inplace ops conflict with
    # Grad-CAM's backward hooks (causes a "view is being modified inplace"
    # RuntimeError from autograd).
    for module in base.features:
        if isinstance(module, nn.ReLU):
            module.inplace = False

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
    """Grad-CAM on VGG16's last conv layer (features[28], the final conv
    before pooling) - shows which regions of the MRI most influenced the
    prediction."""

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

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global avg pool of gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_img, cam, save_path):
    heatmap = cm.jet(cam)[:, :, :3]  # drop alpha
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(original_img.size)
    blended = Image.blend(original_img.convert("RGB"), heatmap_img, alpha=0.4)
    blended.save(save_path)
    return save_path


def call_nvidia_vision(prompt, image_path, model_name):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return ("[No NVIDIA_API_KEY environment variable set. Get a free key at "
                "build.nvidia.com, then run: $env:NVIDIA_API_KEY = \"nvapi-your-key-here\"]")

    b64_image = image_to_base64(image_path)
    data_uri = f"data:image/png;base64,{b64_image}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.3,
        "top_p": 1,
        "stream": False,
    }

    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError:
        return f"[NVIDIA API error {response.status_code}: {response.text}]"
    except Exception as e:
        return f"[Could not reach NVIDIA API. Error: {e}]"


def build_prompt(predicted_class, probs_dict, cam_focus_note):
    prob_lines = "\n".join(f"- {cls}: {p*100:.1f}%" for cls, p in probs_dict.items())
    context = CLASS_CONTEXT[predicted_class]

    return f"""You are helping summarize the output of a research MRI classification model
(NOT a diagnostic tool) for a student's project report. An image is attached showing
a Grad-CAM heatmap overlay on the original MRI scan (red/warm areas = regions the
model focused on most for its prediction). Look at the attached image, then write
3-4 plain-language sentences summarizing the result below, including a brief comment
on where the highlighted (warm-colored) attention regions appear to be located in the
scan. Be factual and measured, avoid alarming language, and explicitly note this is a
research model output, not a medical diagnosis, and any real concern should be
discussed with a doctor.

Predicted class: {predicted_class}
General context for this class: {context}

Confidence breakdown:
{prob_lines}

Automated focus note (for reference, describe what you actually see in the image too): {cam_focus_note}

Write the summary now, plain text, no headers or markdown."""


def main(image_path, llm_model):
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
    x.requires_grad_(False)

    # ---- Prediction ----
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    predicted_idx = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    probs_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}  (confidence: {probs[predicted_idx]*100:.2f}%)")
    for cls, p in sorted(probs_dict.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:20s}: {p*100:.2f}%")

    # ---- Grad-CAM ----
    gradcam = GradCAM(model)
    cam = gradcam.generate(x, predicted_idx)

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    heatmap_path = os.path.join(out_dir, "gradcam_" + os.path.basename(image_path).replace(" ", "_"))
    overlay_heatmap(original_img, cam, heatmap_path)
    print(f"\nGrad-CAM heatmap saved to: {heatmap_path}")

    # Rough note on focus concentration (center vs periphery) - simple,
    # not anatomically precise, just descriptive for the LLM prompt.
    h, w = cam.shape
    center_region = cam[h//4:3*h//4, w//4:3*w//4].mean()
    periphery_region = cam.mean() - center_region if cam.mean() > center_region else 0
    if center_region > cam.mean():
        cam_focus_note = "The model's attention was concentrated more centrally in the scan."
    else:
        cam_focus_note = "The model's attention was spread toward the periphery of the scan."

    # ---- LLM report (vision model looks at the Grad-CAM image directly) ----
    prompt = build_prompt(predicted_class, probs_dict, cam_focus_note)
    print(f"\nGenerating report with NVIDIA vision model '{llm_model}'...")
    report = call_nvidia_vision(prompt, heatmap_path, llm_model)

    print("\n" + "=" * 60)
    print("LLM-GENERATED REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)
    print("\nNote: this is a research model output, not a medical diagnosis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default=NVIDIA_VISION_MODEL,
                         help="NVIDIA API model name, e.g. meta/llama-3.2-90b-vision-instruct")
    args = parser.parse_args()
    main(args.image, args.llm_model)
