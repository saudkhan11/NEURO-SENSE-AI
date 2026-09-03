"""
NeuroAI API server
===================
Wraps the existing pipeline (VGG16 classifier + Grad-CAM + RAG + Ollama LLM)
in a small local Flask API so the neuroai_app.html frontend can talk to it.

This does NOT replace neuroai_assistant.py - it reuses the same model
loading / Grad-CAM / RAG / Ollama-calling logic, just exposed over HTTP
instead of a terminal menu.

Run this from your project root (same folder as neuroai_assistant.py,
config.py, rag_knowledge_base.py, and the outputs/ folder with your
trained .pt file):

    pip install flask flask-cors
    python server.py

Requires Ollama running locally (same as neuroai_assistant.py):
    ollama serve
    ollama pull llama3.2:3b

Then open neuroai_app.html in your browser (double-click it, or serve it -
either works since CORS is enabled below).
"""

import os
import io
import json
import uuid
import base64
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime, timezone
from bson import ObjectId

from rag_knowledge_base import SimpleRAG

# ---------------------------------------------------------------------------
# Config - kept identical to neuroai_assistant.py so behavior matches exactly
# ---------------------------------------------------------------------------
IMG_SIZE = 224
CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
MODEL_PATH = os.path.join("outputs", "vgg16_alzheimer_final.pt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    # Modern Chrome/Edge (Private Network Access) require this explicit
    # header before they'll let a file:// page fetch http://localhost.
    # Without it you get a generic, silent "Failed to fetch" in the browser
    # console with no request ever reaching this server.
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/<path:_>', methods=['OPTIONS'])
def cors_preflight(_):
    return ('', 204)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None  # loaded lazily on first request, cached after that
_RAG = SimpleRAG()

# in-memory chat sessions: session_id -> list of {"role":..., "content":...}
SESSIONS = {}


# ---------------------------------------------------------------------------
# MongoDB (patient records / analysis history)
# ---------------------------------------------------------------------------
# Point this at MongoDB Atlas by setting the MONGO_URI env var, e.g.:
#   setx MONGO_URI "mongodb+srv://user:pass@cluster.mongodb.net"   (Windows, permanent)
#   $env:MONGO_URI = "mongodb+srv://user:pass@cluster.mongodb.net" (Windows, current shell)
# If unset, this falls back to a local MongoDB instance (mongod running on
# localhost:27017, e.g. installed via MongoDB Community Server or Docker).
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "neurosense")

_mongo_client = None
_records_col = None
_mongo_error = None  # human-readable reason we couldn't connect, if any


def get_records_collection():
    """Lazily connect to MongoDB on first use. Returns None (and sets
    _mongo_error) if the database isn't reachable, so the rest of the app
    can keep running and the frontend can show a clear message instead of
    a raw stack trace."""
    global _mongo_client, _records_col, _mongo_error
    if _records_col is not None:
        return _records_col
    try:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        _mongo_client.admin.command("ping")  # fail fast if unreachable
        _records_col = _mongo_client[DB_NAME]["records"]
        _records_col.create_index([("createdAt", DESCENDING)])
        _mongo_error = None
    except ServerSelectionTimeoutError as e:
        _mongo_error = (f"Could not reach MongoDB at {MONGO_URI}. Is 'mongod' running "
                         f"locally, or is MONGO_URI set correctly for Atlas? ({e})")
        _records_col = None
    return _records_col


def serialize_record(doc):
    """Turn a Mongo document into plain JSON (ObjectId -> str, datetime -> ISO string)."""
    out = dict(doc)
    out["_id"] = str(out["_id"])
    if isinstance(out.get("createdAt"), datetime):
        out["createdAt"] = out["createdAt"].isoformat()
    return out


# ---------------------------------------------------------------------------
# Model + Grad-CAM (identical logic to neuroai_assistant.py / explain_prediction_vision.py)
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


def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No saved model found at '{MODEL_PATH}'. Run train_vgg16_alzheimer_torch.py first, "
                f"or run this server from the folder that contains the outputs/ directory."
            )
        m = build_model().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.eval()
        _MODEL = m
    return _MODEL


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


def overlay_heatmap(original_img, cam):
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(original_img.size)
    blended = Image.blend(original_img.convert("RGB"), heatmap_img, alpha=0.4)
    return blended


def image_to_data_uri(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def gradcam_focus_note(cam):
    """Same simple center-vs-periphery heuristic as explain_prediction_vision.py."""
    h, w = cam.shape
    center_region = cam[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()
    if center_region > cam.mean():
        return "Model attention was concentrated more centrally in the scan, consistent with medial structures."
    return "Model attention was spread toward the periphery of the scan."


# ---------------------------------------------------------------------------
# Ollama chat (identical to neuroai_assistant.py)
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
                f"Try 'ollama serve' in another terminal, and make sure the model is "
                f"pulled (e.g. 'ollama pull {model_name}'). Error: {e}]")


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
    return """Using the fixed prediction and reference knowledge already established in
this conversation, write a structured clinical report with these sections:
(1) Result summary, (2) Clinical correlate of this stage (CDR/MMSE range),
(3) Associated risk factors / etiology, (4) Observed Grad-CAM attention
region, (5) Management / risk-reduction considerations, (6) A one-line
disclaimer that this isn't a medical diagnosis. Keep it under 300 words,
plain text, no markdown headers, professional clinical tone."""


def clean_model_name(name):
    """Frontend sends things like 'llama3.2:3b (local)' - strip the label."""
    return name.split(" ")[0].strip() if name else "llama3.2:3b"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    model_ok = os.path.exists(MODEL_PATH)
    col = get_records_collection()
    return jsonify({
        "status": "ok",
        "model_found": model_ok,
        "device": str(DEVICE),
        "mongo_connected": col is not None,
        "mongo_error": _mongo_error,
    })


@app.route("/api/records", methods=["POST"])
def create_record():
    """Save one analysis result as a patient record in MongoDB."""
    col = get_records_collection()
    if col is None:
        return jsonify({"error": _mongo_error or "Database unavailable."}), 503

    data = request.get_json(force=True) or {}
    name = (data.get("name") or "Unnamed patient").strip()
    mrn = (data.get("mrn") or "\u2014").strip()
    stage = data.get("stage")
    badge = data.get("badge")
    confidence = data.get("confidence")

    if stage is None or confidence is None:
        return jsonify({"error": "Missing required fields 'stage' and/or 'confidence'."}), 400

    doc = {
        "name": name,
        "mrn": mrn,
        "stage": stage,
        "badge": badge,
        "confidence": float(confidence),
        "createdAt": datetime.now(timezone.utc),
    }
    result = col.insert_one(doc)
    doc["_id"] = result.inserted_id
    return jsonify(serialize_record(doc)), 201


@app.route("/api/records", methods=["GET"])
def list_records():
    """All saved records, newest first. Supports ?limit=N."""
    col = get_records_collection()
    if col is None:
        return jsonify({"error": _mongo_error or "Database unavailable."}), 503

    limit = request.args.get("limit", type=int)
    cursor = col.find().sort("createdAt", DESCENDING)
    if limit:
        cursor = cursor.limit(limit)
    return jsonify([serialize_record(d) for d in cursor])


@app.route("/api/records/latest", methods=["GET"])
def latest_record():
    """The single most recent analysis on file, for a dashboard 'last analysis' widget."""
    col = get_records_collection()
    if col is None:
        return jsonify({"error": _mongo_error or "Database unavailable."}), 503

    doc = col.find_one(sort=[("createdAt", DESCENDING)])
    if doc is None:
        return jsonify(None)
    return jsonify(serialize_record(doc))


@app.route("/api/records/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    col = get_records_collection()
    if col is None:
        return jsonify({"error": _mongo_error or "Database unavailable."}), 503
    try:
        oid = ObjectId(record_id)
    except Exception:
        return jsonify({"error": "Invalid record id."}), 400
    result = col.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"deleted": True})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected form field 'image')."}), 400

    file = request.files["image"]
    llm_model = clean_model_name(request.form.get("llm_model", "llama3.2:3b"))

    try:
        original_img = Image.open(file.stream).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(original_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    predicted_idx = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    probs_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    # Grad-CAM (needs its own forward+backward pass, done after the no_grad prediction above)
    gradcam = GradCAM(model)
    x_grad = tf(original_img).unsqueeze(0).to(DEVICE)
    cam = gradcam.generate(x_grad, predicted_idx)
    overlay = overlay_heatmap(original_img, cam)
    focus_note = gradcam_focus_note(cam)

    # RAG + LLM report
    query = f"{predicted_class} Alzheimer's dementia stage brain scan"
    snippets = _RAG.retrieve(query, predicted_class, top_k=4)
    system_prompt = build_system_prompt(predicted_class, probs_dict, snippets)
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": build_full_report_prompt(predicted_class, probs_dict, snippets)})
    report = call_ollama_chat(messages, llm_model)
    messages.append({"role": "assistant", "content": report})

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"messages": messages, "llm_model": llm_model}

    return jsonify({
        "predicted": predicted_class,
        "conf": {cls: round(p * 100, 1) for cls, p in probs_dict.items()},
        "original_image": image_to_data_uri(original_img),
        "gradcam_image": image_to_data_uri(overlay),
        "gradcam_note": focus_note,
        "report": report,
        "session_id": session_id,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Empty message."}), 400

    session = SESSIONS.get(session_id)
    if session is None:
        # No prior analysis this session - answer ungrounded, matching the
        # frontend's existing fallback behavior.
        messages = [{"role": "system", "content": "You are NeuroAI, a research assistant for an MRI dementia-stage classifier. No analysis has been run yet this session, so answer generally and suggest the user run an analysis for a grounded answer."}]
        llm_model = clean_model_name(data.get("llm_model", "llama3.2:3b"))
    else:
        messages = session["messages"]
        llm_model = session["llm_model"]

    messages.append({"role": "user", "content": message})
    reply = call_ollama_chat(messages, llm_model)
    messages.append({"role": "assistant", "content": reply})

    if session is not None:
        session["messages"] = messages

    return jsonify({"reply": reply})


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
    print("Starting NeuroAI API server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
