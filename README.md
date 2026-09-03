# NEURO-SENSE-AI
A full-stack explainable AI medical platform leveraging fine-tuned VGG16 (91.4% accuracy), Grad-CAM heatmaps, and edge-deployed Llama 3.2 RAG for dementia diagnostic support

## Local Setup (Core Ultra 7 / RTX 4060 8GB)

### 1. GPU driver setup (do this first, it's the part people get stuck on)

Your GPU is an RTX 4060 (8GB). TensorFlow's GPU support on **native Windows was
dropped after version 2.10** — anything newer needs Linux or WSL2. Pick one:

- **Recommended: WSL2 (Ubuntu) on your Windows machine.**
  Install WSL2, then inside it install CUDA + `tensorflow[and-cuda]==2.15.*`
  from `requirements.txt`. This gets you full modern TF + GPU support.
- **Native Windows, no WSL:** pin `tensorflow==2.10.*` and install
  CUDA 11.2 + cuDNN 8.1 manually (the exact versions TF 2.10 needs). Everything
  in the training script still works on 2.10.
- **Alternative:** if you'd rather avoid this entirely, PyTorch has better
  native Windows CUDA support and I can give you a PyTorch version of this
  script instead — just ask.

Verify the GPU is visible before training:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If that prints an empty list, training will still run but on CPU (much slower,
still doable overnight for ~4000 images but not ideal).

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Unzip your dataset so you have this layout
```
Alzheimer_Balanced/
  NonDemented/
  VeryMildDemented/
  MildDemented/
  ModerateDemented/
```

### 4. Run training
```bash
python train_vgg16_alzheimer.py --data_dir ./Alzheimer_Balanced
```

This runs two phases automatically:
- **Phase 1** (~15 epochs): VGG16 convolutional base frozen, only the new
  classifier head trains. Fast, gets you to a reasonable baseline.
- **Phase 2** (~25 epochs): unfreezes just VGG16's last conv block
  (`block5`) and fine-tunes it at a much lower learning rate. This is where
  most of the accuracy gain happens.

Both phases use early stopping, so it'll stop itself once validation accuracy
plateaus rather than running the full epoch count.

On an RTX 4060, expect roughly 30-70 seconds/epoch at batch size 32 depending
on disk I/O — so well under an hour total.

### 5. What you'll get in `outputs/`
- `best_phase1.keras`, `best_finetuned.keras` — checkpoints
- `vgg16_alzheimer_final.keras` — final model
- Console output: classification report + confusion matrix on the **held-out
  test set**, which never saw any near-duplicate of a training image.

### On the 98% target — an honest note

This dataset's `ModerateDemented` class only has **64 truly unique scans**
(copied ~15x each to pad to 1000), and `MildDemented` has 896 unique scans.
That's a tiny amount of real Moderate-stage data for a model to generalize
from. With the leakage-safe split this script uses:

- You can realistically expect **~90-97% test accuracy** overall, with
  `ModerateDemented` likely being the weakest class simply because there's so
  little genuine variation for the model to learn from.
- If you instead see 98-100%, it's a signal something is leaking (e.g. a
  future change accidentally reintroduces a random split) rather than a
  sign the model is unusually good — worth being skeptical of any Alzheimer's-MRI
  notebook online that reports 99%+ without describing how they handled these
  duplicates.
- If test accuracy on `ModerateDemented` specifically is the bottleneck, the
  honest fix is more real Moderate-stage scans, not more augmentation of the
  same 64 images.

I built it this way so whatever number you get is one you can trust and
actually report.

## Running this in VS Code (Windows)

### 6. Folder layout
Put the data folder anywhere, e.g. `D:\Alzheimer_Balanced\`, with:
```
Alzheimer_Balanced/
  NonDemented/
  VeryMildDemented/
  MildDemented/
  ModerateDemented/
```

### 7. Create a virtual environment
Open the project folder in VS Code, then open a terminal (`` Ctrl+` ``):
```powershell
python -m venv venv
venv\Scripts\activate
```

### 8. Install PyTorch with CUDA (for your RTX 4060)
`requirements.txt` deliberately does NOT list torch/torchvision — installing it via plain
`pip install -r requirements.txt` pulls the CPU-only build from PyPI and silently overwrites
a CUDA build if you install it in the wrong order. Install torch first, from the CUDA index:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then install everything else (won't touch torch, since it's not in the file):
```powershell
pip install -r requirements.txt
```

Check GPU is detected:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True RTX 4060 Laptop GPU` (or similar).

#### If you already hit "Torch not compiled with CUDA enabled"
You have the CPU build installed. Fix it:
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then re-run the check command above before training.

### 9. Select the interpreter in VS Code
`Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.\venv\Scripts\python.exe`.

### 10. Run training
```powershell
python train_vgg16_alzheimer_torch.py --data_dir "D:\Alzheimer_Balanced"
```

### 11. What you get in `outputs/`
- `best_phase1.pt`, `best_finetuned.pt` — best checkpoints per phase
- `phase1_loss.png`, `phase1_accuracy.png` — phase 1 curves
- `finetuned_loss.png`, `finetuned_accuracy.png` — phase 2 curves
- `phase1_history.json`, `finetuned_history.json` — raw numbers if you want to replot
- `confusion_matrix.png` — test set confusion matrix heatmap
- `test_report.txt` — precision/recall/F1 report + confusion matrix as text
- `vgg16_alzheimer_final.pt` — final trained model weights

## Notes
- `num_workers=0` is already set for Windows, so `DataLoader` won't hang.
- Training runs in two phases automatically (frozen head, then fine-tune block5) — no need to run it twice.
- If VRAM runs out on the 8GB 4060, lower `BATCH_SIZE` in the script (e.g. 16).
