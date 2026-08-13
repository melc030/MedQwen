#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# On-VM environment setup for MedQwen-VL training (run ON the GCP VM).
# Assumes an NVIDIA L4 + a CUDA-capable base image (Deep Learning VM or similar).
#
#   git clone <repo> MedQwen && cd MedQwen
#   bash scripts/setup_vm.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "== GPU check =="
nvidia-smi || { echo "no GPU / driver — use a CUDA image"; exit 1; }

echo "== Python venv =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== Core deps (Qwen2.5-VL needs transformers>=4.49) =="
# torch+torchvision with CUDA wheels (cu121 works on L4 / driver 535+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"

echo "== Sanity =="
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('VL class OK')"

echo "done. next: download the base model + data (see GCP_RUNBOOK.md)."
