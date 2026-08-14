#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# On-VM environment setup for MedQwen-VL training (run ON the GCP VM).
# Assumes an NVIDIA L4 + a CUDA-capable base image (Deep Learning VM or similar).
#
#   git clone <repo> MedQwen && cd MedQwen
#   bash scripts/setup_vm.sh                  # normal (checks + fixes the driver)
#   bash scripts/setup_vm.sh --skip-gpu-check # install deps anyway (e.g. CPU box)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SKIP_GPU_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --skip-gpu-check) SKIP_GPU_CHECK=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

echo "== GPU check =="
if [ "$SKIP_GPU_CHECK" -eq 1 ]; then
  echo "skipping GPU check (--skip-gpu-check)"
elif nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi failed — driver not loaded."
  # Deep Learning VMs ship a driver installer; try it before giving up.
  if [ -x /opt/deeplearning/install-driver.sh ]; then
    echo "attempting driver (re)install via /opt/deeplearning/install-driver.sh ..."
    sudo /opt/deeplearning/install-driver.sh || true
  fi
  if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo ""
    echo "still no GPU driver. try one of:"
    echo "  - reboot, then re-run:          sudo reboot"
    echo "  - confirm a GPU is attached:    lspci | grep -i nvidia"
    echo "  - install deps without a GPU:   bash scripts/setup_vm.sh --skip-gpu-check"
    exit 1
  fi
fi

echo "== System prereqs (venv module + build basics) =="
# Ubuntu ships python3 without ensurepip/venv; install the matching -venv pkg.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
sudo apt-get update -qq
sudo apt-get install -yq "python3.${PYVER#*.}-venv" python3-pip tmux || \
  sudo apt-get install -yq python3-venv python3-pip tmux

echo "== Python venv =="
rm -rf .venv                      # clear any half-created venv from a prior run
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== Core deps (Qwen2.5-VL needs transformers>=4.49) =="
# Big CUDA wheels drop on flaky links; --resume-retries picks up partial
# downloads instead of restarting (needs pip >=25.1, ensured by upgrade above).
PIP_DL="--resume-retries 20 --timeout 180"
# torch+torchvision with CUDA wheels (cu121 works on L4 / driver 535+)
pip install $PIP_DL torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install $PIP_DL -r requirements.txt
pip install $PIP_DL -U "huggingface_hub[cli]"

echo "== Sanity =="
python -c "import torch; ok=torch.cuda.is_available(); print('cuda:', ok, torch.cuda.get_device_name(0) if ok else '(no GPU visible)')"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('VL class OK')"

echo "done. next: download the base model + data (see GCP_RUNBOOK.md)."
