# GCP Runbook — MedQwen-VL (Qwen2.5-VL-3B LoRA) Training

End-to-end runbook for fine-tuning the multimodal model on a single GCP **L4 (24 GB)** GPU,
the same class of machine used for the text model. Covers provisioning → data → model →
train → eval → retrieve → shutdown.

> Fits on one L4: 3B in bf16 (~6 GB) + LoRA + frozen ViT, with gradient checkpointing,
> `batch=1`, and the `max_pixels` token cap in `config.py`.

---

## 0. Prerequisites (local, one-time)

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# L4 quota: ensure "NVIDIA L4 GPUs" > 0 in your region
# Console → IAM & Admin → Quotas → filter "L4"  (request increase if 0)
```

Pick a region with L4 stock, e.g. `us-central1`, `us-east1`, `europe-west4`.

---

## 1. Provision the VM

```bash
export ZONE=us-central1-a
export VM=medqwen-vl

gcloud compute instances create $VM \
  --zone=$ZONE \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu123-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True"
```

- `g2-standard-8` = 1×L4, 8 vCPU, 32 GB RAM.
- The Deep Learning image auto-installs the NVIDIA driver on first boot (give it ~2 min).
- 150 GB disk: image (~50 GB) + base model (~7 GB) + images (~1 GB) + checkpoints.

SSH in:

```bash
gcloud compute ssh $VM --zone=$ZONE
```

---

## 2. Clone repo + install deps (on VM)

```bash
git clone <YOUR_REPO_URL> MedQwen && cd MedQwen
bash scripts/setup_vm.sh          # venv + CUDA torch + requirements, with sanity checks
source .venv/bin/activate
```

`setup_vm.sh` prints `cuda: True ...` and `VL class OK` if the environment is good.

---

## 3. Get the data onto the VM

The training jsonl references images by relative path (`images/...`). Bundle them
**locally** first, then copy the self-contained `data/multimodal/` folder straight
to the VM with `scp` — no extra storage service and no per-GB bucket charges.

**Local machine:**
```bash
# build splits AND copy the referenced images into data/multimodal/images/
python src/data/build_vqa.py --copy-images

tar czf mm.tgz -C data multimodal           # ~1 GB (jsonl + 18,751 images)

# copy directly to the VM over SSH (primary route — no bucket needed)
gcloud compute scp mm.tgz $VM:~/MedQwen/ --zone=$ZONE
```

**On VM:**
```bash
cd ~/MedQwen
mkdir -p data && tar xzf mm.tgz -C data       # -> data/multimodal/{train,val,test}.jsonl + images/
```

Because images now live at `data/multimodal/images/`, the default `cfg.images_root`
(`data/multimodal`) just works — no env var needed.

> **Backup route — GCS bucket.** Only if `scp` is slow or flaky on your connection.
> A bucket can incur small storage + egress charges, so prefer `scp` above.
> ```bash
> gsutil mb -l us-central1 gs://<YOUR_BUCKET>     # one-time
> gsutil cp mm.tgz gs://<YOUR_BUCKET>/            # from local
> gsutil cp gs://<YOUR_BUCKET>/mm.tgz .           # on the VM
> ```

---

## 4. Download the base model (on VM)

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct
```

This populates the path `cfg.vl_model_path` expects (`Qwen2.5-VL-3B-Instruct/`).

---

## 5. Preflight (30 sec — catch config/memory issues before a long run)

```bash
python -c "
import sys; sys.path.insert(0,'src')
import train_vl as T
m, p = None, None
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
proc = AutoProcessor.from_pretrained(T.cfg.vl_model_path, min_pixels=T.cfg.vl_min_pixels, max_pixels=T.cfg.vl_max_pixels)
mdl  = Qwen2_5_VLForConditionalGeneration.from_pretrained(T.cfg.vl_model_path, dtype=T.DTYPE, device_map='auto')
print('LoRA targets:', len(T.llm_lora_targets(mdl)))   # >0 and excludes visual.*
print('loaded OK on', T.cfg.device)
"
```

Expect a non-zero LoRA target count (the live module tree confirms `visual.*` is excluded).

---

## 6. Train

Long run — use `tmux` (or `nohup`) so it survives an SSH drop, and tee to a log.

```bash
tmux new -s train
python src/train_vl.py 2>&1 | tee logs/training_vl.log
#   detach: Ctrl-b then d        reattach: tmux attach -t train
```

- Watch the first few `step ... | loss ... | ETA ...` lines to confirm loss is decreasing
  and the ETA is acceptable (2 epochs over ~52k pairs is multi-hour on an L4;
  drop `vl_epochs` to 1 in config.py if you want a shorter first run).
- Best adapter auto-saves to `checkpoints/best-vl-3b/` on eval improvement; early stopping
  guards against overfitting. Ctrl-C is safe — the best checkpoint is already on disk.
- Monitor the GPU in another pane: `watch -n2 nvidia-smi`. If it OOMs, lower
  `vl_max_pixels` in `config.py` (fewer visual tokens) and restart.

---

## 7. Evaluate

```bash
# fine-tuned model
python src/eval/eval_vl.py --out eval_vl_3b.json

# zero-shot baseline, for the lift story
python src/eval/eval_vl.py --base --out eval_vl_3b_base.json

# quick subset while iterating
python src/eval/eval_vl.py --limit 300
```

Reports disease accuracy / **macro-F1** / weighted-F1, per-modality breakdown,
5-way modality-ID accuracy, ICD-10 exact + set-F1, and UMLS CUI exact-match.

---

## 8. Retrieve results

```bash
# pull the adapter + metrics back to your local machine
gcloud compute scp --recurse --zone=$ZONE \
  $VM:~/MedQwen/checkpoints/best-vl-3b ./checkpoints/
gcloud compute scp --zone=$ZONE $VM:~/MedQwen/eval_vl_3b.json .

# (optional) publish the adapter to HuggingFace
hf upload <user>/MedQwen-VL-3B-LoRA checkpoints/best-vl-3b
```

---

## 9. Shut down (avoid idle GPU charges)

```bash
gcloud compute instances stop $VM --zone=$ZONE      # keep disk, stop billing for GPU/CPU
# or, when fully done:
gcloud compute instances delete $VM --zone=$ZONE
```

> An L4 VM bills by the hour while **running**. Stop it the moment training/eval finishes.

---

## Quick reference

| Step | Command |
|------|---------|
| create VM | `gcloud compute instances create ...` (§1) |
| setup | `bash scripts/setup_vm.sh` |
| data | local `build_vqa.py --copy-images` + tar + `gcloud compute scp`; VM `tar xzf` |
| model | `hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct` |
| train | `python src/train_vl.py \| tee logs/training_vl.log` |
| eval | `python src/eval/eval_vl.py --out eval_vl_3b.json` |
| stop | `gcloud compute instances stop $VM --zone=$ZONE` |
