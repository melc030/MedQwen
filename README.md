HnIt# MedQwen — Chinese Medical Q&A Fine-Tuning

LoRA fine-tuning of Qwen2.5-Instruct on 30K Chinese medical dialogue pairs, with multi-metric evaluation and a Gradio chatbot demo.

---

## Overview

This project fine-tunes Qwen2.5 (1.5B and 7B variants) on a Chinese medical Q&A dataset using parameter-efficient LoRA adapters. The goal is to adapt a general-purpose LLM to the medical domain while keeping training costs low — trained locally on Apple Silicon (M5, MPS) and on GCP (L4 GPU).

---

## Training Loss Curves

**Qwen2.5-1.5B** (Apple M5, MPS)
![Loss Curve 1.5B](loss_curve.png)

**Qwen2.5-7B** (GCP L4 GPU)
![Loss Curve 7B](loss_curve_7b.png)

## Results

### Qwen2.5-1.5B (trained on Apple M5)

| Metric | Base | Fine-tuned | Δ |
|--------|------|------------|---|
| ROUGE-1 | 0.0289 | 0.0050 | -2.39% |
| ROUGE-2 | 0.0050 | 0.0026 | -0.24% |
| ROUGE-L | 0.0289 | 0.0050 | -2.39% |
| **BERTScore** | **0.6059** | **0.6620** | **+5.61%** |

### Qwen2.5-7B (trained on GCP L4 GPU)

| Metric | Base | Fine-tuned | Δ |
|--------|------|------------|---|
| ROUGE-1 | 0.0407 | 0.0330 | -0.77% |
| ROUGE-2 | 0.0040 | 0.0024 | -0.16% |
| ROUGE-L | 0.0389 | 0.0330 | -0.59% |
| **BERTScore** | **0.5959** | **0.6670** | **+7.11%** |

> ROUGE scores are lower for the fine-tuned model because it learned to give concise, on-format answers rather than verbose outputs. BERTScore (semantic similarity) is the primary metric for open-ended Chinese generation — the 7B model shows **+7.11% semantic improvement**.


---

## Architecture

```
Data Pipeline (Producer-Consumer)
    ↓
raw .txt Q&A pairs → JSONL (Qwen chat template format)

Training
    ↓
Qwen2.5-Instruct (frozen) + LoRA adapters (trainable ~1% params)
FP16 mixed precision · gradient checkpointing · cosine LR schedule

Serving
    ├── MLX-LM server (Apple Silicon, port 8080)
    └── vLLM server   (CUDA GPU, port 8000)
         ↓
    Gradio chatbot UI (port 7860)

Evaluation
    ├── ROUGE (character-level tokenization for Chinese)
    └── BERTScore (bert-base-chinese)
```

---

## Tech Stack

- **Model**: Qwen2.5-1.5B / 7B-Instruct
- **Fine-tuning**: LoRA via PEFT (`r=8`, `alpha=16`, 7 target modules)
- **Training**: PyTorch, gradient accumulation, FP16 autocast
- **Hardware**: Apple M5 MPS (1.5B) · GCP L4 24GB (7B)
- **Serving**: MLX-LM (Apple Silicon) · vLLM (CUDA)
- **UI**: Gradio
- **Evaluation**: ROUGE (character-level), BERTScore (bert-base-chinese)

---

## Project Structure

```
MedQwen/
├── config.py              # centralized hyperparameters and paths
├── train.py               # LoRA fine-tuning loop
├── evaluate.py            # ROUGE + BERTScore evaluation
├── LLM-as-judge.py        # GPT-4o judge evaluation
├── inference_test.py      # qualitative base vs fine-tuned comparison
├── app.py                 # Gradio chatbot UI
├── serve/
│   ├── mlx_serve.py       # MLX-LM OpenAI-compatible server (Mac)
│   └── vllm_serve.py      # vLLM OpenAI-compatible server (GPU)
├── data_preprocess/
│   └── convert_data.py    # producer-consumer data pipeline
└── data/
    ├── medical_train.jsonl
    └── medical_valid.jsonl
```

---

## Setup

```bash
git clone https://github.com/melc030/MedQwen.git
cd MedQwen
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download base model

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
```

### Train

```bash
python train.py
```

### Evaluate

```bash
python evaluate.py
```

### Run chatbot

Two serving backends are supported — pick based on your hardware:

**Option A — Apple Silicon Mac (MLX-LM, port 8080)**
```bash
# fuse LoRA weights into base model first (one-time step)
python -m mlx_lm.fuse \
  --model Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/best \
  --save-path checkpoints/mlx-medqwen

# start server
python serve/mlx_serve.py
```

**Option B — Cloud GPU / CUDA (vLLM, port 8000)**
```bash
pip install vllm
python serve/vllm_serve.py
```

> vLLM requires CUDA — it will not run on Mac. For Mac, use MLX-LM.

Terminal 2 — launch Gradio UI (works with either server):
```bash
python app.py                                      # defaults to localhost:8080 (MLX)
INFERENCE_URL=http://localhost:8000 python app.py  # point at vLLM
INFERENCE_URL=http://<vm-ip>:8000 python app.py   # point at remote GPU VM
```

Open `http://localhost:7860`

---

## Trained Model

The fine-tuned 7B LoRA adapter is available on HuggingFace:
[mellee030/MedQwen-7B-LoRA](https://huggingface.co/mellee030/MedQwen-7B-LoRA)

---

## Dataset

30,284 training pairs and 412 validation pairs from a Chinese medical Q&A corpus, converted to Qwen2.5 chat template format (system / user / assistant).
