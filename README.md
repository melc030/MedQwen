HnIt# MedQwen — Chinese Medical Q&A Fine-Tuning

LoRA fine-tuning of Qwen2.5-Instruct on 30K Chinese medical dialogue pairs, with multi-metric evaluation and a Gradio chatbot demo.

---

## Overview

This project fine-tunes Qwen2.5 (1.5B, 3B, and 7B variants) on a Chinese medical Q&A dataset using parameter-efficient LoRA adapters. The goal is to adapt a general-purpose LLM to the medical domain while keeping training costs low — trained locally on Apple Silicon (M5, MPS) and on GCP (L4 GPU).

---

## Demo

![MedQwen Gradio Chatbot](frontend.png)

> Gradio chatbot UI served via vLLM on GCP L4 GPU.

## Training Loss Curves

**Qwen2.5-1.5B** (Apple M5, MPS)
![Loss Curve 1.5B](loss_curve.png)

**Qwen2.5-3B r=8** (GCP L4 GPU)
![Loss Curve 3B](loss_curve_3b.png)

**Qwen2.5-7B** (GCP L4 GPU)
![Loss Curve 7B](loss_curve_7b.png)

---

## Results

### Model Scale Comparison (all r=8, lr=2e-4)

| Model | ROUGE-1 Δ | ROUGE-L Δ | BERTScore (base) | BERTScore (ft) | Δ |
|-------|-----------|-----------|-----------------|----------------|---|
| Qwen2.5-1.5B | -2.39% | -2.39% | 0.6059 | 0.6620 | +5.61% |
| Qwen2.5-3B   | +2.04% | +1.70% | 0.6159 | 0.6683 | +5.24% |
| Qwen2.5-7B   | -0.77% | -0.59% | 0.5959 | 0.6670 | +7.11% |

### LoRA Rank Ablation (Qwen2.5-3B, lr=2e-4)

| Rank | BERTScore (base) | BERTScore (ft) | Δ |
|------|-----------------|----------------|---|
| r=8  | 0.6159 | 0.6683 | +5.24% |
| r=16 | 0.6159 | 0.6751 | +5.92% |

> **Key findings:**
> - BERTScore (semantic similarity) is the primary metric for open-ended Chinese generation — ROUGE is less reliable as fine-tuned models learn to give concise, on-format answers rather than verbose outputs.
> - The 7B model achieves the highest absolute BERTScore improvement (+7.11%), consistent with larger models having more capacity to absorb domain-specific patterns.
> - Higher LoRA rank (r=16) outperforms r=8 on the 3B model (+5.92% vs +5.24%), showing that increased adapter expressiveness helps for this task — but only when learning rate is held constant.

---

## Architecture

```
Data Pipeline (Producer-Consumer)
    ↓
raw .txt Q&A pairs → JSONL (Qwen chat template format)

Training
    ↓
Qwen2.5-Instruct (frozen) + LoRA adapters (trainable ~1% params)
FP16 mixed precision · gradient checkpointing · cosine LR schedule · early stopping

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

- **Model**: Qwen2.5-1.5B / 3B / 7B-Instruct
- **Fine-tuning**: LoRA via PEFT (`r=8/16`, `alpha=2×rank`, 7 target modules)
- **Training**: PyTorch, gradient accumulation, FP16 autocast, early stopping
- **Hardware**: Apple M5 MPS (1.5B) · GCP L4 24GB (3B, 7B)
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
├── plot_loss.py           # training loss curve plotting
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
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
```

### Train

```bash
python train.py
```

### Evaluate

```bash
python evaluate.py
```

### Plot loss curve

```bash
python plot_loss.py training.log loss_curve.png
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

## Trained Models

| Model | HuggingFace |
|-------|-------------|
| Qwen2.5-7B LoRA (r=8) | [mellee030/MedQwen-7B-LoRA](https://huggingface.co/mellee030/MedQwen-7B-LoRA) |
| Qwen2.5-3B LoRA (r=8) | [mellee030/MedQwen-3B-LoRA](https://huggingface.co/mellee030/MedQwen-3B-LoRA) |
| Qwen2.5-3B LoRA (r=16) | [mellee030/MedQwen-3B-LoRA-r16](https://huggingface.co/mellee030/MedQwen-3B-LoRA-r16) |

---

## Dataset

30,284 training pairs and 412 validation pairs from a Chinese medical Q&A corpus, converted to Qwen2.5 chat template format (system / user / assistant).
