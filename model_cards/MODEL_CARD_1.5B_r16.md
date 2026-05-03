---
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
- zh
license: apache-2.0
tags:
- medical
- chinese
- lora
- qwen2.5
- fine-tuned
---

# MedQwen-1.5B-LoRA-r16

LoRA fine-tuned [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on ~25K Chinese medical Q&A dialogue pairs with rank-16 adapters and a proper 80/10/10 train/val/test split.

> Supersedes [mellee030/MedQwen-1.5B-LoRA](https://huggingface.co/mellee030/MedQwen-1.5B-LoRA) — improved data split, higher LoRA rank, evaluated on a held-out test set.

## Evaluation Results

| Metric | Base | Fine-tuned | Δ |
|--------|------|------------|---|
| ROUGE-1 | 0.0672 | 0.0464 | -2.08% |
| ROUGE-2 | 0.0488 | 0.0400 | -0.88% |
| ROUGE-L | 0.0672 | 0.0436 | -2.37% |
| **BERTScore** | **0.5892** | **0.6726** | **+8.34%** |

> BERTScore uses `bert-base-chinese`. ROUGE uses character-level tokenization for Chinese.
> ROUGE decreases because the fine-tuned model learned to give concise, on-format answers rather than verbose outputs — BERTScore (semantic similarity) is the primary metric.

## Model Comparison

| Model | Dataset | Rank | BERTScore (ft) | Δ |
|-------|---------|------|----------------|---|
| MedQwen-1.5B (v1) | original split | r=8 | 0.6620 | +5.61% |
| **MedQwen-1.5B (v2)** | **80/10/10 split** | **r=16** | **0.6726** | **+8.34%** |
| MedQwen-3B | original split | r=16 | 0.6751 | +5.92% |
| MedQwen-7B | original split | r=8 | 0.6670 | +7.11% |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-1.5B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training samples | 24,531 (80% of 30,664) |
| Validation samples | 3,066 (10%) — used for early stopping |
| Test samples | 3,067 (10%) — held out for final evaluation |
| Max sequence length | 256 |
| Batch size (effective) | 8 |
| Learning rate | 2e-4 |
| LR schedule | Cosine with warmup |
| Epochs | 3 |
| Early stopping | patience=5, eval every 500 steps |
| Hardware | GCP L4 24GB |
| Precision | FP16 |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id    = "mellee030/MedQwen-1.5B-LoRA-r16"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": "你是一个专业的医疗问答助手，请根据用户的问题给出准确、简洁的医疗建议。"},
    {"role": "user",   "content": "糖尿病的早期症状有哪些？"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Dataset

~30,664 Chinese medical Q&A pairs derived from a structured Chinese medical knowledge base containing disease records with fields such as name, description, symptoms, causes, and treatments. Split 80/10/10 with seed=42 for reproducibility.

## Disclaimer

This model is for research purposes only and should not be used as a substitute for professional medical advice.
