---
base_model: Qwen/Qwen2.5-3B-Instruct
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

# MedQwen-3B-LoRA

LoRA fine-tuned [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on 30K Chinese medical Q&A dialogue pairs.

## Evaluation Results

| Metric | Base | Fine-tuned | Δ |
|--------|------|------------|---|
| ROUGE-1 | 0.0160 | 0.0364 | +2.04% |
| ROUGE-2 | 0.0000 | 0.0200 | +2.00% |
| ROUGE-L | 0.0160 | 0.0330 | +1.70% |
| **BERTScore** | **0.6159** | **0.6683** | **+5.24%** |

> BERTScore uses `bert-base-chinese`. ROUGE uses character-level tokenization for Chinese.

## Model Comparison

| Model | Fine-tuned BERTScore | Δ |
|-------|---------------------|---|
| MedQwen-1.5B (r=8) | 0.6620 | +5.61% |
| **MedQwen-3B (r=8)** | **0.6683** | **+5.24%** |
| MedQwen-7B (r=8) | 0.6670 | +7.11% |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-3B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training samples | 30,284 |
| Validation samples | 412 |
| Max sequence length | 512 |
| Batch size (effective) | 8 |
| Learning rate | 2e-4 |
| LR schedule | Cosine with warmup |
| Hardware | GCP L4 24GB |
| Precision | FP16 |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_id    = "mellee030/MedQwen-3B-LoRA"

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

30,284 training pairs and 412 validation pairs from a Chinese medical Q&A corpus, formatted as Qwen2.5 chat template (system / user / assistant turns).

## Disclaimer

This model is for research purposes only and should not be used as a substitute for professional medical advice.
