---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
language:
- zh
tags:
- medical
- chinese
- lora
- peft
- qwen2.5
- fine-tuned
---

# MedQwen-7B-LoRA

LoRA fine-tuned [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on 30K Chinese medical Q&A dialogue pairs for domain-specific medical question answering.

## Model Description

This is a LoRA adapter trained on top of Qwen2.5-7B-Instruct. It adapts the base model to Chinese medical dialogue using parameter-efficient fine-tuning — only ~1% of parameters are trained, keeping the adapter lightweight (~150MB) while preserving the base model's general capabilities.

- **Base model**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning method**: LoRA (PEFT), r=8, alpha=16
- **Training hardware**: GCP L4 GPU (24GB VRAM)
- **Training time**: ~12 hours
- **Language**: Chinese (Simplified)
- **Domain**: Medical Q&A

## Training Data

30,284 Chinese medical dialogue pairs covering a wide range of medical topics including symptoms, medications, diseases, and treatment advice. Data was converted to Qwen2.5 chat template format (system / user / assistant).

## Evaluation Results

Evaluated on 412 held-out validation samples (50-sample subset):

| Metric | Base Qwen2.5-7B | MedQwen-7B | Δ |
|--------|----------------|------------|---|
| ROUGE-1 | 0.0407 | 0.0330 | -0.77% |
| ROUGE-2 | 0.0040 | 0.0024 | -0.16% |
| ROUGE-L | 0.0389 | 0.0330 | -0.59% |
| **BERTScore** | **0.5959** | **0.6670** | **+7.11%** |

> ROUGE scores are lower for the fine-tuned model because it learned concise, on-format answers rather than verbose outputs. BERTScore (semantic similarity using bert-base-chinese) is the primary metric for open-ended Chinese generation and shows a **+7.11% improvement**.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id    = "mellee030/MedQwen-7B-LoRA"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": "你是一个专业的医疗问答助手，请根据用户的问题给出准确、简洁的医疗建议。"},
    {"role": "user",   "content": "糖尿病的早期症状有哪些？"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 |
| Batch size | 1 (grad accum steps: 8, effective batch: 8) |
| Epochs | 3 |
| Max sequence length | 256 |
| Precision | FP16 |
| LR scheduler | Cosine with warmup |

## Limitations

- Intended for Chinese medical Q&A only
- Not a substitute for professional medical advice
- Performance may vary on medical topics underrepresented in training data
- Evaluated on a specific Chinese medical Q&A corpus — results may differ on other datasets

## Repository

Full training code, evaluation scripts, and Gradio chatbot demo:
[github.com/melc030/MedQwen](https://github.com/melc030/MedQwen)
