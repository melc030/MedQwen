"""
Quick inference comparison: base Qwen2.5-1.5B vs fine-tuned MedQwen.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config

cfg = Config()

QUESTIONS = [
    "无症状颈动脉粥样硬化的影像学检查有些什么？",
    "头孢地尼胶囊能治理什么疾病?",
    "糖尿病人能喝不含蔗糖的中老年奶粉吗？",
    "宝宝支气管炎要如何治疗",
    "我最近头疼、发烧，可能是什么病？",
]


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system",  "content": cfg.system_prompt},
        {"role": "user",    "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, question, max_new_tokens=150):
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for deterministic comparison
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run(model, tokenizer, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[Q{i}] {q}")
        answer = generate(model, tokenizer, q)
        print(f"[A]  {answer}")


def main():
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # ── Base model ──────────────────────────────────────────────
    print("loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.float16, trust_remote_code=True
    ).to(cfg.device)
    base_model.eval()
    run(base_model, tokenizer, "BASE Qwen2.5-1.5B-Instruct (no fine-tuning)")

    # ── Fine-tuned model ────────────────────────────────────────
    print("\nloading LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, cfg.best_dir).to(cfg.device)
    ft_model.eval()
    run(ft_model, tokenizer, "FINE-TUNED MedQwen (LoRA)")

    print("\ndone.")


if __name__ == "__main__":
    main()
