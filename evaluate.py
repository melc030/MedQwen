"""
ROUGE evaluation: base model vs fine-tuned MedQwen on validation set.
Runs on a sample of the validation set for speed.
"""
import json
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config

cfg = Config()
SAMPLE_SIZE = 50  # increase for more thorough eval, 50 is fast enough


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, question, max_new_tokens=100):
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, samples, label):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    references, predictions = [], []

    def char_tokenize(text):
        return ' '.join(list(text))

    print(f"\nevaluating {label} on {len(samples)} samples...")
    for i, sample in enumerate(samples):
        question   = sample['messages'][1]['content']
        reference  = sample['messages'][2]['content']
        prediction = generate(model, tokenizer, question)

        references.append(reference)
        predictions.append(prediction)

        s = scorer.score(char_tokenize(reference), char_tokenize(prediction))
        scores['rouge1'].append(s['rouge1'].fmeasure)
        scores['rouge2'].append(s['rouge2'].fmeasure)
        scores['rougeL'].append(s['rougeL'].fmeasure)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done...")

    r1 = sum(scores['rouge1']) / len(scores['rouge1'])
    r2 = sum(scores['rouge2']) / len(scores['rouge2'])
    rl = sum(scores['rougeL']) / len(scores['rougeL'])

    # BERTScore — uses multilingual model, handles Chinese semantics properly
    print("  computing BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang='zh', verbose=False)
    bs = F1.mean().item()

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  ROUGE-1    : {r1:.4f}")
    print(f"  ROUGE-2    : {r2:.4f}")
    print(f"  ROUGE-L    : {rl:.4f}")
    print(f"  BERTScore  : {bs:.4f}")
    print(f"{'─'*40}")
    return r1, r2, rl, bs


def main():
    import random
    random.seed(42)

    with open(cfg.valid_jsonl, 'r') as f:
        all_samples = [json.loads(l) for l in f]
    samples = random.sample(all_samples, min(SAMPLE_SIZE, len(all_samples)))

    print("loading tokenizer + base model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.float16, trust_remote_code=True
    ).to(cfg.device)
    base_model.eval()

    b1, b2, bl, bbs = evaluate(base_model, tokenizer, samples, "BASE Qwen2.5-1.5B-Instruct")

    print("\nloading LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, cfg.best_dir).to(cfg.device)
    ft_model.eval()

    f1, f2, fl, fbs = evaluate(ft_model, tokenizer, samples, "FINE-TUNED MedQwen")

    print(f"\n{'='*40}")
    print(f"  IMPROVEMENT SUMMARY")
    print(f"{'='*40}")
    print(f"  ROUGE-1    : {b1:.4f} → {f1:.4f}  ({(f1-b1)*100:+.2f}%)")
    print(f"  ROUGE-2    : {b2:.4f} → {f2:.4f}  ({(f2-b2)*100:+.2f}%)")
    print(f"  ROUGE-L    : {bl:.4f} → {fl:.4f}  ({(fl-bl)*100:+.2f}%)")
    print(f"  BERTScore  : {bbs:.4f} → {fbs:.4f}  ({(fbs-bbs)*100:+.2f}%)")
    print(f"{'='*40}")

    # save results to JSON
    import datetime
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sample_size": SAMPLE_SIZE,
        "base_model": cfg.hf_model_id,
        "fine_tuned": cfg.best_dir,
        "base": {"rouge1": b1, "rouge2": b2, "rougeL": bl, "bertscore": bbs},
        "fine_tuned_scores": {"rouge1": f1, "rouge2": f2, "rougeL": fl, "bertscore": fbs},
        "improvement": {
            "rouge1":    round((f1 - b1) * 100, 4),
            "rouge2":    round((f2 - b2) * 100, 4),
            "rougeL":    round((fl - bl) * 100, 4),
            "bertscore": round((fbs - bbs) * 100, 4),
        }
    }
    out_path = "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == '__main__':
    main()
