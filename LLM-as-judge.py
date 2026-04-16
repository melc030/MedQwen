"""
LLM-as-Judge evaluation: GPT-4o rates base vs fine-tuned MedQwen responses.

Each sample is scored on three dimensions (1–5):
  - coherence  : is the response fluent and logically structured?
  - accuracy   : is the medical information correct and relevant?
  - format     : is the response concise and appropriately formatted?

GPT-4o receives the question + response only (no reference answer),
so it judges output quality rather than n-gram overlap.
"""
import os
import json
import random
import time
import datetime
import torch
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config

cfg = Config()
SAMPLE_SIZE    = 20   # GPT-4o calls cost money — 20 is plenty for a clear signal
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://socrates-rag.openai.azure.com/")
AZURE_API_KEY  = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_API_VER  = "2025-01-01-preview"
DEPLOYMENT     = "gpt-4o"

JUDGE_PROMPT = """\
你是一位医学领域的专业评审员。请对以下医疗问答进行评分。

【问题】
{question}

【回答】
{answer}

请从以下三个维度对回答进行评分（每项1-5分，5分最高）：
1. 连贯性（coherence）：回答是否流畅、逻辑清晰？
2. 准确性（accuracy）：医疗信息是否正确、切题？
3. 格式（format）：回答是否简洁、格式恰当？

请严格按照以下JSON格式输出，不要添加任何其他内容：
{{"coherence": <1-5>, "accuracy": <1-5>, "format": <1-5>, "comment": "<一句话说明>"}}
"""


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user",   "content": question},
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
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def judge(client, question, answer, retries=3):
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            return scores
        except Exception as e:
            print(f"    judge error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return {"coherence": None, "accuracy": None, "format": None, "comment": "parse error"}


def run_judge_eval(model, tokenizer, samples, label, client):
    print(f"\nevaluating {label} ({len(samples)} samples)...")
    results = []
    totals  = {"coherence": 0, "accuracy": 0, "format": 0}

    for i, sample in enumerate(samples):
        question = sample['messages'][1]['content']
        answer   = generate(model, tokenizer, question)
        scores   = judge(client, question, answer)

        results.append({
            "question": question,
            "answer":   answer,
            "scores":   scores,
        })

        for k in totals:
            if scores.get(k) is not None:
                totals[k] += scores[k]

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(samples)} done...")

    n = len(samples)
    avgs = {k: round(totals[k] / n, 4) for k in totals}

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Coherence  : {avgs['coherence']:.2f} / 5")
    print(f"  Accuracy   : {avgs['accuracy']:.2f} / 5")
    print(f"  Format     : {avgs['format']:.2f} / 5")
    print(f"  Overall    : {sum(avgs.values())/3:.2f} / 5")
    print(f"{'─'*40}")

    return avgs, results


def main():
    random.seed(42)

    with open(cfg.valid_jsonl, 'r') as f:
        all_samples = [json.loads(l) for l in f]
    samples = random.sample(all_samples, min(SAMPLE_SIZE, len(all_samples)))

    client = AzureOpenAI(
        api_version=AZURE_API_VER,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )

    print("loading tokenizer + base model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(cfg.device)
    base_model.eval()

    base_avgs, base_results = run_judge_eval(base_model, tokenizer, samples, "BASE Qwen2.5-1.5B-Instruct", client)

    print("\nloading LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, cfg.best_dir).to(cfg.device)
    ft_model.eval()

    ft_avgs, ft_results = run_judge_eval(ft_model, tokenizer, samples, "FINE-TUNED MedQwen", client)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"  IMPROVEMENT SUMMARY (GPT-4o Judge)")
    print(f"{'='*40}")
    for k in ["coherence", "accuracy", "format"]:
        delta = ft_avgs[k] - base_avgs[k]
        print(f"  {k.capitalize():<12}: {base_avgs[k]:.2f} → {ft_avgs[k]:.2f}  ({delta:+.2f})")
    base_overall = sum(base_avgs.values()) / 3
    ft_overall   = sum(ft_avgs.values()) / 3
    print(f"  {'Overall':<12}: {base_overall:.2f} → {ft_overall:.2f}  ({ft_overall-base_overall:+.2f})")
    print(f"{'='*40}")

    # ── Save ─────────────────────────────────────────────────────
    out = {
        "timestamp":   datetime.datetime.now().isoformat(),
        "sample_size": SAMPLE_SIZE,
        "judge_model": f"Azure GPT-4o ({DEPLOYMENT})",
        "base_model":  cfg.hf_model_id,
        "fine_tuned":  cfg.best_dir,
        "base_scores":     base_avgs,
        "ft_scores":       ft_avgs,
        "improvement": {k: round(ft_avgs[k] - base_avgs[k], 4) for k in base_avgs},
        "base_details":    base_results,
        "ft_details":      ft_results,
    }
    out_path = "llm_judge_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
