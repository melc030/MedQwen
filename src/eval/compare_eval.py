"""
compare_eval.py — Compare MedQwen (fine-tuned) vs MedGraphRAG on 400 stratified test questions.

Metric: BERTScore (bert-base-chinese) — semantic similarity vs ground truth, all 14 question types.

MedGraphRAG is queried via its Python API (requires Neo4j + Azure OpenAI running).
Use --skip-medgraphrag to evaluate MedQwen only, or --medgraphrag-answers to load
pre-saved MedGraphRAG answers from a JSON file.

Usage:
    # MedQwen only
    python src/eval/compare_eval.py --skip-medgraphrag

    # Full comparison (MedGraphRAG pipeline must be running)
    python src/eval/compare_eval.py

    # Full comparison with pre-saved MedGraphRAG answers
    python src/eval/compare_eval.py --medgraphrag-answers medgraphrag_answers.json
"""

import sys
import json
import random
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config

cfg = Config()

SAMPLE_PER_TYPE = 35   # 35 per type × 13 types ≈ 400 samples (2 types have < 35 in test set)
SEED            = 42

# ── Question type detection ────────────────────────────────────────────────────

QUESTION_TYPES = [
    ("是什么病",       "desc"),
    ("症状有哪些",     "symptom"),
    ("病因是什么",     "cause"),
    ("怎么预防",       "prevent"),
    ("治疗方法有哪些", "cure_way"),
    ("需要做哪些检查", "check"),
    ("推荐药物有哪些", "drug"),
    ("并发症有哪些",   "complication"),
    ("应该去哪个科室", "department"),
    ("适合吃什么",     "do_eat"),
    ("不能吃什么",     "not_eat"),
    ("传播方式是什么", "get_way"),
    ("治愈率是多少",   "cured_prob"),
    ("治疗周期",       "cure_lasttime"),
]


def detect_type(question: str) -> str:
    for keyword, qtype in QUESTION_TYPES:
        if keyword in question:
            return qtype
    return "other"


# ── Sampling ──────────────────────────────────────────────────────────────────

def load_stratified_samples(test_path: str, per_type: int, seed: int):
    """Load test.jsonl and sample up to per_type examples per question type."""
    random.seed(seed)
    buckets = defaultdict(list)
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            q = sample["messages"][1]["content"]
            qtype = detect_type(q)
            if qtype != "other":
                buckets[qtype].append(sample)

    samples = []
    for qtype, items in buckets.items():
        chosen = random.sample(items, min(per_type, len(items)))
        for item in chosen:
            item["_type"] = qtype
        samples.extend(chosen)

    random.shuffle(samples)
    print(f"Sampled {len(samples)} questions across {len(buckets)} types:")
    for keyword, qtype in QUESTION_TYPES:
        if qtype in buckets:
            print(f"  {qtype:<15} {min(per_type, len(buckets[qtype]))}")
    return samples


# ── MedQwen generation ────────────────────────────────────────────────────────

def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_medqwen(model, tokenizer, question, max_new_tokens=150):
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


# ── MedGraphRAG generation ────────────────────────────────────────────────────

def load_medgraphrag():
    """Import MedGraphRAG answer_question — requires Neo4j + Azure OpenAI running."""
    medgraphrag_path = str(Path(__file__).resolve().parent.parent.parent / "MedGraphRAG")
    if medgraphrag_path not in sys.path:
        sys.path.insert(0, medgraphrag_path)
    from generation.generator import answer_question
    return answer_question


def generate_medgraphrag(answer_question_fn, question):
    try:
        result = answer_question_fn(question)
        return result["answer"] if isinstance(result, dict) else str(result)
    except Exception as e:
        print(f"  MedGraphRAG error: {e}")
        return ""


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_bertscore(predictions, references):
    """BERTScore F1 using bert-base-chinese."""
    P, R, F1 = bert_score(predictions, references, lang="zh", verbose=False)
    return F1.tolist()


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, samples, label, answer_fn=None):
    """Generate answers for all samples."""
    print(f"\n── Generating answers: {label} ──")
    predictions, references, qtypes = [], [], []

    for i, sample in enumerate(samples):
        question  = sample["messages"][1]["content"]
        reference = sample["messages"][2]["content"]
        qtype     = sample["_type"]

        if answer_fn is not None:
            pred = generate_medgraphrag(answer_fn, question)
        else:
            pred = generate_medqwen(model, tokenizer, question)

        predictions.append(pred)
        references.append(reference)
        qtypes.append(qtype)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} done...")

    return predictions, references, qtypes


def score_and_print(predictions, references, label):
    print(f"\n── Scoring: {label} ──")
    print("  computing BERTScore...")
    bs_scores = compute_bertscore(predictions, references)
    bs_mean   = sum(bs_scores) / len(bs_scores)

    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  BERTScore  : {bs_mean:.4f}  (n={len(bs_scores)})")
    print(f"{'─'*40}")

    return {"bertscore_mean": round(bs_mean, 4), "bertscore_scores": bs_scores}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",             default=cfg.best_dir,          help="Path to LoRA adapter")
    parser.add_argument("--test",                default=cfg.test_jsonl,        help="Path to test.jsonl")
    parser.add_argument("--per-type",            type=int, default=SAMPLE_PER_TYPE)
    parser.add_argument("--skip-medgraphrag",    action="store_true",           help="Skip MedGraphRAG, evaluate MedQwen only")
    parser.add_argument("--medgraphrag-answers", default=None,                  help="Path to pre-saved MedGraphRAG answers JSON")
    parser.add_argument("--medqwen-out",         default="medqwen_answers.json",     help="Save MedQwen answers to this file")
    parser.add_argument("--out",                 default="compare_results.json",     help="Save final comparison to this file")
    args = parser.parse_args()

    # ── Load samples ──────────────────────────────────────────────────────────
    samples = load_stratified_samples(args.test, args.per_type, SEED)

    # ── Load MedQwen ──────────────────────────────────────────────────────────
    print("\nloading tokenizer + base model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    print("loading LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter)
    ft_model.eval()

    # ── MedQwen answers + scores ──────────────────────────────────────────────
    mq_preds, references, qtypes = run_eval(ft_model, tokenizer, samples, "MedQwen")
    mq_scores = score_and_print(mq_preds, references, "MedQwen (fine-tuned)")

    # save MedQwen answers separately for transparency
    questions = [s["messages"][1]["content"] for s in samples]
    mq_out = {
        "timestamp": datetime.datetime.now().isoformat(),
        "adapter":   args.adapter,
        "answers":   {q: p for q, p in zip(questions, mq_preds)},
    }
    with open(args.medqwen_out, "w", encoding="utf-8") as f:
        json.dump(mq_out, f, ensure_ascii=False, indent=2)
    print(f"MedQwen answers saved to {args.medqwen_out}")

    # ── MedGraphRAG answers + scores ──────────────────────────────────────────
    mg_scores = None
    mg_preds  = None

    if not args.skip_medgraphrag:
        if args.medgraphrag_answers:
            print(f"\nloading pre-saved MedGraphRAG answers from {args.medgraphrag_answers}...")
            with open(args.medgraphrag_answers, encoding="utf-8") as f:
                saved = json.load(f)
            answers = saved.get("answers", saved)   # support both wrapped and flat format
            mg_preds = [answers.get(s["messages"][1]["content"], "") for s in samples]
        else:
            print("\nloading MedGraphRAG pipeline...")
            answer_fn = load_medgraphrag()
            mg_preds, _, _ = run_eval(None, None, samples, "MedGraphRAG", answer_fn=answer_fn)

        mg_scores = score_and_print(mg_preds, references, "MedGraphRAG (RAG)")

    # ── Comparison summary ────────────────────────────────────────────────────
    if mg_scores:
        delta = mq_scores["bertscore_mean"] - mg_scores["bertscore_mean"]
        print(f"\n{'='*40}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*40}")
        print(f"  MedQwen     BERTScore : {mq_scores['bertscore_mean']:.4f}")
        print(f"  MedGraphRAG BERTScore : {mg_scores['bertscore_mean']:.4f}")
        print(f"  Δ                     : {delta:+.4f}")
        print(f"{'='*40}")

    # ── Save results ──────────────────────────────────────────────────────────
    questions = [s["messages"][1]["content"] for s in samples]
    out = {
        "timestamp":   datetime.datetime.now().isoformat(),
        "sample_size": len(samples),
        "per_type":    args.per_type,
        "adapter":     args.adapter,
        "medqwen": {
            "scores":  mq_scores,
            "details": [
                {"question": q, "type": t, "reference": r, "prediction": p}
                for q, t, r, p in zip(questions, qtypes, references, mq_preds)
            ],
        },
        "medgraphrag": {
            "scores":  mg_scores,
            "details": [
                {"question": q, "type": t, "reference": r, "prediction": p}
                for q, t, r, p in zip(questions, qtypes, references, mg_preds)
            ] if mg_preds else None,
        } if not args.skip_medgraphrag else None,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
