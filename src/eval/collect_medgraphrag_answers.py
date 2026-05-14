"""
collect_medgraphrag_answers.py — Query MedGraphRAG on ~400 stratified test questions
and save answers to medgraphrag_answers.json.

Sample size: 35 per question type × 13 types ≈ 400 total
(get_way and cure_lasttime have fewer than 35 samples in the test set)

Prerequisites:
  - Neo4j running:  docker compose up -d  (in MedGraphRAG directory)
  - Azure OpenAI env vars set: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY

Usage:
    python src/eval/collect_medgraphrag_answers.py
    python src/eval/collect_medgraphrag_answers.py --out my_answers.json
"""

import sys
import json
import random
import argparse
import datetime
import time
from pathlib import Path
from collections import defaultdict

MEDGRAPHRAG_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "MedGraphRAG")

# Add src/ to path so `from config import Config` works
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

DEFAULT_TEST_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "medgraphrag_qa_clean" / "test.jsonl")

SAMPLE_PER_TYPE = 35
SEED            = 42

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


def load_stratified_samples(test_path: str, per_type: int, seed: int):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=DEFAULT_TEST_PATH, help="Path to test.jsonl")
    parser.add_argument("--out",  default="medgraphrag_answers.json")
    args = parser.parse_args()

    # ── Load samples ──────────────────────────────────────────────────────────
    samples = load_stratified_samples(args.test, SAMPLE_PER_TYPE, SEED)
    questions = [s["messages"][1]["content"] for s in samples]

    # ── Load MedGraphRAG ──────────────────────────────────────────────────────
    # Temporarily put MedGraphRAG at front of sys.path so its config.py is
    # found instead of MedQwen's config.py when generator.py does `from config import ...`
    print(f"\nloading MedGraphRAG from {MEDGRAPHRAG_PATH}...")
    from generation.generator import answer_question

    # ── Generate answers ──────────────────────────────────────────────────────
    print(f"\ngenerating answers for {len(questions)} questions...")
    answers = {}
    errors  = []

    for i, q in enumerate(questions):
        try:
            result   = answer_question(q)
            answer   = result["answer"] if isinstance(result, dict) else str(result)
            answers[q] = answer
        except Exception as e:
            print(f"  [error] q={i+1}: {e}")
            answers[q] = ""
            errors.append({"idx": i, "question": q, "error": str(e)})
            time.sleep(1)   # brief pause on error to avoid hammering the API

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(questions)} done...")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "sample_size":  len(questions),
        "error_count":  len(errors),
        "answers":      answers,
        "errors":       errors,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\ndone — {len(answers) - len(errors)} successful, {len(errors)} errors")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
