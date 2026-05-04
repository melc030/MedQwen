"""
Generate MedQwen-style Q&A pairs from medical.json.

Input:  data/medical.json  (8,807-disease MongoDB export)
Output: data/qa_train.txt  (question\nanswer\n\n)
        data/qa_train.jsonl (Qwen chat format, ready for fine-tuning)

Each disease record produces up to 14 Q&A pairs, one per populated field.
List fields are joined with "；". Text fields are used as-is.
Empty / missing fields are skipped — no pair is emitted.

Usage:
    python generate_qa.py                 # default paths
    python generate_qa.py --split         # also produce train/valid/test splits (80/10/10)
    python generate_qa.py --help
"""

import argparse
import json
import random
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

MEDICAL_JSON = Path(__file__).parent.parent / "MedGraphRAG" / "data" / "medical.json"
OUT_DIR      = Path(__file__).parent.parent / "medgraphrag_qa"

SYSTEM_PROMPT = "你是一个专业的医疗问答助手，请根据用户的问题给出准确、简洁的医疗建议。"

# ── Field → (question template, value extractor) ──────────────────────────────
# Each entry: (question_template_fn, value_fn)
#   question_template_fn(name) → question string
#   value_fn(record)           → answer string, or None to skip

def _join(values):
    """Join a non-empty list with '；', else return None."""
    v = [s.strip() for s in values if s and s.strip()]
    return "；".join(v) if v else None

def _text(value):
    """Return stripped text, or None if blank."""
    v = value.strip() if isinstance(value, str) else ""
    return v if v else None

TEMPLATES = [
    (
        lambda name: f"{name}是什么病？",
        lambda r: _text(r.get("desc", ""))
    ),
    (
        lambda name: f"{name}的症状有哪些？",
        lambda r: _join(r.get("symptom", []))
    ),
    (
        lambda name: f"{name}的病因是什么？",
        lambda r: _text(r.get("cause", ""))
    ),
    (
        lambda name: f"{name}怎么预防？",
        lambda r: _text(r.get("prevent", ""))
    ),
    (
        lambda name: f"{name}的治疗方法有哪些？",
        lambda r: _join(r.get("cure_way", []))
    ),
    (
        lambda name: f"{name}需要做哪些检查？",
        lambda r: _join(r.get("check", []))
    ),
    (
        lambda name: f"{name}的推荐药物有哪些？",
        lambda r: _join(r.get("recommand_drug", []))
    ),
    (
        lambda name: f"{name}的并发症有哪些？",
        lambda r: _join(r.get("acompany", []))
    ),
    (
        lambda name: f"{name}应该去哪个科室就诊？",
        lambda r: _join(r.get("cure_department", []))
    ),
    (
        lambda name: f"{name}患者适合吃什么食物？",
        lambda r: _join(r.get("do_eat", []))
    ),
    (
        lambda name: f"{name}患者不能吃什么？",
        lambda r: _join(r.get("not_eat", []))
    ),
    (
        lambda name: f"{name}的传播方式是什么？",
        lambda r: _text(r.get("get_way", ""))
    ),
    (
        lambda name: f"{name}的治愈率是多少？",
        lambda r: _text(r.get("cured_prob", ""))
    ),
    (
        lambda name: f"{name}的治疗周期大概是多久？",
        lambda r: _text(r.get("cure_lasttime", ""))
    ),
]

# ── Parsing ───────────────────────────────────────────────────────────────────

def load_records(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ── Generation ────────────────────────────────────────────────────────────────

def generate_pairs(records):
    """Yield (question, answer) for every populated field of every record."""
    for record in records:
        name = record.get("name", "").strip()
        if not name:
            continue
        for q_fn, a_fn in TEMPLATES:
            answer = a_fn(record)
            if answer:
                yield q_fn(name), answer


# ── Writers ───────────────────────────────────────────────────────────────────

def write_txt(pairs, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(f"{q}\n{a}\n\n")


def write_jsonl(pairs, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for idx, (q, a) in enumerate(pairs):
            record = {
                "id": idx,
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": a},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_and_write(pairs, out_dir: Path, seed=42, suffix=""):
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    splits = {
        "train": pairs[:n_train],
        "valid": pairs[n_train : n_train + n_valid],
        "test":  pairs[n_train + n_valid :],
    }
    for split_name, split_pairs in splits.items():
        write_txt(split_pairs,   out_dir / f"qa_{split_name}{suffix}.txt")
        write_jsonl(split_pairs, out_dir / f"qa_{split_name}{suffix}.jsonl")
        print(f"  {split_name:6s}: {len(split_pairs):6,} pairs")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from medical.json")
    parser.add_argument("--src",        default=str(MEDICAL_JSON), help="Path to medical.json")
    parser.add_argument("--out",        default=str(OUT_DIR),      help="Output directory")
    parser.add_argument("--split",      action="store_true",       help="Also write 80/10/10 train/valid/test splits")
    parser.add_argument("--seed",       type=int, default=42,      help="Random seed for splitting")
    parser.add_argument("--min-answer", type=int, default=0,
                        help="Drop pairs whose answer is shorter than this many characters (default: 0 = keep all)")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {src} ...")
    records = load_records(src)
    print(f"  {len(records):,} records loaded")

    print("Generating Q&A pairs ...")
    pairs = list(generate_pairs(records))
    print(f"  {len(pairs):,} pairs generated")

    if args.min_answer > 0:
        before = len(pairs)
        pairs = [(q, a) for q, a in pairs if len(a) >= args.min_answer]
        print(f"  {before - len(pairs):,} pairs dropped (answer < {args.min_answer} chars) → {len(pairs):,} kept")

    suffix = f"_min{args.min_answer}" if args.min_answer > 0 else ""

    # full dump
    write_txt(pairs,   out / f"qa_all{suffix}.txt")
    write_jsonl(pairs, out / f"qa_all{suffix}.jsonl")
    print(f"Written: {out / f'qa_all{suffix}.txt'}, {out / f'qa_all{suffix}.jsonl'}")

    if args.split:
        print("Writing train/valid/test splits ...")
        split_and_write(pairs, out, seed=args.seed, suffix=suffix)

    print("Done.")


if __name__ == "__main__":
    main()
