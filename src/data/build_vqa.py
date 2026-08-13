"""
Build multimodal VQA training data for MedQwen-VL.

Reads the rich `dataset.jsonl` exported from MedGraphRAG and produces
English, label-grounded instruction/answer pairs in Qwen2.5-VL chat format,
split 80/10/10 (train/val/test) stratified by class.

Why re-split instead of using the provided `split` field?
  The shipped split covers only 23 of 135 classes in `test`, which makes
  per-class classification eval impossible. We re-stratify by class so every
  class with enough images gets proportional train/val/test coverage.

Rare-class rule:
  Classes with < --min-class-size images go entirely to `train` (you cannot
  stratify a 1-image class, and a 1-image "test set" is meaningless). They
  still contribute to learning, they just aren't measured at eval time.

Templates (all derived strictly from existing metadata — no fabricated
clinical description, or we'd teach the model to hallucinate findings):
  1. classification : "What does this medical image show?"  (always)
  2. modality       : "What imaging modality is this?"      (always)
  3. icd10          : "What is the ICD-10 code ...?"         (only when present)

The noisy `kind=normal` field is intentionally NOT used for a
normal/abnormal template — it mixes anatomical landmarks, prep-quality
labels and actual diseases, so a binary answer would often be wrong.

Bilingual (future): every record carries full `meta` (including
`diseases_zh`), and templates are keyed by language. Add a `zh` template set
and pass --lang zh to regenerate Chinese pairs without touching the splits.

Usage:
    python src/data/build_vqa.py
    python src/data/build_vqa.py --source-dir /path/to/multimodal_export
    python src/data/build_vqa.py --ratios 0.8 0.1 0.1 --min-class-size 10
    python src/data/build_vqa.py --copy-images   # bundle images for GCP upload
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SOURCE = Path("/Users/yuxincui/PycharmProjects/MedGraphRAG/data/multimodal_export")
DEFAULT_OUT    = Path(__file__).resolve().parent.parent.parent / "data" / "multimodal"

# Human-readable modality phrasing (with article) for natural English answers.
MODALITY_EN = {
    "clinical_photo": "a clinical photograph",
    "endoscopy":      "an endoscopic",
    "fundus":         "a fundus",
    "xray":           "an X-ray",
    "ultrasound":     "an ultrasound",
}


# ── Templates ─────────────────────────────────────────────────────────────────
def _clean(name: str) -> str:
    """Tidy raw label strings for display."""
    return (name or "").strip()


def make_pairs_en(rec: dict):
    """Yield (question, answer) tuples for one image record (English)."""
    modality = MODALITY_EN.get(rec["modality"], rec["modality"])
    name_en  = _clean(rec.get("name_en")) or rec["class"]
    icd10    = _clean(rec.get("icd10"))
    kind     = rec.get("kind")

    # 1. classification — phrasing adapts to kind, stays grounded
    if kind == "disease":
        head = f"This is {modality} image showing {name_en}"
        ans  = f"{head} (ICD-10: {icd10})." if icd10 else f"{head}."
    elif kind == "finding":
        head = f"This is {modality} image showing the finding: {name_en}"
        ans  = f"{head} (ICD-10: {icd10})." if icd10 else f"{head}."
    else:  # normal / landmark / quality — no disease claim
        ans = f"This is {modality} image showing {name_en}."
    yield ("What does this medical image show?", ans)

    # 2. modality
    yield ("What imaging modality is this?",
           f"This is {modality} image.")

    # 3. icd10 (only when a real code exists and it's a disease/finding)
    if icd10 and kind in ("disease", "finding"):
        codes = ", ".join(c.strip() for c in icd10.split(";") if c.strip())
        yield (f"What is the ICD-10 code for the condition shown in this image?",
               f"The ICD-10 code for {name_en} is {codes}.")

    # 4. UMLS concept grounding (canonical term + CUI), when available
    cui      = _clean(rec.get("cui"))
    cui_name = _clean(rec.get("cui_name"))
    if cui and cui_name and kind in ("disease", "finding"):
        yield ("What is the standardized UMLS medical concept for the condition shown?",
               f"The UMLS concept is {cui_name} (CUI: {cui}).")


TEMPLATES = {"en": make_pairs_en}


# ── Split ─────────────────────────────────────────────────────────────────────
def stratified_split(records, ratios, min_class_size, seed):
    """Split image records by class. Returns dict split_name -> [records]."""
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)

    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    train_r, val_r, test_r = ratios

    for cls, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        if n < min_class_size:
            out["train"].extend(items)         # too small to stratify
            continue
        n_test = round(n * test_r)
        n_val  = round(n * val_r)
        out["test"].extend(items[:n_test])
        out["val"].extend(items[n_test:n_test + n_val])
        out["train"].extend(items[n_test + n_val:])

    for s in out:
        rng.shuffle(out[s])
    return out


# ── Record formatting ─────────────────────────────────────────────────────────
def to_chat_record(image_path, question, answer, rec):
    """One Qwen2.5-VL chat example (content-list format) + preserved meta."""
    return {
        "image": image_path,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": question},
            ]},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "image_id":    rec.get("image_id"),
            "class":       rec.get("class"),
            "modality":    rec.get("modality"),
            "name_en":     rec.get("name_en"),
            "icd10":       rec.get("icd10"),
            "kind":        rec.get("kind"),
            "cui":         rec.get("cui", ""),           # UMLS concept id
            "cui_name":    rec.get("cui_name", ""),      # UMLS canonical term
            "diseases_zh": rec.get("diseases_zh", []),   # for future bilingual
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE,
                    help="dir containing dataset.jsonl and images/")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--min-class-size", type=int, default=10,
                    help="classes below this size go entirely to train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lang", choices=sorted(TEMPLATES), default="en")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images into out-dir/images for a portable bundle")
    args = ap.parse_args()

    src = args.source_dir / "dataset.jsonl"
    records = [json.loads(l) for l in open(src, encoding="utf-8")]
    print(f"loaded {len(records)} image records from {src}")

    splits = stratified_split(records, args.ratios, args.min_class_size, args.seed)
    make_pairs = TEMPLATES[args.lang]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len({r["class"] for r in records})

    for split_name, recs in splits.items():
        out_path = args.out_dir / f"{split_name}.jsonl"
        n_pairs = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in recs:
                for q, a in make_pairs(rec):
                    chat = to_chat_record(rec["image"], q, a, rec)
                    f.write(json.dumps(chat, ensure_ascii=False) + "\n")
                    n_pairs += 1
        cov = len({r["class"] for r in recs})
        print(f"  {split_name:5s}: {len(recs):6d} images -> {n_pairs:6d} pairs "
              f"| {cov}/{n_classes} classes")

    if args.copy_images:
        dst = args.out_dir / "images"
        dst.mkdir(exist_ok=True)
        print(f"copying images -> {dst} ...")
        for rec in records:
            shutil.copy2(args.source_dir / rec["image"], dst / Path(rec["image"]).name)
        print("  done.")
    else:
        print(f"\nimages NOT copied. jsonl stores relative paths like 'images/...'.")
        print(f"point training at images root: {args.source_dir}")
        print(f"(or re-run with --copy-images to bundle for GCP upload)")


if __name__ == "__main__":
    main()
