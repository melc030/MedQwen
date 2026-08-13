"""
Classification eval for MedQwen-VL (Qwen2.5-VL-3B + LoRA).

Realistic generate+match eval: runs the exact serving path (same chat format,
no system prompt — matching training — greedy decoding), then maps the
free-text answer to one of the closed-set classes with lenient matching, so a
correct-but-differently-phrased answer still gets credit.

Metrics (the multimodal task is classification, not open-ended generation, so
ROUGE/BERTScore don't apply):
  - disease  : accuracy, macro-F1, weighted-F1, and per-modality breakdown
               (macro-F1 is the headline — classes range 350 -> 1 images)
  - modality : 5-way accuracy (sanity check, should be ~99%)
  - icd10    : code-set exact-match + set-F1
  - cui      : UMLS concept-id exact-match

Usage:
    python src/eval/eval_vl.py                     # fine-tuned (cfg.vl_best_dir)
    python src/eval/eval_vl.py --base              # zero-shot base, for the lift
    python src/eval/eval_vl.py --limit 200         # quick subset
    python src/eval/eval_vl.py --out eval_vl_3b.json
"""

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

cfg = Config()
DTYPE = cfg.vl_dtype if cfg.device.startswith('cuda') else torch.float32

# question templates from build_vqa.py — used to route each test record
CLS_Q = "What does this medical image show?"
MOD_Q = "What imaging modality is this?"
ICD_Q = "What is the ICD-10 code"
CUI_Q = "What is the standardized UMLS medical concept"

# answer keyword -> modality (for parsing the modality question)
MODALITY_KW = {
    "clinical photograph": "clinical_photo",
    "endoscop":            "endoscopy",
    "fundus":              "fundus",
    "x-ray":               "xray",
    "ultrasound":          "ultrasound",
}


# ── Matching helpers ────────────────────────────────────────────────────────────

def norm(s):
    """Lowercase, collapse to alphanumerics — for lenient string comparison."""
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def extract_disease(text):
    """Pull the disease span out of a generated classification answer."""
    t = text
    low = t.lower()
    for marker in ('showing the finding:', 'showing the finding', 'showing'):
        if marker in low:
            t = t[low.rindex(marker) + len(marker):]
            break
    for cut in ('(ICD', '(icd', '\n'):
        if cut in t:
            t = t[:t.index(cut)]
    return t.strip(' :.\n')


def match_class(text, norm_name_to_class):
    """Map a generated answer to a class via exact -> substring -> fuzzy match."""
    p = norm(extract_disease(text))
    if not p:
        return None
    if p in norm_name_to_class:                       # exact
        return norm_name_to_class[p]
    subs = [n for n in norm_name_to_class if n and (n in p or p in n)]
    if subs:                                          # substring (longest wins)
        return norm_name_to_class[max(subs, key=len)]
    close = difflib.get_close_matches(p, list(norm_name_to_class), n=1, cutoff=0.6)
    if close:                                         # fuzzy
        return norm_name_to_class[close[0]]
    return None


def match_modality(text):
    low = text.lower()
    for kw, mod in MODALITY_KW.items():
        if kw in low:
            return mod
    return None


def extract_codes(text):
    """Set of ICD-10 codes mentioned in the text, e.g. {'L70', 'L71'}."""
    return set(re.findall(r'[A-Z]\d{2}(?:\.\d+)?', text))


def extract_cui(text):
    """First UMLS CUI (C + 6-8 digits) mentioned in the text, or ''."""
    m = re.search(r'C\d{6,8}', text)
    return m.group(0) if m else ''


# ── Model ───────────────────────────────────────────────────────────────────────

def load_model(use_base):
    src = cfg.vl_model_path if use_base else cfg.vl_best_dir
    proc_src = cfg.vl_model_path if use_base else cfg.vl_best_dir
    processor = AutoProcessor.from_pretrained(
        proc_src, min_pixels=cfg.vl_min_pixels, max_pixels=cfg.vl_max_pixels)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.vl_model_path, dtype=DTYPE, device_map='auto')
    if not use_base:
        model = PeftModel.from_pretrained(model, cfg.vl_best_dir)
    model.eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, image_path, question, max_new_tokens):
    image = Image.open(image_path).convert('RGB')
    messages = []
    if cfg.vl_system_prompt:
        messages.append({"role": "system", "content": cfg.vl_system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": question},
    ]})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── Metrics ─────────────────────────────────────────────────────────────────────

def cls_metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n":           len(y_true),
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":    round(f1_score(y_true, y_pred, labels=labels,
                                      average='macro', zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels,
                                      average='weighted', zero_division=0), 4),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', action='store_true', help='eval zero-shot base model')
    ap.add_argument('--limit', type=int, default=None, help='cap records (quick run)')
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--out', type=str, default=None, help='write metrics JSON here')
    args = ap.parse_args()

    images_root = Path(cfg.images_root)
    records = [json.loads(l) for l in open(cfg.mm_test_jsonl, encoding='utf-8')]
    if args.limit:
        records = records[:args.limit]

    # closed set: normalized name_en -> class
    norm_name_to_class = {
        norm(r['meta']['name_en']): r['meta']['class']
        for r in records if r['meta'].get('name_en')
    }

    print(f"model  : {'BASE ' + cfg.vl_model_id if args.base else cfg.vl_best_dir}")
    print(f"records: {len(records)}  | classes in test: {len(set(norm_name_to_class.values()))}")
    model, processor = load_model(args.base)

    # disease (per-modality), modality, icd accumulators
    dz_true, dz_pred, dz_mod = [], [], []
    mod_true, mod_pred = [], []
    icd_exact, icd_f1s = [], []
    cui_exact = []

    for i, r in enumerate(records, 1):
        q   = r['messages'][0]['content'][1]['text']
        m   = r['meta']
        img = str(images_root / r['image'])
        ans = generate(model, processor, img, q, args.max_new_tokens)

        if q == CLS_Q:
            pred = match_class(ans, norm_name_to_class)
            dz_true.append(m['class'])
            dz_pred.append(pred or '__none__')
            dz_mod.append(m['modality'])
        elif q == MOD_Q:
            mod_true.append(m['modality'])
            mod_pred.append(match_modality(ans) or '__none__')
        elif q.startswith(ICD_Q):
            gold = extract_codes(m.get('icd10', '').replace(';', ' '))
            pred = extract_codes(ans)
            icd_exact.append(gold == pred)
            inter = len(gold & pred)
            prec = inter / len(pred) if pred else 0.0
            rec  = inter / len(gold) if gold else 0.0
            icd_f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        elif q.startswith(CUI_Q):
            gold = m.get('cui', '')
            cui_exact.append(bool(gold) and extract_cui(ans) == gold)

        if i % 100 == 0:
            print(f"  {i}/{len(records)} ...")

    # ── assemble report ──────────────────────────────────────────
    report = {"model": cfg.vl_model_id if args.base else cfg.vl_best_dir,
              "base": args.base, "n_records": len(records)}

    if dz_true:
        report["disease"] = cls_metrics(dz_true, dz_pred)
        per_mod = {}
        by_mod = defaultdict(lambda: ([], []))
        for t, p, mo in zip(dz_true, dz_pred, dz_mod):
            by_mod[mo][0].append(t)
            by_mod[mo][1].append(p)
        for mo, (t, p) in sorted(by_mod.items()):
            per_mod[mo] = cls_metrics(t, p)
        report["disease_per_modality"] = per_mod

    if mod_true:
        report["modality_id_accuracy"] = round(accuracy_score(mod_true, mod_pred), 4)
        report["modality_id_n"] = len(mod_true)

    if icd_exact:
        report["icd10_exact_match"] = round(sum(icd_exact) / len(icd_exact), 4)
        report["icd10_set_f1"]      = round(sum(icd_f1s) / len(icd_f1s), 4)
        report["icd10_n"]           = len(icd_exact)

    if cui_exact:
        report["cui_exact_match"] = round(sum(cui_exact) / len(cui_exact), 4)
        report["cui_n"]           = len(cui_exact)

    # ── print ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if "disease" in report:
        d = report["disease"]
        print(f"DISEASE   n={d['n']}  acc={d['accuracy']}  "
              f"macro-F1={d['macro_f1']}  weighted-F1={d['weighted_f1']}")
        print(f"{'  by modality':22s}{'acc':>8}{'macroF1':>10}{'n':>7}")
        for mo, mm in report["disease_per_modality"].items():
            print(f"    {mo:18s}{mm['accuracy']:>8}{mm['macro_f1']:>10}{mm['n']:>7}")
    if "modality_id_accuracy" in report:
        print(f"MODALITY  n={report['modality_id_n']}  "
              f"acc={report['modality_id_accuracy']}")
    if "icd10_exact_match" in report:
        print(f"ICD-10    n={report['icd10_n']}  "
              f"exact={report['icd10_exact_match']}  set-F1={report['icd10_set_f1']}")
    if "cui_exact_match" in report:
        print(f"UMLS CUI  n={report['cui_n']}  exact={report['cui_exact_match']}")
    print("=" * 60)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
