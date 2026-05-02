"""
Re-split the medical Q&A dataset into 80/10/10 train/val/test.

Reads from the original medical_train.txt and medical_valid.txt,
combines them, shuffles, and saves new splits alongside the old files.

Output files (in data/):
    medical_train_v2.txt   80%  train
    medical_valid_v2.txt   10%  validation  (used for early stopping)
    medical_test_v2.txt    10%  test        (held out for final evaluation only)
"""

import random
from pathlib import Path

SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# test = remaining 10%

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'

def read_pairs(path):
    """Parse alternating Q/A lines separated by blank lines."""
    pairs = []
    lines = Path(path).read_text(encoding='utf-8').strip().splitlines()
    i = 0
    while i < len(lines):
        q = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        a = lines[i].strip()
        i += 1
        # skip blank separator
        if i < len(lines) and lines[i].strip() == '':
            i += 1
        if q and a:
            pairs.append((q, a))
    return pairs

def write_pairs(pairs, path):
    with open(path, 'w', encoding='utf-8') as f:
        for q, a in pairs:
            f.write(q + '\n')
            f.write(a + '\n')
            f.write('\n')
    print(f'wrote {len(pairs):,} pairs → {path}')


# ── Load & combine ────────────────────────────────────────────────────────────
train_pairs = read_pairs(DATA_DIR / 'medical_train.txt')
valid_pairs = read_pairs(DATA_DIR / 'medical_valid.txt')
all_pairs   = train_pairs + valid_pairs

print(f'total pairs: {len(all_pairs):,}')

# ── Shuffle ───────────────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(all_pairs)

# ── Split ─────────────────────────────────────────────────────────────────────
n       = len(all_pairs)
n_train = int(n * TRAIN_FRAC)
n_val   = int(n * VAL_FRAC)

train = all_pairs[:n_train]
val   = all_pairs[n_train:n_train + n_val]
test  = all_pairs[n_train + n_val:]

print(f'train: {len(train):,} ({len(train)/n*100:.1f}%)')
print(f'val  : {len(val):,}   ({len(val)/n*100:.1f}%)')
print(f'test : {len(test):,}   ({len(test)/n*100:.1f}%)')

# ── Write ─────────────────────────────────────────────────────────────────────
write_pairs(train, DATA_DIR / 'medical_train_v2.txt')
write_pairs(val,   DATA_DIR / 'medical_valid_v2.txt')
write_pairs(test,  DATA_DIR / 'medical_test_v2.txt')

print('done. old files untouched.')
