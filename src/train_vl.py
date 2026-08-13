"""
LoRA fine-tuning of Qwen2.5-VL-3B-Instruct on multimodal medical VQA.

The VL counterpart of train.py. Same recipe (frozen base + LoRA on the LLM,
gradient accumulation, cosine schedule, early stopping), with the additions a
vision-language model needs:

- AutoProcessor (tokenizer + image processor) instead of just a tokenizer
- images turned into pixel_values / image_grid_thw by the processor
- LoRA restricted to the language model (vision tower stays frozen) — Qwen2.5-VL
  vision blocks reuse gate/up/down_proj names, so we select full module names
  and drop anything under 'visual.'
- collate concatenates pixel_values / image_grid_thw (they're flattened
  per-image) while padding input_ids / labels / attention_mask

Data: data/multimodal/{train,val}.jsonl from build_vqa.py (Qwen chat format).
Images: resolved against cfg.images_root (set MEDQWEN_IMAGES_ROOT locally).

Usage:
    python src/train_vl.py
"""

import os
import math
import time
import json
from pathlib import Path
from functools import partial

import torch
import peft
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from config import Config

cfg = Config()

# bf16 needs a capable GPU; fall back to fp32 for CPU/MPS smoke tests.
DTYPE = cfg.vl_dtype if cfg.device.startswith('cuda') else torch.float32


# ── Dataset ───────────────────────────────────────────────────────────────────

class MMDataset(Dataset):
    """Each item -> input_ids, attention_mask, labels, pixel_values, image_grid_thw."""

    def __init__(self, jsonl_path, processor):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(l) for l in f]
        self.processor   = processor
        self.tokenizer   = processor.tokenizer
        self.images_root = Path(cfg.images_root)
        # assistant header marks where the answer (loss region) begins
        self._assistant_ids = self.tokenizer.encode(
            '<|im_start|>assistant', add_special_tokens=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        messages = sample['messages']
        image    = Image.open(self.images_root / sample['image']).convert('RGB')

        # render chat text (vision placeholder kept un-tokenized), then let the
        # processor expand <|image_pad|> to match the image and emit pixel_values
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        enc = self.processor(
            text=[text], images=[image], padding=False, return_tensors='pt')

        input_ids = enc['input_ids'][0]
        attn_mask = enc['attention_mask'][0]

        # labels: mask everything up to (and including) the assistant header,
        # so loss is computed only on the answer. Image tokens live in the user
        # turn (before the header) and are therefore masked out automatically.
        labels = input_ids.clone()
        seq, a = input_ids.tolist(), self._assistant_ids
        mask_until = 0
        for i in range(len(seq) - len(a)):
            if seq[i:i + len(a)] == a:
                mask_until = i + len(a)
        labels[:mask_until] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attn_mask,
            'labels':         labels,
            'pixel_values':   enc['pixel_values'].to(DTYPE),
            'image_grid_thw': enc['image_grid_thw'],
        }


def collate_fn(batch, pad_id):
    n = len(batch)
    max_len = max(b['input_ids'].size(0) for b in batch)

    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((n, max_len), dtype=torch.long)
    labels    = torch.full((n, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b['input_ids'].size(0)
        input_ids[i, :L] = b['input_ids']
        attn_mask[i, :L] = b['attention_mask']
        labels[i, :L]    = b['labels']

    # pixel_values / image_grid_thw are flattened per-image -> concatenate
    pixel_values   = torch.cat([b['pixel_values']   for b in batch], dim=0)
    image_grid_thw = torch.cat([b['image_grid_thw'] for b in batch], dim=0)

    return {
        'input_ids':      input_ids,
        'attention_mask': attn_mask,
        'labels':         labels,
        'pixel_values':   pixel_values,
        'image_grid_thw': image_grid_thw,
    }


def to_device(batch):
    return {k: v.to(cfg.device) for k, v in batch.items()}


# ── LoRA target selection (language model only) ────────────────────────────────

def llm_lora_targets(model):
    """Full module names of LLM linear projections, excluding the vision tower."""
    leaves = set(cfg.lora_target_modules)
    targets = {
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and name.split('.')[-1] in leaves
        and 'visual' not in name
    }
    return sorted(targets)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            loss = model(**to_device(batch)).loss
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    print(f'device: {cfg.device}  dtype: {DTYPE}')

    # ── Processor ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        cfg.vl_model_path,
        min_pixels=cfg.vl_min_pixels,
        max_pixels=cfg.vl_max_pixels,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = 'right'
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # ── Base model ───────────────────────────────────────────────
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.vl_model_path,
        dtype=DTYPE,
        device_map='auto',
        trust_remote_code=True,
    )
    if cfg.device.startswith('cuda'):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    # ── LoRA (LLM only; vision tower frozen) ─────────────────────
    targets = llm_lora_targets(model)
    print(f'LoRA targets: {len(targets)} LLM linear modules (visual excluded)')
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
    )
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data ─────────────────────────────────────────────────────
    collate = partial(collate_fn, pad_id=pad_id)
    train_dataset = MMDataset(cfg.mm_train_jsonl, processor)
    valid_dataset = MMDataset(cfg.mm_valid_jsonl, processor)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.vl_batch_size,
        shuffle=True, collate_fn=collate, drop_last=True, num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.vl_batch_size,
        shuffle=False, collate_fn=collate, drop_last=True, num_workers=4,
    )

    # ── Optimizer & scheduler ────────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=cfg.vl_learning_rate, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.vl_grad_accum_steps)
    total_steps     = steps_per_epoch * cfg.vl_epochs
    warmup_steps    = int(cfg.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training ─────────────────────────────────────────────────
    os.makedirs(cfg.vl_best_dir, exist_ok=True)

    best_eval_loss   = float('inf')
    global_step      = 0
    patience_counter = 0
    tic              = time.time()

    for epoch in range(1, cfg.vl_epochs + 1):
        print(f'\n=== Epoch {epoch}/{cfg.vl_epochs} ===')
        model.train()
        optimizer.zero_grad()
        loss_buf = []
        stop_training = False

        for batch_idx, batch in enumerate(train_loader, start=1):
            loss = model(**to_device(batch)).loss

            raw_loss = loss.item()
            (loss / cfg.vl_grad_accum_steps).backward()
            loss_buf.append(raw_loss)

            should_step = (
                batch_idx % cfg.vl_grad_accum_steps == 0
                or batch_idx == len(train_loader)
            )
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % cfg.logging_steps == 0:
                elapsed   = time.time() - tic
                avg_loss  = sum(loss_buf) / len(loss_buf)
                speed     = cfg.logging_steps / elapsed
                remaining = (total_steps - global_step) / speed
                print(
                    f'step {global_step}/{total_steps} '
                    f'({100*global_step/total_steps:.1f}%) | '
                    f'loss {avg_loss:.4f} | '
                    f'lr {scheduler.get_last_lr()[0]:.2e} | '
                    f'ETA {remaining/60:.1f}min'
                )
                loss_buf = []
                tic      = time.time()

            if global_step % cfg.save_steps == 0:
                eval_loss = evaluate(model, valid_loader)
                print(f'eval loss: {eval_loss:.4f}')
                if eval_loss < best_eval_loss:
                    best_eval_loss   = eval_loss
                    patience_counter = 0
                    model.save_pretrained(cfg.vl_best_dir)
                    processor.save_pretrained(cfg.vl_best_dir)
                    print(f'best model saved (loss={best_eval_loss:.4f})')
                else:
                    patience_counter += 1
                    print(f'no improvement ({patience_counter}/{cfg.early_stopping_patience})')
                    if patience_counter >= cfg.early_stopping_patience:
                        print('early stopping triggered — stopping training.')
                        stop_training = True
                        break
                tic = time.time()

        if stop_training:
            break

    # final eval + save
    eval_loss = evaluate(model, valid_loader)
    print(f'\nfinal eval loss: {eval_loss:.4f}')
    if eval_loss < best_eval_loss:
        model.save_pretrained(cfg.vl_best_dir)
        processor.save_pretrained(cfg.vl_best_dir)
    print('training complete.')


if __name__ == '__main__':
    train()
