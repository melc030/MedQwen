"""
LoRA fine-tuning of Qwen2.5-7B-Instruct on Chinese medical dialogue.

Training stack:
- HuggingFace Transformers  : model + tokenizer
- PEFT                      : LoRA adapters
- torch.autocast            : FP16 mixed precision (CUDA / MPS)
- gradient checkpointing    : reduces memory ~30-50%
- gradient accumulation     : effective batch size without OOM
"""

import os
import math
import time

import torch
import peft
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from config import Config

cfg = Config()


# ── Mixed precision context (CUDA / MPS / CPU) ────────────────────────────────

from contextlib import contextmanager

@contextmanager
def autocast(device):
    if device.startswith('cuda'):
        with torch.autocast(device_type='cuda'):
            yield
    elif device.startswith('mps'):
        with torch.autocast(device_type='mps'):
            yield
    else:
        yield


# ── Dataset ───────────────────────────────────────────────────────────────────

class MedDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        import json
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]['messages']

        # apply Qwen2.5 chat template, add generation prompt so the model
        # learns to produce the assistant turn
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self.tokenizer(
            text,
            max_length=cfg.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)

        # labels: mask everything up to (and including) the last user turn
        # so loss is only computed on the assistant response
        labels = input_ids.clone()
        # find assistant token position using chat template markers
        assistant_token = self.tokenizer.encode('<|im_start|>assistant', add_special_tokens=False)
        seq = input_ids.tolist()
        mask_until = 0
        for i in range(len(seq) - len(assistant_token)):
            if seq[i:i+len(assistant_token)] == assistant_token:
                mask_until = i + len(assistant_token)
        labels[:mask_until] = -100

        return input_ids, labels


def collate_fn(batch):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)

    padded_input = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    padded_labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_ids_list, labels_list)):
        padded_input[i, :inp.size(0)]  = inp
        padded_labels[i, :lab.size(0)] = lab

    return padded_input, padded_labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(cfg.device)
            labels    = labels.to(cfg.device)
            with autocast(cfg.device):
                loss = model(input_ids=input_ids, labels=labels).loss
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print(f'device: {cfg.device}')

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    # ── Base model ───────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    # gradient checkpointing is unstable on MPS — only enable for CUDA
    if cfg.device.startswith('cuda'):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    # ── LoRA ─────────────────────────────────────────────────────
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(cfg.device)

    # ── Data ─────────────────────────────────────────────────────
    train_dataset = MedDataset(cfg.train_jsonl, tokenizer)
    valid_dataset = MedDataset(cfg.valid_jsonl, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, drop_last=True,
    )

    # ── Optimizer & scheduler ────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = int(cfg.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training ─────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.best_dir, exist_ok=True)

    best_eval_loss = float('inf')
    global_step    = 0
    tic            = time.time()

    for epoch in range(1, cfg.epochs + 1):
        print(f'\n=== Epoch {epoch}/{cfg.epochs} ===')
        model.train()
        optimizer.zero_grad()
        loss_buf = []

        for batch_idx, (input_ids, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(cfg.device)
            labels    = labels.to(cfg.device)

            with autocast(cfg.device):
                loss = model(input_ids=input_ids, labels=labels).loss

            raw_loss = loss.item()
            loss     = loss / cfg.grad_accum_steps
            loss.backward()
            loss_buf.append(raw_loss)

            should_step = (
                batch_idx % cfg.grad_accum_steps == 0
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
                    best_eval_loss = eval_loss
                    model.save_pretrained(cfg.best_dir)
                    tokenizer.save_pretrained(cfg.best_dir)
                    print(f'best model saved (loss={best_eval_loss:.4f})')
                tic = time.time()

    # final eval + save
    eval_loss = evaluate(model, valid_loader)
    print(f'\nfinal eval loss: {eval_loss:.4f}')
    if eval_loss < best_eval_loss:
        model.save_pretrained(cfg.best_dir)
        tokenizer.save_pretrained(cfg.best_dir)
    print('training complete.')


if __name__ == '__main__':
    train()
