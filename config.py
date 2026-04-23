import os
import torch
from pathlib import Path

class Config:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent

        # ── Device ────────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # ── Model ─────────────────────────────────────────────────
        # Using Instruct variant: already understands chat template format
        self.hf_model_id  = 'Qwen/Qwen2.5-3B-Instruct'
        self.model_path   = str(self.project_root / 'Qwen2.5-3B-Instruct')

        # ── Data ──────────────────────────────────────────────────
        self.train_raw    = str(self.project_root / 'data' / 'medical_train.txt')
        self.valid_raw    = str(self.project_root / 'data' / 'medical_valid.txt')
        self.train_jsonl  = str(self.project_root / 'data' / 'medical_train.jsonl')
        self.valid_jsonl  = str(self.project_root / 'data' / 'medical_valid.jsonl')

        # ── LoRA ──────────────────────────────────────────────────
        self.lora_rank      = 16
        self.lora_alpha     = 32        # typically 2x rank
        self.lora_dropout   = 0.05
        self.lora_target_modules = [    # Qwen2.5 attention projection layers
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ]

        # ── Training ──────────────────────────────────────────────
        self.batch_size             = 1
        self.grad_accum_steps       = 8   # effective batch = 8
        self.epochs                 = 5
        self.learning_rate          = 2e-4
        self.weight_decay           = 0.01
        self.warmup_ratio           = 0.05
        self.max_seq_len            = 512
        self.max_grad_norm          = 1.0
        self.logging_steps          = 10
        self.save_steps             = 500
        self.early_stopping_patience = 5  # stop if eval loss doesn't improve for 5 evals

        # ── Output ────────────────────────────────────────────────
        self.save_dir     = str(self.project_root / 'checkpoints')
        self.best_dir     = str(self.project_root / 'checkpoints' / 'best')

        # ── System prompt ─────────────────────────────────────────
        self.system_prompt = '你是一个专业的医疗问答助手，请根据用户的问题给出准确、简洁的医疗建议。'

        # ── Data conversion ───────────────────────────────────────
        self.converter_threads  = 4   # consumer thread count
        self.queue_max_size     = 50  # cap in-memory task queue


if __name__ == '__main__':
    cfg = Config()
    print(f'device : {cfg.device}')
    print(f'model  : {cfg.model_path}')
    print(f'train  : {cfg.train_jsonl}')
