"""
vLLM serving for MedQwen-VL (Qwen2.5-VL-3B + LoRA). CUDA required — cloud only.

Requirements:
    pip install vllm

Usage:
    python src/serve/vllm_serve_vl.py
    python src/serve/vllm_serve_vl.py --no-adapter      # serve base VL model

Exposes an OpenAI-compatible chat endpoint that accepts image_url content:
    POST http://localhost:8000/v1/chat/completions

Point the VL Gradio app at it:
    INFERENCE_URL=http://<vm-ip>:8000 python src/app_vl.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

cfg = Config()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host',       default='0.0.0.0')
    ap.add_argument('--port',       type=int, default=8000)
    ap.add_argument('--no-adapter', action='store_true')
    ap.add_argument('--dtype',      default='bfloat16')
    ap.add_argument('--max-pixels', type=int, default=cfg.vl_max_pixels)
    args = ap.parse_args()

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model',            cfg.vl_model_path,
        '--port',             str(args.port),
        '--host',             args.host,
        '--dtype',            args.dtype,
        '--trust-remote-code',
        # cap visual tokens to match training-time budget
        '--mm-processor-kwargs', f'{{"max_pixels": {args.max_pixels}}}',
    ]

    if not args.no_adapter and Path(cfg.vl_best_dir).exists():
        cmd += ['--enable-lora', '--lora-modules', f'medqwen-vl={cfg.vl_best_dir}']
        print(f'serving {cfg.vl_model_path} + LoRA from {cfg.vl_best_dir}')
        print('  model name for API calls: "medqwen-vl"')
    else:
        print(f'serving {cfg.vl_model_path} (base model only)')
        print('  model name for API calls: the base model path')

    print(f'vLLM VL server starting at http://{args.host}:{args.port}')
    print(f'point Gradio at: INFERENCE_URL=http://<vm-ip>:{args.port} python src/app_vl.py')
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
