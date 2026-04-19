"""
vLLM serving — high-throughput GPU inference (CUDA required, use on cloud).

Uses vLLM's built-in OpenAI-compatible API server.
Do NOT run this on Mac — vLLM requires CUDA.

Requirements:
    pip install vllm

Usage:
    python serve/vllm_serve.py

    # or directly via vLLM CLI:
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen2.5-7B-Instruct \\
        --enable-lora \\
        --lora-modules medqwen=checkpoints/best \\
        --port 8000 \\
        --dtype float16 \\
        --trust-remote-code

The server exposes an OpenAI-compatible chat endpoint:
    POST http://localhost:8000/v1/chat/completions

Point Gradio at this server:
    INFERENCE_URL=http://<vm-ip>:8000 python app.py

Example curl:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "medqwen",
        "messages": [{"role": "user", "content": "糖尿病的并发症有哪些？"}],
        "max_tokens": 256
      }'
"""

import subprocess
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

cfg = Config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',       default='0.0.0.0')
    parser.add_argument('--port',       type=int, default=8000)
    parser.add_argument('--no-adapter', action='store_true')
    parser.add_argument('--dtype',      default='float16')
    args = parser.parse_args()

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model',            cfg.model_path,
        '--port',             str(args.port),
        '--host',             args.host,
        '--dtype',            args.dtype,
        '--trust-remote-code',
    ]

    if not args.no_adapter and Path(cfg.best_dir).exists():
        cmd += [
            '--enable-lora',
            '--lora-modules', f'medqwen={cfg.best_dir}',
        ]
        print(f'serving {cfg.model_path} + LoRA adapter from {cfg.best_dir}')
    else:
        print(f'serving {cfg.model_path} (base model only)')

    print(f'vLLM server starting at http://{args.host}:{args.port}')
    print(f'point Gradio at: INFERENCE_URL=http://<vm-ip>:{args.port} python app.py')

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
