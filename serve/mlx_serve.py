"""
MLX-LM serving — Apple Silicon native inference.

Requires:
    pip install mlx-lm

Usage:
    # Serve the fine-tuned LoRA adapter on top of base model
    python serve/mlx_serve.py

    # Or serve the base model directly (no adapter)
    python serve/mlx_serve.py --no-adapter

The server exposes an OpenAI-compatible chat endpoint:
    POST http://localhost:8080/v1/chat/completions

Example curl:
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "MedQwen",
        "messages": [{"role": "user", "content": "糖尿病的并发症有哪些？"}],
        "max_tokens": 256
      }'
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from config import Config

cfg = Config()


def load_model(use_adapter=True):
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError('mlx-lm not installed. Run: pip install mlx-lm')

    # MLX-LM requires a fused model — it cannot load raw PEFT adapters.
    # Run once to produce checkpoints/mlx-medqwen:
    #   python -m mlx_lm.fuse \
    #     --model Qwen2.5-1.5B-Instruct \
    #     --adapter-path checkpoints/best \
    #     --save-path checkpoints/mlx-medqwen
    mlx_fused = str(cfg.project_root / 'checkpoints' / 'mlx-medqwen')

    if use_adapter and Path(mlx_fused).exists():
        print(f'loading fused MLX model from {mlx_fused}')
        model, tokenizer = load(mlx_fused)
    else:
        if use_adapter and not Path(mlx_fused).exists():
            print(f'fused model not found at {mlx_fused}, falling back to base model')
            print('run: python -m mlx_lm.fuse --model Qwen2.5-1.5B-Instruct --adapter-path checkpoints/best --save-path checkpoints/mlx-medqwen')
        else:
            print('loading base model (no adapter)')
        model, tokenizer = load(cfg.model_path)

    return model, tokenizer


def chat(model, tokenizer, messages, max_tokens=256, temperature=0.7):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=temperature)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    return response


class ChatHandler(BaseHTTPRequestHandler):
    model     = None
    tokenizer = None

    def log_message(self, format, *args):
        # suppress default access log clutter
        pass

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_error(404)
            return

        length  = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length))

        messages    = payload.get('messages', [])
        max_tokens  = payload.get('max_tokens', 256)
        temperature = payload.get('temperature', 0.7)

        # prepend system prompt if not already present
        if not messages or messages[0].get('role') != 'system':
            messages = [{'role': 'system', 'content': cfg.system_prompt}] + messages

        try:
            answer = chat(
                ChatHandler.model,
                ChatHandler.tokenizer,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = {
                'id':      f'chatcmpl-{uuid.uuid4().hex[:8]}',
                'object':  'chat.completion',
                'created': int(time.time()),
                'model':   'MedQwen',
                'choices': [{
                    'index':         0,
                    'message':       {'role': 'assistant', 'content': answer},
                    'finish_reason': 'stop',
                }],
            }
            body = json.dumps(response, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(500, str(e))

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',       default='0.0.0.0')
    parser.add_argument('--port',       type=int, default=8080)
    parser.add_argument('--no-adapter', action='store_true')
    args = parser.parse_args()

    model, tokenizer = load_model(use_adapter=not args.no_adapter)
    ChatHandler.model     = model
    ChatHandler.tokenizer = tokenizer

    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f'MedQwen MLX server running at http://{args.host}:{args.port}')
    print(f'health check: GET http://localhost:{args.port}/health')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nserver stopped.')


if __name__ == '__main__':
    main()
