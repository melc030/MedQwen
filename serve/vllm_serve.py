"""
vLLM serving — high-throughput GPU inference (CUDA required, use on cloud).

Requires:
    pip install vllm

Usage:
    python serve/vllm_serve.py

The server exposes an OpenAI-compatible chat endpoint:
    POST http://localhost:8000/v1/chat/completions

Example curl:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "MedQwen",
        "messages": [{"role": "user", "content": "糖尿病的并发症有哪些？"}],
        "max_tokens": 256
      }'
"""

import sys
sys.path.append('..')

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from config import Config

cfg = Config()


def load_engine(use_adapter=True):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError('vllm not installed. Run: pip install vllm')

    adapter_path = cfg.best_dir if use_adapter and Path(cfg.best_dir).exists() else None

    if adapter_path:
        print(f'loading vLLM engine with LoRA adapter from {adapter_path}')
        # vLLM loads the base model and applies the LoRA adapter at request time
        llm = LLM(
            model=cfg.model_path,
            enable_lora=True,
            max_lora_rank=cfg.lora_rank,
            dtype='float16',
            trust_remote_code=True,
        )
    else:
        print('loading vLLM engine (base model only)')
        llm = LLM(
            model=cfg.model_path,
            dtype='float16',
            trust_remote_code=True,
        )

    return llm, adapter_path


def chat(llm, adapter_path, messages, max_tokens=256, temperature=0.7):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )

    if adapter_path:
        lora_request = LoRARequest('medqwen_adapter', 1, adapter_path)
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text


class ChatHandler(BaseHTTPRequestHandler):
    llm          = None
    adapter_path = None

    def log_message(self, format, *args):
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

        if not messages or messages[0].get('role') != 'system':
            messages = [{'role': 'system', 'content': cfg.system_prompt}] + messages

        try:
            answer = chat(
                ChatHandler.llm,
                ChatHandler.adapter_path,
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
    parser.add_argument('--port',       type=int, default=8000)
    parser.add_argument('--no-adapter', action='store_true')
    args = parser.parse_args()

    llm, adapter_path       = load_engine(use_adapter=not args.no_adapter)
    ChatHandler.llm          = llm
    ChatHandler.adapter_path = adapter_path

    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f'MedQwen vLLM server running at http://{args.host}:{args.port}')
    print(f'health check: GET http://localhost:{args.port}/health')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nserver stopped.')


if __name__ == '__main__':
    main()
