"""
Latency + memory benchmark against an OpenAI-compatible inference server
(works with both mlx_serve.py on :8080 and vllm_serve.py on :8000).

Reports per-request:
  - TTFT       (time to first token, seconds)
  - decode tps (tokens/sec after first token)
  - total      (end-to-end seconds)
Reports peak memory sampled during the run:
  - CUDA: nvidia-smi memory.used (MiB)
  - MLX : mlx.core.metal.get_peak_memory() (MiB) — only when running locally
          on the same Mac as the server

Usage:
    python src/eval/benchmark.py                              # MLX defaults
    python src/eval/benchmark.py --url http://localhost:8000 --model medqwen
    python src/eval/benchmark.py --n 20 --max-tokens 256
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

cfg = Config()


def gpu_mem_mib():
    """Peak GPU memory in MiB, or None if no probe is available."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        return max(int(x) for x in out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        import mlx.core as mx
        return mx.metal.get_peak_memory() / (1024 * 1024)
    except (ImportError, AttributeError):
        return None


def stream_one(url, model, question, max_tokens):
    """Stream a single completion, return (ttft, decode_tps, total, n_tokens)."""
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': cfg.system_prompt},
            {'role': 'user',   'content': question},
        ],
        'max_tokens': max_tokens,
        'temperature': 0.0,
        'stream': True,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0

    with requests.post(f'{url}/v1/chat/completions',
                       json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b'data: '):
                continue
            chunk = raw[6:]
            if chunk == b'[DONE]':
                break
            delta = json.loads(chunk)['choices'][0].get('delta', {})
            if not delta.get('content'):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_tokens += 1

    total = time.perf_counter() - t0
    decode_tps = (n_tokens - 1) / (total - ttft) if ttft and n_tokens > 1 else 0.0
    return ttft or total, decode_tps, total, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url',        default='http://localhost:8080')
    ap.add_argument('--model',      default='medqwen')
    ap.add_argument('--n',          type=int, default=10, help='requests to send')
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--warmup',     type=int, default=2)
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    with open(cfg.test_jsonl) as f:
        samples = [json.loads(l) for l in f]
    questions = [s['messages'][1]['content']
                 for s in random.sample(samples, args.n + args.warmup)]

    print(f'server : {args.url}  model={args.model}')
    print(f'prompts: {args.n} (+{args.warmup} warmup)  max_tokens={args.max_tokens}')

    mem_before = gpu_mem_mib()

    for q in questions[:args.warmup]:
        stream_one(args.url, args.model, q, args.max_tokens)

    rows = []
    for q in questions[args.warmup:]:
        rows.append(stream_one(args.url, args.model, q, args.max_tokens))

    mem_after = gpu_mem_mib()

    ttfts  = [r[0] for r in rows]
    tps    = [r[1] for r in rows if r[1] > 0]
    totals = [r[2] for r in rows]
    toks   = [r[3] for r in rows]

    def pct(xs, p):
        s = sorted(xs)
        return s[min(len(s) - 1, int(len(s) * p))]

    print(f'\n{"─"*48}')
    print(f'  LATENCY  (n={len(rows)})')
    print(f'{"─"*48}')
    print(f'  TTFT       mean {sum(ttfts)/len(ttfts):.3f}s  p50 {pct(ttfts,.5):.3f}s  p95 {pct(ttfts,.95):.3f}s')
    print(f'  decode tps mean {sum(tps)/len(tps):.1f}    p50 {pct(tps,.5):.1f}    p95 {pct(tps,.95):.1f}')
    print(f'  total      mean {sum(totals)/len(totals):.3f}s  p50 {pct(totals,.5):.3f}s  p95 {pct(totals,.95):.3f}s')
    print(f'  tokens/req mean {sum(toks)/len(toks):.1f}')

    print(f'\n{"─"*48}')
    print(f'  MEMORY')
    print(f'{"─"*48}')
    if mem_before is None:
        print('  no probe available (no nvidia-smi, no local mlx).')
        print('  on the server host run:  watch -n0.5 nvidia-smi   (CUDA)')
    else:
        delta = mem_after - mem_before
        print(f'  peak before  : {mem_before:.0f} MiB')
        print(f'  peak after   : {mem_after:.0f} MiB   (Δ {delta:+.0f})')


if __name__ == '__main__':
    main()