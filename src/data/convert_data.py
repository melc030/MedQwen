"""
Convert raw medical Q&A txt → Qwen chat JSONL format.

Input format (medical_train.txt):
    question\n
    answer\n
    \n

Output format (medical_train.jsonl):
    {"id": 0, "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "question"},
        {"role": "assistant", "content": "answer"}
    ]}

Uses producer-consumer pattern (ref: Deepseek-finetune/generateData/producerProcesserMode.py):
- Producer  : reads Q&A pairs from txt, skips already-converted IDs (crash recovery)
- Consumers : convert each pair to chat JSONL (N threads)
- Writer    : streams results to output file
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import threading
import json
import os
from queue import Queue

from config import Config

cfg = Config()

task_queue   = Queue()
result_queue = Queue()
_STOP_SIGNAL = object()   # sentinel to signal consumers to stop


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_pairs(file_path):
    """Yield (question, answer) pairs from the raw txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]

    i = 0
    while i < len(lines):
        # skip blank lines between pairs
        if not lines[i].strip():
            i += 1
            continue
        question = lines[i].strip()
        i += 1
        if i < len(lines) and lines[i].strip():
            answer = lines[i].strip()
            i += 1
            yield question, answer


def load_finished_ids(target_path):
    """Read already-converted IDs from output file for crash recovery."""
    finished = set()
    if os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'id' in data:
                        finished.add(data['id'])
                except json.JSONDecodeError:
                    continue
    return finished


def to_chat_format(pair_id, question, answer):
    """Convert a Q&A pair to Qwen chat JSONL format."""
    return {
        'id': pair_id,
        'messages': [
            {'role': 'system',    'content': cfg.system_prompt},
            {'role': 'user',      'content': question},
            {'role': 'assistant', 'content': answer},
        ]
    }


# ── Producer ──────────────────────────────────────────────────────────────────

def producer(source_path, target_path, num_consumers):
    finished_ids = load_finished_ids(target_path)
    if finished_ids:
        print(f'[producer] crash recovery: skipping {len(finished_ids)} already converted pairs')

    pair_id = 0
    for question, answer in read_pairs(source_path):
        if pair_id not in finished_ids:
            # throttle: don't flood the queue
            while task_queue.qsize() >= cfg.queue_max_size:
                threading.Event().wait(0.1)
            task_queue.put((pair_id, question, answer))
        pair_id += 1

    # send stop signals to each consumer
    for _ in range(num_consumers):
        task_queue.put(_STOP_SIGNAL)

    print(f'[producer] done — queued {pair_id - len(finished_ids)} pairs')


# ── Consumer ──────────────────────────────────────────────────────────────────

def consumer(thread_idx):
    while True:
        item = task_queue.get(block=True)
        if item is _STOP_SIGNAL:
            result_queue.put(_STOP_SIGNAL)   # forward stop signal to writer
            break
        pair_id, question, answer = item
        try:
            record = to_chat_format(pair_id, question, answer)
            result_queue.put((True, record))
        except Exception as e:
            result_queue.put((False, f'[consumer-{thread_idx}] error on id={pair_id}: {e}'))


# ── Writer ────────────────────────────────────────────────────────────────────

def writer(target_path, num_consumers):
    stops_received = 0
    written = 0
    # append mode — safe for crash recovery
    with open(target_path, 'a', encoding='utf-8') as f:
        while True:
            item = result_queue.get(block=True)
            if item is _STOP_SIGNAL:
                stops_received += 1
                if stops_received == num_consumers:
                    break
                continue
            success, payload = item
            if success:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
                f.flush()
                written += 1
                if written % 1000 == 0:
                    print(f'[writer] {written} records written...')
            else:
                print(f'[writer] warning: {payload}')
    print(f'[writer] done — {written} records written to {target_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def convert(source_path, target_path, num_consumers=None):
    if num_consumers is None:
        num_consumers = cfg.converter_threads

    print(f'converting: {source_path} → {target_path}')
    print(f'threads: 1 producer + {num_consumers} consumers + 1 writer')

    # producer thread
    p = threading.Thread(target=producer, args=(source_path, target_path, num_consumers), daemon=True)
    p.start()

    # consumer threads
    for i in range(num_consumers):
        t = threading.Thread(target=consumer, args=(i,), daemon=True)
        t.start()

    # writer runs on main thread so we block until done
    writer(target_path, num_consumers)
    p.join()


if __name__ == '__main__':
    convert(cfg.train_raw, cfg.train_jsonl)
    convert(cfg.valid_raw, cfg.valid_jsonl)
    print('all done.')
