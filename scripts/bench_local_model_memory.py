#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import resource
import tracemalloc

import psutil

from AgentLaboratory.inference import query_model, local_model_runtime_config


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    tracemalloc.start()
    tracemalloc.reset_peak()

    os.environ.setdefault('AGENTLAB_DEVICE', 'cpu')
    os.environ.setdefault('AGENTLAB_LOCAL_MAX_NEW_TOKENS', '128')

    before = rss_mb()
    t0 = time.time()
    answer = query_model(
        model_str=os.getenv('AGENTLAB_BENCH_MODEL', 'local:Qwen/Qwen2.5-0.5B-Instruct'),
        prompt='Explain how to reduce memory during local LLM inference.',
        system_prompt='Be concise.',
        tries=1,
        timeout=120,
        temp=0.0,
        print_cost=False,
    )
    dt = time.time() - t0
    cur_py, peak_py = tracemalloc.get_traced_memory()

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram = 0.0
    except Exception:
        peak_vram = 0.0

    print('runtime_config=', local_model_runtime_config())
    print('answer_len_chars=', len(answer))
    print('time_s=', round(dt, 3))
    print('rss_before_mb=', round(before, 3))
    print('rss_after_mb=', round(rss_mb(), 3))
    print('peak_rss_mb=', round(peak_rss_mb(), 3))
    print('py_current_mb=', round(cur_py / (1024 ** 2), 3))
    print('py_peak_mb=', round(peak_py / (1024 ** 2), 3))
    print('peak_vram_mb=', round(peak_vram, 3))


if __name__ == '__main__':
    main()
