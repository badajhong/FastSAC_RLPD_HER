"""Compare an UNENABLED fused GRU prototype against the production pipeline.

Reports torch.equal and unchanged torch.testing.assert_close independently.
Any arithmetic difference is reported explicitly, not accepted as bit-exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmark_exact_online_encoder import require_idle_cuda
from prototype_fused_exact_gru import (
    build_policy_pair, compare_full_pipeline, comparison_precision, inputs, run_graph,
)


def timed_inference(policy, raw):
    def synchronize():
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)

    run_graph(policy, raw)
    synchronize()
    elapsed = []
    for _ in range(3):
        synchronize()
        start = time.perf_counter()
        run_graph(policy, raw)
        synchronize()
        elapsed.append(time.perf_counter() - start)
    return {"median_seconds": statistics.median(elapsed), "runs_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--allow-process-name", action="append", default=[])
    args = parser.parse_args()
    guard = require_idle_cuda(args.allow_process_name) if args.device.startswith("cuda") else None
    original_precision = torch.get_float32_matmul_precision()
    original_cudnn = torch.backends.cudnn.allow_tf32
    results = {}
    for strict in (False, True):
        with comparison_precision(strict_fp32=strict):
            policy, candidate = build_policy_pair(device=args.device)
            comparisons = compare_full_pipeline(policy, candidate)
            raw = inputs(policy)
            results["strict_fp32" if strict else "existing_precision"] = {
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "comparisons": comparisons,
                "production_pipeline": timed_inference(policy, raw),
                "experimental_pipeline": timed_inference(candidate, raw),
            }
            del policy, candidate, raw
    assert torch.get_float32_matmul_precision() == original_precision
    assert torch.backends.cudnn.allow_tf32 == original_cudnn
    print(json.dumps({
        "status": "experimental_only_NOT_ENABLED",
        "scope": "Synthetic full perception forward, not rollout/training throughput",
        "device": args.device, "guard": guard,
        "episodes": 3, "full_steps": 36, "latent_dim": 256,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
