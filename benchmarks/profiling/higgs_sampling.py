# SPDX-License-Identifier: Apache-2.0
"""CUDA sampler validation and fixed-work A/B; see README.md in this directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "bench"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Diagnostic only; timings are instrumented.",
    )
    parser.add_argument(
        "--worker", choices=("baseline", "gumbel"), help=argparse.SUPPRESS
    )
    return parser


def _run_pair(args: argparse.Namespace, pair: int) -> dict:
    order = ("baseline", "gumbel") if pair % 2 == 0 else ("gumbel", "baseline")
    results = {}
    for backend in order:
        output_dir = args.output_dir / f"pair_{pair + 1}" / backend
        command = [
            sys.executable,
            "-m",
            "benchmarks.profiling.higgs_sampling",
            "--mode",
            args.mode,
            "--worker",
            backend,
            "--output-dir",
            str(output_dir),
            "--batch-sizes",
            *map(str, args.batch_sizes),
            "--iterations",
            str(args.iterations),
            "--blocks",
            str(args.blocks),
            "--draws",
            str(args.draws),
        ]
        if args.trace:
            command.append("--trace")
        print(f"Pair {pair + 1}: {backend} {args.mode}", flush=True)
        subprocess.run(command, check=True)
        results[backend] = json.loads((output_dir / "results.json").read_text())
    if args.mode == "validate":
        assert (
            results["baseline"]["deterministic_signature"]
            == results["gumbel"]["deterministic_signature"]
        ), "Seeded/greedy/FSM outputs changed between independently initialized modes"
    return {"pair": pair + 1, "order": list(order), "results": results}


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if min(*args.batch_sizes, args.iterations, args.blocks, args.pairs) < 1:
        parser.error("batch sizes, iterations, blocks, and pairs must be positive")
    if args.draws < 100_000:
        parser.error("--draws must be >= 100000")
    if args.trace and args.mode != "bench":
        parser.error("--trace is only for diagnostic bench runs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.worker:
        # Import the production sampler only after selecting the worker's mode.
        os.environ["SGLANG_OMNI_HIGGS_USE_GUMBEL_SAMPLE"] = str(
            int(args.worker == "gumbel")
        )
        os.environ["SGLANG_OMNI_HIGGS_PROFILE_SAMPLING"] = str(int(args.trace))
        from benchmarks.profiling.higgs_sampling_cases import run_worker

        results = run_worker(args)
    else:
        results = {
            "mode": args.mode,
            "instrumented": args.trace,
            "pairs": [
                _run_pair(args, pair)
                for pair in range(args.pairs if args.mode == "bench" else 1)
            ],
        }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
