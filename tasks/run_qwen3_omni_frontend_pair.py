# SPDX-License-Identifier: Apache-2.0
"""Run a matched full-corpus frontend pair with frozen serving and scorer trees."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

TACTIC_CACHE_SHA256 = "c1fc08713750c4f53c1d3432f41377523496220290bac6da2739b609f0676cfa"
SIMILARITY_SHA256 = "c089056ff6e5023961a961f001a1a716cb71031e8b10a805a63a5115cc62f577"


def main(
    output: Path,
    control: Path,
    candidate: Path,
    tools: Path,
    scorer: Path,
    checkpoint: Path,
    asr_checkpoint: Path,
    cache: Path,
    concurrency: int,
) -> None:
    assert not output.exists()
    assert concurrency > 0
    output.mkdir()
    sources = {"stock": control, "fixed": candidate, "tools": tools, "scorer": scorer}
    pins = {
        name: subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        for name, source in sources.items()
    }
    tactic_files = [
        path
        for path in cache.rglob("rank_tp0_pp0_dp0.json")
        if sha256(path.read_bytes()).hexdigest() == TACTIC_CACHE_SHA256
    ]
    assert len(tactic_files) == 1
    tactic_file = tactic_files[0]
    cache_hash = TACTIC_CACHE_SHA256
    similarity_path = scorer / "benchmarks/metrics/speaker_similarity.py"
    similarity_hash = sha256(similarity_path.read_bytes()).hexdigest()
    assert similarity_hash == SIMILARITY_SHA256
    with (output / "source-pins.json").open("x") as receipt:
        json.dump(
            {
                "sources": pins,
                "checkpoint": checkpoint.name,
                "asr_checkpoint": asr_checkpoint.name,
                "tactic_cache_sha256": cache_hash,
                "similarity_sha256": similarity_hash,
                "comparison": "stock is the pinned control; fixed adds the request-owned audio frontend",
            },
            receipt,
            indent=2,
        )
    cache_relative = tactic_file.relative_to(cache)
    for arm, source in (("stock", control), ("fixed", candidate)):
        arm_cache = output / f"{arm}-cache"
        destination = arm_cache / cache_relative
        destination.parent.mkdir(parents=True)
        shutil.copyfile(tactic_file, destination)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = f"{source}:{tools / 'tasks'}"
        environment["OMP_NUM_THREADS"] = "8"
        print(f"Starting {arm}: {pins[arm]}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(tools / "tasks/run_qwen3_omni_full_default_qualification.py"),
                str(output / arm),
                str(checkpoint),
                "--gpu-index",
                "0",
                "--concurrency",
                str(concurrency),
                "--cache-directory",
                str(arm_cache),
            ],
            cwd=source,
            env=environment,
            check=True,
        )
        assert sha256(destination.read_bytes()).hexdigest() == cache_hash
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{scorer}:{tools / 'tasks'}"
    environment["OMP_NUM_THREADS"] = "8"
    print("Starting all quality scorers", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(tools / "tasks/score_qwen3_omni_full_default_qualification.py"),
            str(output),
            str(asr_checkpoint),
            "--gpu-index",
            "0",
        ],
        cwd=scorer,
        env=environment,
        check=True,
    )
    assert sha256(similarity_path.read_bytes()).hexdigest() == similarity_hash
    for name, source in sources.items():
        assert (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
            == pins[name]
        )
    print("Completed both full-corpus arms and all quality scorers", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "control",
        "candidate",
        "tools",
        "scorer",
        "checkpoint",
        "asr_checkpoint",
        "cache",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("--concurrency", type=int, required=True)
    arguments = parser.parse_args()
    main(
        arguments.output,
        arguments.control,
        arguments.candidate,
        arguments.tools,
        arguments.scorer,
        arguments.checkpoint,
        arguments.asr_checkpoint,
        arguments.cache,
        arguments.concurrency,
    )
else:
    pass
