"""Qwen3-Omni preprocessing outputs and time per call at the process's OMP thread count.

usage: OMP_NUM_THREADS=16 PYTHONPATH=<tree> python3 preprocess_threads.py run --out t16.json [--speech N] [--videos N]
       python3 preprocess_threads.py compare t224.json t32.json t16.json

run: Qwen3OmniPreprocessor.call_impl, serially, on SeedTTS EN reference audio (the voice-clone input) and on the
Video-MME CI videos at the CI's video settings (fps 2, 128 frames, 401,408 pixels, max_seq_len 32768); records a hash
of every tensor in the output state and the wall time of each call. One fresh process per thread count.
compare: per input, the tensors whose hash differs from the first file's, and each file's time per call.
"""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.proto.request import OmniRequest, StagePayload

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS = "zhaochenyang20/seed-tts-eval-arrow"
VIDEOMME_CI = "zhaochenyang20/Video_MME_ci"
VIDEO_SETTINGS = {"video_fps": 2.0, "video_max_frames": 128, "video_max_pixels": 401408}


def tensor_hashes(value: object, prefix: str, hashes: dict[str, str]) -> None:
    if isinstance(value, torch.Tensor):
        flat = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        hashes[prefix] = (
            f"{value.dtype} {tuple(value.shape)} "
            + hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
        )
    elif isinstance(value, dict):
        for name, item in value.items():
            tensor_hashes(item, f"{prefix}.{name}", hashes)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensor_hashes(item, f"{prefix}[{index}]", hashes)
    else:
        pass


def run_one(
    preprocessor: Qwen3OmniPreprocessor, request_id: str, inputs: dict[str, object]
) -> dict[str, object]:
    payload = StagePayload(
        request_id=request_id, request=OmniRequest(inputs=inputs, params={}), data={}
    )
    start = time.perf_counter()
    result = asyncio.run(preprocessor.call_impl(payload))
    elapsed_ms = (time.perf_counter() - start) * 1e3
    hashes: dict[str, str] = {}
    tensor_hashes(result.data, "state", hashes)
    return {"id": request_id, "ms": elapsed_ms, "hashes": hashes}


def run(args: argparse.Namespace) -> None:
    preprocessor = Qwen3OmniPreprocessor(MODEL, max_seq_len=32768)
    records: dict[str, object] = {
        "threads": torch.get_num_threads(),
        "speech": [],
        "video": [],
    }
    for sample in load_seedtts_samples(SEEDTTS, args.speech, split="en"):
        inputs = {
            "messages": [{"role": "user", "content": sample.target_text}],
            "audios": [sample.ref_audio],
        }
        records["speech"].append(
            run_one(preprocessor, f"speech:{sample.sample_id}", inputs)
        )
    for sample in load_videomme_samples(args.videos, repo_id=VIDEOMME_CI):
        inputs = {
            "messages": [{"role": "user", "content": sample.prompt or sample.question}],
            "videos": [sample.video_path],
            **VIDEO_SETTINGS,
        }
        records["video"].append(
            run_one(preprocessor, f"video:{sample.sample_id}", inputs)
        )
    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(records, out)


def compare(args: argparse.Namespace) -> None:
    files = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.files]
    for kind in ("speech", "video"):
        reference = files[0][kind]
        for path, data in zip(args.files, files):
            records = data[kind]
            differing = [
                (mine["id"], name)
                for base, mine in zip(reference, records)
                for name in base["hashes"]
                if base["hashes"][name] != mine["hashes"].get(name)
            ]
            times = sorted(record["ms"] for record in records)
            print(
                f"{kind} {path} threads {data['threads']}: "
                f"{len(records)} inputs, {len(differing)} differing tensors, "
                f"ms per call p50 {statistics.median(times):.1f} mean {statistics.mean(times):.1f} "
                f"p95 {times[int(0.95 * len(times))]:.1f}"
            )
            for request_id, name in differing[:10]:
                print(f"  {request_id} {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--speech", type=int, default=None)
    run_parser.add_argument("--videos", type=int, default=None)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
