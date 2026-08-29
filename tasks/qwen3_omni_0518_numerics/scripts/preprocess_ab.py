# SPDX-License-Identifier: Apache-2.0
"""Time the CPU preprocessing of one video request, library set A against B.

The Qwen3-Omni preprocessing stage does two heavy things per request: decode
and resize the video frames (sglang_omni.preprocessing.video, torchcodec and
torchvision under qwen_vl_utils) and run the HF processor (transformers). This
script times both on the videos of an existing benchmark result file with the
CI request settings (2 fps, 128 frames, 401408 pixels), so the same command
can run in the server's venv and in a venv holding the previous torch,
torchvision and torchcodec. It imports only sglang_omni.preprocessing and
transformers, so the second venv does not need sglang.

Usage:
    python preprocess_ab.py --samples-json RESULTS.json --model-path MODEL \
        [--repeats 3] [--full] --out preprocess_ab_<label>.json

--full additionally times the real Qwen3OmniPreprocessor on the same payloads
(needs the full sglang_omni import chain, so the server's venv) to show that
the two steps account for the stage time seen in the server log.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROMPT = "Describe the video briefly."
VIDEO_FPS = 2.0
VIDEO_MAX_FRAMES = 128
VIDEO_MAX_PIXELS = 401408


def _versions() -> dict:
    out = {}
    for name in (
        "torch",
        "torchvision",
        "torchcodec",
        "transformers",
        "qwen-vl-utils",
        "av",
        "librosa",
        "numpy",
        "pillow",
    ):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    import torch

    out["torch_num_threads"] = torch.get_num_threads()
    out["cpu_count"] = os.cpu_count()
    out["TORCHCODEC_NUM_THREADS"] = os.environ.get("TORCHCODEC_NUM_THREADS")
    return out


def _resolve_model_dir(model_path: str) -> str:
    if Path(model_path).is_dir():
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_path,
        local_files_only=True,
        allow_patterns=["*.json", "*.txt", "*.py", "*.jinja", "*.model"],
    )


def _video_paths(samples_json: str) -> list[tuple[str, str]]:
    body = json.load(open(samples_json))
    seen = set()
    out = []
    for rec in body["per_sample"]:
        path = rec.get("video_path")
        if not path or path in seen:
            continue
        seen.add(path)
        out.append((rec.get("sample_id", Path(path).stem), path))
    return out


async def _two_step(processor, path: str) -> dict:
    from sglang_omni.preprocessing.video import ensure_video_list_async

    t0 = time.perf_counter()
    videos, sampled_fps, _ = await ensure_video_list_async(
        [path],
        fps=VIDEO_FPS,
        max_frames=VIDEO_MAX_FRAMES,
        min_pixels=None,
        max_pixels=VIDEO_MAX_PIXELS,
        total_pixels=None,
        extract_audio=False,
        audio_target_sr=16000,
    )
    t1 = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": PROMPT}],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    videos_kwargs = {
        "fps": sampled_fps[0] if sampled_fps else VIDEO_FPS,
        "max_frames": VIDEO_MAX_FRAMES,
        "max_pixels": VIDEO_MAX_PIXELS,
        "device": "cpu",
    }
    hf = processor(
        text=text,
        videos=videos,
        add_special_tokens=False,
        return_tensors="pt",
        videos_kwargs=videos_kwargs,
    )
    t2 = time.perf_counter()
    frames = videos[0]
    return {
        "load_s": t1 - t0,
        "processor_s": t2 - t1,
        "total_s": t2 - t0,
        "frames": list(frames.shape) if hasattr(frames, "shape") else None,
        "input_ids": int(hf["input_ids"].shape[-1]),
        "pixel_values_videos": (
            list(hf["pixel_values_videos"].shape)
            if "pixel_values_videos" in hf
            else None
        ),
    }


async def _full(preprocessor, sample_id: str, path: str) -> dict:
    from sglang_omni.proto.request import OmniRequest, StagePayload

    request = OmniRequest(
        inputs={
            "messages": [{"role": "user", "content": PROMPT}],
            "videos": [path],
            "video_fps": VIDEO_FPS,
            "video_max_frames": VIDEO_MAX_FRAMES,
            "video_max_pixels": VIDEO_MAX_PIXELS,
        },
        params={},
        metadata={},
    )
    payload = StagePayload(request_id=f"ab-{sample_id}", request=request, data=None)
    t0 = time.perf_counter()
    out = await preprocessor(payload)
    t1 = time.perf_counter()
    prompt = (out.data or {}).get("prompt") or {}
    ids = prompt.get("input_ids")
    return {
        "full_s": t1 - t0,
        "input_ids": int(ids.shape[-1]) if hasattr(ids, "shape") else None,
    }


def _median(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


async def _run(args) -> dict:
    from transformers import Qwen3OmniMoeProcessor

    model_dir = _resolve_model_dir(args.model_path)
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    preprocessor = None
    full_error = None
    if args.full:
        try:
            from sglang_omni.models.qwen3_omni.components.preprocessor import (
                Qwen3OmniPreprocessor,
            )

            preprocessor = Qwen3OmniPreprocessor(
                model_dir,
                max_seq_len=32768,
                video_fps=VIDEO_FPS,
                video_max_frames=VIDEO_MAX_FRAMES,
                video_max_pixels=VIDEO_MAX_PIXELS,
            )
        except Exception as exc:  # the old venv has no sglang
            full_error = f"{type(exc).__name__}: {exc}"
    samples = _video_paths(args.samples_json)
    if args.max_videos:
        samples = samples[: args.max_videos]
    per_video = []
    for sample_id, path in samples:
        await _two_step(processor, path)
        runs = [await _two_step(processor, path) for _ in range(args.repeats)]
        row = {
            "sample_id": sample_id,
            "video": path,
            "load_s": _median(runs, "load_s"),
            "processor_s": _median(runs, "processor_s"),
            "total_s": _median(runs, "total_s"),
            "frames": runs[0]["frames"],
            "input_ids": runs[0]["input_ids"],
            "pixel_values_videos": runs[0]["pixel_values_videos"],
        }
        if preprocessor is not None:
            await _full(preprocessor, sample_id, path)
            full_runs = [
                await _full(preprocessor, sample_id, path) for _ in range(args.repeats)
            ]
            row["full_s"] = _median(full_runs, "full_s")
            row["full_input_ids"] = full_runs[0]["input_ids"]
        per_video.append(row)
        print(
            f"{sample_id:8s} load={row['load_s']:.3f}s processor={row['processor_s']:.3f}s "
            f"total={row['total_s']:.3f}s"
            + (f" full={row['full_s']:.3f}s" if "full_s" in row else "")
            + f" frames={row['frames']} input_ids={row['input_ids']}",
            flush=True,
        )
    summary = {
        "videos": len(per_video),
        "repeats": args.repeats,
        "load_s_median": _median(per_video, "load_s"),
        "load_s_mean": statistics.mean(r["load_s"] for r in per_video),
        "processor_s_median": _median(per_video, "processor_s"),
        "processor_s_mean": statistics.mean(r["processor_s"] for r in per_video),
        "total_s_median": _median(per_video, "total_s"),
        "total_s_mean": statistics.mean(r["total_s"] for r in per_video),
        "full_s_mean": (
            statistics.mean(r["full_s"] for r in per_video)
            if per_video and "full_s" in per_video[0]
            else None
        ),
        "full_error": full_error,
    }
    print(json.dumps(summary, indent=1))
    return {"env": _versions(), "summary": summary, "per_video": per_video}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples-json", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--full", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = asyncio.run(_run(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
