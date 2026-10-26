"""Video decode time per Video-MME CI video: the qwen-vl-utils torchcodec reader against the same frames decoded by
several single-thread decoders in parallel threads (frame indices split into chunks), and by the GPU decoder.

usage: PYTHONPATH=<tree> python3 video_decode_probe.py --out decode.json [--device cuda:1] [--decoders 4,8,16]

Per video the reader's own frame indices (calculate_video_frame_range, smart_nframes at the CI's fps 2 and 128
frames), then: the reader as served (one decoder, TORCHCODEC_NUM_THREADS ffmpeg threads); k decoders, each on
its chunk of the indices in its own thread; the decoder on --device. Each form's frames are compared with the
reader's (equal, or the max and mean absolute difference). Prints per form p50 / mean / p95 ms and the frame check.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from qwen_vl_utils import vision_process as qwen_vision
from torchcodec.decoders import VideoDecoder

from benchmarks.dataset.videomme import load_videomme_samples

VIDEOMME_CI = "zhaochenyang20/Video_MME_ci"


def reader_indices(path: str) -> list[int]:
    ele = {"video": path, "fps": 2.0, "max_frames": 128}
    decoder = VideoDecoder(path)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = qwen_vision.calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = qwen_vision.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    return torch.linspace(start_frame, end_frame, nframes).round().long().tolist()


def decode_chunks(path: str, indices: list[int], decoders: int) -> torch.Tensor:
    chunks = [chunk.tolist() for chunk in np.array_split(indices, decoders) if len(chunk) > 0]

    def decode(chunk: list[int]) -> torch.Tensor:
        return VideoDecoder(path, num_ffmpeg_threads=1).get_frames_at(indices=chunk).data

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        return torch.cat(list(pool.map(decode, chunks)))


def check(reference: torch.Tensor, frames: torch.Tensor) -> object:
    frames = frames.cpu()
    if reference.shape != frames.shape:
        return f"shape {tuple(reference.shape)} against {tuple(frames.shape)}"
    elif torch.equal(reference, frames):
        return "equal"
    else:
        gap = (reference.int() - frames.int()).abs().float()
        return [gap.max().item(), gap.mean().item()]


def timed(function) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    result = function()
    if result.is_cuda:
        torch.cuda.synchronize(result.device)
    else:
        pass
    return result, (time.perf_counter() - start) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--decoders", default="4,8,16")
    parser.add_argument("--videos", type=int, default=None)
    args = parser.parse_args()
    counts = [int(item) for item in args.decoders.split(",")]
    records = []
    for sample in load_videomme_samples(args.videos, repo_id=VIDEOMME_CI):
        path = sample.video_path
        indices = reader_indices(path)
        reference, reader_ms = timed(lambda: qwen_vision.VIDEO_READER_BACKENDS["torchcodec"]({"video": path, "fps": 2.0, "max_frames": 128})[0])
        record = {"id": sample.sample_id, "frames": len(indices), "shape": list(reference.shape), "reader": reader_ms, "checks": {}}
        for count in counts:
            frames, elapsed = timed(lambda: decode_chunks(path, indices, count))
            record[f"{count} decoders"] = elapsed
            record["checks"][f"{count} decoders"] = check(reference, frames)
        if args.device is not None:
            try:
                frames, elapsed = timed(lambda: VideoDecoder(path, device=args.device).get_frames_at(indices=indices).data)
                record["device"] = elapsed
                record["checks"]["device"] = check(reference, frames)
            except Exception as error:
                record["checks"]["device"] = f"failed: {type(error).__name__}: {error}"
        else:
            pass
        records.append(record)
    kept = records[1:]
    forms = ["reader"] + [f"{count} decoders" for count in counts] + (["device"] if args.device else [])
    for form in forms:
        values = sorted(record[form] for record in kept if form in record)
        if values:
            print(f"{form:12s} p50 {statistics.median(values):8.1f} mean {statistics.mean(values):8.1f} "
                  f"p95 {values[int(0.95 * len(values))]:8.1f} ms over {len(values)} videos")
        else:
            print(f"{form:12s} no values")
        verdicts = {json.dumps(record["checks"].get(form)) for record in records if form in record["checks"]}
        if verdicts:
            print(f"  frames against the reader: {sorted(verdicts)[:5]}")
        else:
            pass
    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(records, out)


if __name__ == "__main__":
    main()
