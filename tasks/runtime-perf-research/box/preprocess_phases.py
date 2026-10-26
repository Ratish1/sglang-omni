"""Where a Qwen3-Omni preprocessing call spends its time, and what its transforms cost on a GPU.

usage: OMP_NUM_THREADS=32 PYTHONPATH=<tree> python3 preprocess_phases.py --out t32.json [--device cuda:1]

Serially, on the 1,088 SeedTTS EN reference audios (the voice clone input) and the 50 Video-MME CI videos at the
CI's video settings, Qwen3OmniPreprocessor.call_impl with each phase timed: video decode (the qwen-vl-utils
reader), the frame resize after it (load_video_path minus decode), the HF processor and inside it the video
processor, the feature extractor (Whisper log-mel) and the tokenizer. With --device, every HF processor call is
replayed with the processor's device kwarg set to that device (the on-device preprocessing form), and every decoded
video's resize is replayed there, each timed with a device sync and each output tensor compared with the CPU one
(max and mean absolute difference). The first --skip calls of each kind are left out of the summary.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import defaultdict

import torch
from qwen_vl_utils import vision_process as qwen_vision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_f

from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.preprocessing import video as video_module
from sglang_omni.proto.request import OmniRequest, StagePayload

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS = "zhaochenyang20/seed-tts-eval-arrow"
VIDEOMME_CI = "zhaochenyang20/Video_MME_ci"
VIDEO_SETTINGS = {"video_fps": 2.0, "video_max_frames": 128, "video_max_pixels": 401408}

phases: dict[str, float] = defaultdict(float)
differences: dict[str, object] = {}
replay = {"device": None, "active": False, "frames": None}


def difference(cpu: torch.Tensor, other: torch.Tensor) -> object:
    other = other.detach().cpu()
    if cpu.shape != other.shape:
        return f"shape {tuple(cpu.shape)} against {tuple(other.shape)}"
    elif cpu.is_floating_point():
        gap = (cpu.float() - other.float()).abs()
        return [gap.max().item(), gap.mean().item()]
    else:
        return int((cpu != other).sum().item())


def sync() -> None:
    if replay["device"] is not None:
        torch.cuda.synchronize(replay["device"])
    else:
        pass


def time_calls(cls: type, name: str) -> None:
    original = cls.__call__

    def timed(self, *args, **kwargs):
        start = time.perf_counter()
        result = original(self, *args, **kwargs)
        if replay["active"]:
            sync()
            phases[f"device {name}"] += (time.perf_counter() - start) * 1e3
        else:
            phases[name] += (time.perf_counter() - start) * 1e3
        return result

    cls.__call__ = timed


def time_processor(cls: type) -> None:
    original = cls.__call__

    def timed(self, *args, **kwargs):
        start = time.perf_counter()
        result = original(self, *args, **kwargs)
        phases["hf processor"] += (time.perf_counter() - start) * 1e3
        if replay["device"] is not None:
            replay_kwargs = dict(kwargs)
            if "videos_kwargs" in replay_kwargs:
                # note (ratish): a device in both the modality dict and the flat kwargs is rejected by the merge
                replay_kwargs["videos_kwargs"] = {
                    key: value for key, value in replay_kwargs["videos_kwargs"].items() if key != "device"
                }
            else:
                pass
            replay_kwargs["device"] = replay["device"]
            sync()
            replay["active"] = True
            start = time.perf_counter()
            replayed = original(self, *args, **replay_kwargs)
            sync()
            replay["active"] = False
            phases["device hf processor"] += (time.perf_counter() - start) * 1e3
            for key, value in result.items():
                other = replayed.get(key)
                if isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor):
                    differences[key] = difference(value, other)
                else:
                    pass
        else:
            pass
        return result

    cls.__call__ = timed


def time_readers() -> None:
    for backend, reader in list(qwen_vision.VIDEO_READER_BACKENDS.items()):

        def timed(ele, reader=reader):
            start = time.perf_counter()
            frames, fps = reader(ele)
            phases["video decode"] += (time.perf_counter() - start) * 1e3
            replay["frames"] = frames
            return frames, fps

        qwen_vision.VIDEO_READER_BACKENDS[backend] = timed


def time_video_load() -> None:
    original = video_module.load_video_path

    def timed(*args, **kwargs):
        start = time.perf_counter()
        frames, fps = original(*args, **kwargs)
        phases["video load"] += (time.perf_counter() - start) * 1e3
        if replay["device"] is not None:
            sync()
            start = time.perf_counter()
            resized = tv_f.resize(
                replay["frames"].to(replay["device"]),
                list(frames.shape[-2:]),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            sync()
            phases["device resize"] += (time.perf_counter() - start) * 1e3
            differences["resized frames"] = difference(frames, resized)
        else:
            pass
        return frames, fps

    video_module.load_video_path = timed


def run_one(preprocessor: Qwen3OmniPreprocessor, request_id: str, inputs: dict[str, object]) -> dict[str, object]:
    phases.clear()
    differences.clear()
    payload = StagePayload(request_id=request_id, request=OmniRequest(inputs=inputs, params={}), data={})
    start = time.perf_counter()
    asyncio.run(preprocessor.call_impl(payload))
    total = (time.perf_counter() - start) * 1e3
    record = {"id": request_id, "call": total, **phases, "differences": dict(differences)}
    if "video load" in phases:
        record["video resize"] = phases["video load"] - phases.get("video decode", 0.0)
    else:
        pass
    return record


def summarize(records: list[dict[str, object]], skip: int) -> None:
    kept = records[skip:]
    names = sorted({key for record in kept for key in record if key not in ("id", "differences")})
    for name in names:
        values = sorted(record.get(name, 0.0) for record in kept)
        print(f"  {name:24s} p50 {statistics.median(values):8.2f} mean {statistics.mean(values):8.2f} "
              f"p95 {values[int(0.95 * len(values))]:8.2f} ms")
    worst: dict[str, object] = {}
    for record in kept:
        for key, value in record["differences"].items():
            if isinstance(value, list):
                previous = worst.get(key, [0.0, 0.0])
                worst[key] = [max(previous[0], value[0]), max(previous[1], value[1])]
            elif isinstance(value, int):
                worst[key] = max(worst.get(key, 0), value)
            else:
                worst[key] = value
    for key, value in worst.items():
        if isinstance(value, list):
            print(f"  device against CPU {key}: max abs {value[0]:.3g}, largest mean abs {value[1]:.3g}")
        else:
            print(f"  device against CPU {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--speech", type=int, default=None)
    parser.add_argument("--videos", type=int, default=None)
    parser.add_argument("--skip", type=int, default=3)
    args = parser.parse_args()
    replay["device"] = args.device
    preprocessor = Qwen3OmniPreprocessor(MODEL, max_seq_len=32768)
    processor = preprocessor.processor
    time_processor(type(processor))
    time_calls(type(processor.video_processor), "hf video processor")
    time_calls(type(processor.feature_extractor), "hf feature extractor")
    time_calls(type(processor.tokenizer), "hf tokenizer")
    time_readers()
    time_video_load()
    records: dict[str, object] = {"threads": torch.get_num_threads(), "device": args.device, "speech": [], "video": []}
    for sample in load_seedtts_samples(SEEDTTS, args.speech, split="en"):
        inputs = {"messages": [{"role": "user", "content": sample.target_text}], "audios": [sample.ref_audio]}
        records["speech"].append(run_one(preprocessor, f"speech:{sample.sample_id}", inputs))
    for sample in load_videomme_samples(args.videos, repo_id=VIDEOMME_CI):
        inputs = {
            "messages": [{"role": "user", "content": sample.prompt or sample.question}],
            "videos": [sample.video_path],
            **VIDEO_SETTINGS,
        }
        records["video"].append(run_one(preprocessor, f"video:{sample.sample_id}", inputs))
    records["process threads"] = len(os.listdir("/proc/self/task"))
    print(f"intra-op threads {records['threads']}, process threads at the end {records['process threads']}, "
          f"device {args.device}")
    for kind in ("speech", "video"):
        print(f"{kind}: {len(records[kind])} calls")
        summarize(records[kind], args.skip)
    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(records, out)


if __name__ == "__main__":
    main()
