"""Where the preprocessing stage's serial path spends its time per OpenMP setting: wall and process CPU per request, and per op.

usage: OMP_NUM_THREADS=<n> [OMP_WAIT_POLICY=PASSIVE] PYTHONPATH=<tree> taskset -c <cpus> python3 omp_serial_profile.py run --out cell.json
       python3 omp_serial_profile.py compare cell_a.json cell_b.json ...

run: Qwen3OmniPreprocessor.call_impl serially on the thread that calls it, as the stage's scheduler thread does for a request
without video, on SeedTTS EN voice-clone inputs (text plus reference audio) and the MMSU CI audio questions. The first pass warms
caches and is not reported. The measured passes record wall time and process CPU time (every thread of the process, so OpenMP
spin shows) per request. One profiled pass records torch CPU op self time by op and input shape. One fresh process per setting.
compare: per input kind, wall and CPU per request for each file, and the ops whose self time differs most from the first file.
"""

import argparse
import asyncio
import json
import os
import resource
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.preprocessing.resource_connector import MEDIA_WORKERS
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.utils.cpu import automatic_video_decode_threads

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS = "zhaochenyang20/seed-tts-eval-arrow"
MMSU = "zhaochenyang20/mmsu-ci-2000"


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def build_inputs(
    speech: int, audio: int
) -> dict[str, list[tuple[str, dict[str, object]]]]:
    kinds: dict[str, list[tuple[str, dict[str, object]]]] = {"speech": [], "audio": []}
    for sample in load_seedtts_samples(SEEDTTS, speech, split="en"):
        kinds["speech"].append(
            (
                f"speech:{sample.sample_id}",
                {
                    "messages": [{"role": "user", "content": sample.target_text}],
                    "audios": [sample.ref_audio],
                },
            )
        )
    for sample in load_mmsu_samples(audio, repo_id=MMSU):
        kinds["audio"].append(
            (
                f"audio:{sample.sample_id}",
                {
                    "messages": [{"role": "user", "content": sample.question}],
                    "audios": [sample.audio_path],
                },
            )
        )
    return kinds


def call(
    preprocessor: Qwen3OmniPreprocessor, request_id: str, inputs: dict[str, object]
) -> None:
    payload = StagePayload(
        request_id=request_id, request=OmniRequest(inputs=inputs, params={}), data={}
    )
    asyncio.run(preprocessor.call_impl(payload))


def run(args: argparse.Namespace) -> None:
    preprocessor = Qwen3OmniPreprocessor(
        MODEL,
        max_seq_len=32768,
        video_decode_threads=automatic_video_decode_threads(MEDIA_WORKERS),
    )
    kinds = build_inputs(args.speech, args.audio)
    record: dict[str, object] = {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "unset"),
        "torch_threads": torch.get_num_threads(),
        "affinity": len(os.sched_getaffinity(0)),
        "kinds": {},
    }
    for kind, requests in kinds.items():
        for request_id, inputs in requests:
            call(preprocessor, request_id, inputs)
        wall_ms: list[float] = []
        cpu_ms: list[float] = []
        for _ in range(args.passes):
            for request_id, inputs in requests:
                cpu_start = process_cpu_seconds()
                wall_start = time.perf_counter()
                call(preprocessor, request_id, inputs)
                wall_ms.append((time.perf_counter() - wall_start) * 1e3)
                cpu_ms.append((process_cpu_seconds() - cpu_start) * 1e3)
        profiled = requests[: args.profiled]
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as profiler:
            for request_id, inputs in profiled:
                call(preprocessor, request_id, inputs)
        ops = [
            {
                "op": event.key,
                "shapes": str(event.input_shapes)[:160],
                "calls": event.count / len(profiled),
                "self_us": event.self_cpu_time_total / len(profiled),
            }
            for event in profiler.key_averages(group_by_input_shape=True)
            if event.self_cpu_time_total > 0
        ]
        ops.sort(key=lambda op: -op["self_us"])
        record["kinds"][kind] = {
            "requests": len(requests),
            "wall_ms": wall_ms,
            "cpu_ms": cpu_ms,
            "ops_per_request": ops[:60],
        }
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")


def compare(args: argparse.Namespace) -> None:
    files = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.files]
    for kind in files[0]["kinds"]:
        print(f"== {kind}")
        for path, data in zip(args.files, files):
            cell = data["kinds"][kind]
            wall = sorted(cell["wall_ms"])
            cpu = cell["cpu_ms"]
            print(
                f"  {Path(path).stem:24s} omp {data['omp_num_threads']:>4} {data['omp_wait_policy']:8s} torch {data['torch_threads']:>3} "
                f"wall mean {statistics.fmean(wall):7.2f} p50 {statistics.median(wall):7.2f} p90 {wall[int(0.9 * len(wall))]:7.2f} "
                f"cpu mean {statistics.fmean(cpu):7.2f} ms a request"
            )
        reference = {
            (op["op"], op["shapes"]): op
            for op in files[0]["kinds"][kind]["ops_per_request"]
        }
        for path, data in zip(args.files[1:], files[1:]):
            mine = {
                (op["op"], op["shapes"]): op
                for op in data["kinds"][kind]["ops_per_request"]
            }
            deltas = sorted(
                (
                    (mine[key]["self_us"] - reference[key]["self_us"], key)
                    for key in mine
                    if key in reference
                ),
                key=lambda item: -abs(item[0]),
            )
            print(
                f"  ops against {Path(args.files[0]).stem}, {Path(path).stem}, us a request:"
            )
            for delta, key in deltas[:12]:
                print(
                    f"    {delta:+8.1f}  {key[0][:40]:40s} calls {mine[key]['calls']:5.1f} "
                    f"{reference[key]['self_us']:8.1f} -> {mine[key]['self_us']:8.1f}  {key[1][:70]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--speech", type=int, default=300)
    run_parser.add_argument("--audio", type=int, default=300)
    run_parser.add_argument("--passes", type=int, default=2)
    run_parser.add_argument("--profiled", type=int, default=40)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
