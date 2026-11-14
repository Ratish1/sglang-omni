"""Synthetic Qwen3-TTS profile: one server boot under nsys, one report per regime cell.

Regimes, as SGLang's one-batch tools and InferenceMAX shape them:
- prefill: each request clones a 20.48 s reference (256 codec frames, the largest captured
  reference encoder bucket) in ICL mode and stops after one codec frame;
- decode: each request clones a 3 s reference by x-vector, speaks a one-token text and
  decodes 2048 frames with the codec EOS masked.
References are noise, a new one per request, so no reference cache hits.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 24000
CODEC_FRAMES_PER_S = 12.5
PCM_BYTES_PER_FRAME = int(SAMPLE_RATE / CODEC_FRAMES_PER_S) * 2
PREFILL_REFERENCE_FRAMES = 256
DECODE_REFERENCE_S = 3.0
DECODE_FRAMES = 2048
PREFILL_ROUNDS = 3
DECODE_WINDOW_START_FRAMES = 256
DECODE_WINDOW_S = 3.0
BATCH_SIZES = (1, 16, 64)
REFERENCE_TEXT = " ".join(["the"] * 40)
PREFILL_TEXT = " ".join(["the"] * 20)
DECODE_TEXT = "the"
LAUNCH_OPTIONS = {
    "light": [
        "--trace=cuda,nvtx",
        "--cuda-graph-trace=graph",
        f"--python-functions-trace={HERE / 'marks.json'}",
    ],
    "timing": [
        "--trace=cuda,nvtx,osrt",
        "--cuda-graph-trace=graph",
        "--cuda-event-trace=true",
        f"--python-functions-trace={HERE / 'marks.json'}",
    ],
    "attribution": [
        "--trace=cuda,nvtx,osrt,python-gil",
        "--cuda-graph-trace=node:nvtx-precapture",
        "--pytorch=autograd-nvtx",
        "--cudabacktrace=sync:1000,memory:1000",
        "--python-backtrace=cuda",
        "--python-sampling=true",
    ],
}
NCU_SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "ComputeWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
)
START_OPTIONS = {
    "light": ["--sample=none", "--cpuctxsw=none"],
    "timing": [
        "--sample=none",
        "--cpuctxsw=process-tree",
        "--gpu-metrics-devices=0",
        "--gpu-metrics-set=gh100",
        "--gpu-metrics-frequency=10000",
    ],
    "attribution": ["--sample=process-tree", "--cpuctxsw=process-tree"],
}


def write_reference(path: Path, seconds: float, seed: int) -> None:
    samples = np.random.default_rng(seed).normal(0.0, 0.1, int(seconds * SAMPLE_RATE))
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


class References:
    """A new noise reference per request, so every request misses the reference caches."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.count = 0
        directory.mkdir(parents=True, exist_ok=True)

    def take(self, seconds: float) -> str:
        self.count += 1
        path = self.directory / f"reference_{self.count:05d}_{seconds:g}s.wav"
        write_reference(path, seconds, self.count)
        return str(path)


def prefill_payload(references: References) -> dict:
    return {
        "model": MODEL,
        "input": PREFILL_TEXT,
        "ref_audio": references.take(PREFILL_REFERENCE_FRAMES / CODEC_FRAMES_PER_S),
        "ref_text": REFERENCE_TEXT,
        "response_format": "pcm",
        "stream": True,
        "max_new_tokens": 1,
    }


def decode_payload(references: References, frames: int) -> dict:
    return {
        "model": MODEL,
        "input": DECODE_TEXT,
        "ref_audio": references.take(DECODE_REFERENCE_S),
        "response_format": "pcm",
        "stream": True,
        "max_new_tokens": frames,
    }


class StreamRecord:
    def __init__(self) -> None:
        self.bytes = 0
        self.first_audio_s: float | None = None
        self.done_s: float | None = None
        self.error: str | None = None


def stream_request(port: int, payload: dict, record: StreamRecord, t0: float) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                else:
                    pass
                if record.first_audio_s is None:
                    record.first_audio_s = time.monotonic() - t0
                else:
                    pass
                record.bytes += len(chunk)
    except Exception as exc:
        record.error = repr(exc)
    record.done_s = time.monotonic() - t0


def start_batch(port: int, payloads: list[dict]) -> tuple[list[StreamRecord], list[threading.Thread]]:
    t0 = time.monotonic()
    records = [StreamRecord() for _ in payloads]
    threads = [
        threading.Thread(target=stream_request, args=(port, payload, record, t0))
        for payload, record in zip(payloads, records)
    ]
    for thread in threads:
        thread.start()
    return records, threads


def finish_batch(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join()


def summarize(records: list[StreamRecord]) -> dict:
    return {
        "requests": len(records),
        "errors": [record.error for record in records if record.error],
        "frames": [record.bytes // PCM_BYTES_PER_FRAME for record in records],
        "first_audio_ms": [
            None if record.first_audio_s is None else round(record.first_audio_s * 1000, 2)
            for record in records
        ],
        "done_ms": [round(record.done_s * 1000, 2) for record in records],
    }


class NsysSession:
    def __init__(self, name: str, mode: str, directory: Path) -> None:
        self.name = name
        self.mode = mode
        self.directory = directory

    def start(self, cell: str) -> None:
        subprocess.run(
            [
                "nsys",
                "start",
                f"--session={self.name}",
                f"--output={self.directory / cell}",
                "--force-overwrite=true",
                *START_OPTIONS[self.mode],
            ],
            check=True,
        )

    def stop(self) -> None:
        subprocess.run(["nsys", "stop", f"--session={self.name}"], check=True)


def launch_server(
    tree: Path,
    mode: str,
    session: str,
    port: int,
    log: Path,
    profile_step: str | None,
    ncu_filter: list[str],
    server_args: list[str],
) -> subprocess.Popen:
    environment = {
        **os.environ,
        "PYTHONPATH": f"{HERE / 'probe'}:{tree}",
        "QWEN3_TTS_SYNTHETIC": "1",
        "SGLANG_OMNI_STRICT_PORT": "1",
    }
    if mode == "none":
        profiler = []
    elif mode == "ncu":
        assert profile_step is not None, "ncu mode profiles one step named by --profile-step"
        environment["QWEN3_TTS_PROFILE_STEP"] = profile_step
        profiler = [
            "ncu",
            "--profile-from-start",
            "off",
            "--target-processes",
            "all",
            "--graph-profiling",
            "node",
            *[f"--section={section}" for section in NCU_SECTIONS],
            *ncu_filter,
            f"--export={log.parent / ('ncu_' + profile_step.replace(':', '_'))}",
            "--force-overwrite",
        ]
    elif mode == "attribution":
        # note (ratish): nsys takes one function list, and graph nodes take the names of the ranges open when they were captured.
        merged = log.parent / "marks_attribution.json"
        merged.write_text(
            json.dumps(
                json.loads((HERE / "marks.json").read_text())
                + json.loads((HERE / "marks_capture.json").read_text())
            )
        )
        profiler = [
            "nsys",
            "launch",
            f"--session-new={session}",
            *LAUNCH_OPTIONS[mode],
            f"--python-functions-trace={merged}",
        ]
    else:
        profiler = ["nsys", "launch", f"--session-new={session}", *LAUNCH_OPTIONS[mode]]
    command = [
        *profiler,
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tts_engine.gpu",
        "0",
        "--vocoder.gpu",
        "0",
        *server_args,
    ]
    with log.open("w") as handle:
        server = subprocess.Popen(
            command,
            cwd=tree,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (log.parent / "server.pgid").write_text(str(server.pid))
    return server


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        else:
            pass
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError("server did not become healthy")


def run_prefill_cell(port: int, references: References, batch_size: int, session: NsysSession | None, cell: str) -> dict:
    rounds = []
    if session is not None:
        session.start(cell)
    else:
        pass
    for _ in range(PREFILL_ROUNDS):
        records, threads = start_batch(
            port, [prefill_payload(references) for _ in range(batch_size)]
        )
        finish_batch(threads)
        rounds.append(summarize(records))
        time.sleep(0.5)
    if session is not None:
        session.stop()
    else:
        pass
    return {"cell": cell, "rounds": rounds}


def run_decode_cell(port: int, references: References, batch_size: int, frames: int, session: NsysSession | None, cell: str) -> dict:
    records, threads = start_batch(
        port, [decode_payload(references, frames) for _ in range(batch_size)]
    )
    window = None
    if session is not None:
        threshold = DECODE_WINDOW_START_FRAMES * PCM_BYTES_PER_FRAME
        while min(record.bytes for record in records) < threshold:
            if all(record.done_s is not None for record in records):
                raise RuntimeError(f"{cell} finished before the window opened")
            else:
                pass
            time.sleep(0.01)
        start_frames = [record.bytes // PCM_BYTES_PER_FRAME for record in records]
        session.start(cell)
        time.sleep(DECODE_WINDOW_S)
        stop_frames = [record.bytes // PCM_BYTES_PER_FRAME for record in records]
        session.stop()
        window = {"start_frames": start_frames, "stop_frames": stop_frames}
    else:
        pass
    finish_batch(threads)
    return {"cell": cell, "window": window, "requests": summarize(records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=[*sorted(LAUNCH_OPTIONS), "none", "ncu"], required=True)
    parser.add_argument("--profile-step", help="MODE:BATCH:INDEX of the talker step Nsight Compute profiles")
    parser.add_argument("--ncu-kernel-regex", help="profile only kernels whose name matches")
    parser.add_argument("--ncu-launch-count", type=int, help="profile at most this many matching kernels")
    parser.add_argument("--server-arg", action="append", default=[], help="extra serve argument, repeatable")
    parser.add_argument("--port", type=int, default=24100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    parser.add_argument("--regimes", nargs="+", choices=["prefill", "decode"], default=["prefill", "decode"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    references = References(args.out / "references")
    session_name = f"qwen3tts-{args.mode}"
    ncu_filter = []
    if args.ncu_kernel_regex is not None:
        ncu_filter.append(f"--kernel-name=regex:{args.ncu_kernel_regex}")
    else:
        pass
    if args.ncu_launch_count is not None:
        ncu_filter.append(f"--launch-count={args.ncu_launch_count}")
    else:
        pass
    server = launch_server(
        args.tree,
        args.mode,
        session_name,
        args.port,
        args.out / "server.log",
        args.profile_step,
        ncu_filter,
        args.server_arg,
    )
    results = []
    try:
        wait_healthy(args.port, server)
        session = NsysSession(session_name, args.mode, args.out)
        for batch_size in args.batch_sizes:
            run_prefill_cell(args.port, references, batch_size, None, f"warm_prefill_bs{batch_size}")
            run_decode_cell(args.port, references, batch_size, 64, None, f"warm_decode_bs{batch_size}")
        sessions = [None] if args.mode in ("none", "ncu") else [None, session]
        for batch_size in args.batch_sizes:
            for cell_session in sessions:
                prefix = "plain_" if cell_session is None else ""
                if "prefill" in args.regimes:
                    results.append(
                        run_prefill_cell(args.port, references, batch_size, cell_session, f"{prefix}prefill_bs{batch_size}")
                    )
                    print(json.dumps(results[-1]["cell"]), flush=True)
                else:
                    pass
                if "decode" in args.regimes:
                    results.append(
                        run_decode_cell(args.port, references, batch_size, DECODE_FRAMES, cell_session, f"{prefix}decode_bs{batch_size}")
                    )
                    print(json.dumps(results[-1]["cell"]), flush=True)
                else:
                    pass
    finally:
        (args.out / "cells.json").write_text(json.dumps(results, indent=1))
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
        else:
            pass


if __name__ == "__main__":
    main()
