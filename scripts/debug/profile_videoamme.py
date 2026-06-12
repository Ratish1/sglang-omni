# SPDX-License-Identifier: Apache-2.0
"""Profile Qwen3-Omni Video-AMME traffic against a live server.

This is an investigation harness for issue #765-style failures. It brackets
the benchmark with the built-in profiler endpoints, polls GPU memory/process
state, and writes a self-contained artifact directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.request(method, url, json=payload, timeout=30)
    response.raise_for_status()
    if response.content:
        return response.json()
    return None


def _safe_url_name(base_url: str) -> str:
    parsed = urlparse(base_url)
    label = parsed.netloc or parsed.path or base_url
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


def _run_command(args: list[str], output_path: Path) -> None:
    try:
        proc = subprocess.run(
            args,
            check=False,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_path.write_text(proc.stdout, encoding="utf-8")
    except Exception as exc:
        output_path.write_text(f"failed to run {args!r}: {exc}\n", encoding="utf-8")


class NvidiaSmiPoller:
    def __init__(self, out_dir: Path, interval_s: float) -> None:
        self.out_dir = out_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 2))

    def _run(self) -> None:
        gpu_log = self.out_dir / "gpu_memory_util.csv"
        apps_log = self.out_dir / "compute_apps.csv"
        gpu_query = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,uuid,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
        ]
        apps_query = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        with (
            gpu_log.open("w", encoding="utf-8") as gpu_fp,
            apps_log.open("w", encoding="utf-8") as apps_fp,
        ):
            gpu_fp.write(
                "epoch_ns,timestamp,index,name,uuid,memory_used_mb,memory_free_mb,"
                "gpu_util_pct,mem_util_pct,power_w\n"
            )
            apps_fp.write("epoch_ns,pid,process_name,gpu_uuid,used_memory_mb\n")
            while not self._stop.is_set():
                epoch_ns = time.time_ns()
                self._append_query(gpu_query, gpu_fp, epoch_ns)
                self._append_query(apps_query, apps_fp, epoch_ns)
                self._stop.wait(self.interval_s)

    @staticmethod
    def _append_query(args: list[str], fp, epoch_ns: int) -> None:
        try:
            proc = subprocess.run(
                args,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            fp.write(f"{epoch_ns},nvidia-smi-not-found\n")
            return
        if proc.returncode != 0:
            fp.write(f"{epoch_ns},query-failed,{proc.stderr.strip()!r}\n")
            return
        for line in proc.stdout.splitlines():
            if line.strip():
                fp.write(f"{epoch_ns},{line}\n")


def _scenario_defaults(args: argparse.Namespace) -> None:
    if args.max_samples is None:
        if args.scenario == "stage9-text":
            args.max_samples = 50
        elif args.scenario == "stage11-tp2":
            args.max_samples = 10
        else:
            args.max_samples = 20
    if args.max_concurrency is None:
        args.max_concurrency = 16
    if args.audio_output is None:
        args.audio_output = args.scenario != "stage9-text"


def _discover_profile_urls(args: argparse.Namespace) -> list[str]:
    if args.profile_base_url:
        return args.profile_base_url
    if args.discover_router_workers:
        payload = _json_request("GET", f"{args.traffic_base_url.rstrip('/')}/workers")
        workers = payload.get("workers") or []
        urls = [worker["url"] for worker in workers if worker.get("url")]
        if urls:
            return urls
    return [args.traffic_base_url]


def _write_system_artifacts(out_dir: Path, args: argparse.Namespace) -> None:
    system_dir = out_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    env_subset = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("CUDA", "NCCL", "PYTORCH", "SGLANG", "HF_", "HTTP"))
    }
    (system_dir / "env.json").write_text(
        json.dumps(env_subset, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _run_command(["git", "rev-parse", "HEAD"], system_dir / "git_head.txt")
    _run_command(
        ["git", "status", "--short", "--branch"],
        system_dir / "git_status.txt",
    )
    _run_command(["nvidia-smi", "-L"], system_dir / "nvidia_smi_l.txt")
    _run_command(["nvidia-smi"], system_dir / "nvidia_smi.txt")
    _run_command(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print('torch', torch.__version__); "
                "print('cuda', torch.version.cuda); "
                "print('cuda_available', torch.cuda.is_available()); "
                "print('device_count', torch.cuda.device_count())"
            ),
        ],
        system_dir / "torch_cuda.txt",
    )


def _render_profiler_report(events_dir: Path, out_dir: Path) -> None:
    from sglang_omni.profiler.views import build_report, format_table

    report = build_report(events_dir)
    (out_dir / "profiler_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    table = (
        f"# Requests: {report['request_count']}\n\n"
        "## Stage breakdown\n"
        + format_table(
            report["stage_breakdown"],
            ["stage", "interval", "count", "total_ms", "avg_ms", "p95_ms", "max_ms"],
        )
        + "\n## Hop breakdown\n"
        + format_table(
            report["hop_breakdown"],
            ["src", "dst", "kind", "count", "total_ms", "avg_ms", "p95_ms", "max_ms"],
        )
    )
    (out_dir / "profiler_report.txt").write_text(table, encoding="utf-8")


async def _run_videoamme(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    from benchmarks.dataset.prepare import DATASETS
    from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig

    results_dir = out_dir / "benchmark"
    config = VideoEvalConfig(
        model=args.model,
        base_url=args.traffic_base_url,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        output_dir=str(results_dir),
        repo_id=args.repo_id or DATASETS["videoamme-ci-50"],
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
        video_max_pixels=args.video_max_pixels,
        enable_audio=args.audio_output,
        disable_tqdm=args.disable_tqdm,
        timeout_s=args.timeout_s,
    )
    return await run_videoamme_eval(config, compute_wer=False)


def _maybe_router_snapshot(base_url: str, path: Path) -> None:
    try:
        payload = _json_request("GET", f"{base_url.rstrip('/')}/workers")
    except Exception as exc:
        payload = {"error": repr(exc)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _start_profiles(
    profile_urls: list[str],
    *,
    run_id: str,
    event_dir: Path,
    trace_dir: Path,
    enable_torch: bool,
) -> list[dict[str, Any]]:
    started = []
    for base_url in profile_urls:
        name = _safe_url_name(base_url)
        payload = {
            "run_id": run_id,
            "event_dir": str(event_dir),
            "enable_torch": enable_torch,
            "trace_path_template": str(trace_dir / name / "{stage}"),
        }
        response = _json_request(
            "POST",
            f"{base_url.rstrip('/')}/start_profile",
            payload,
        )
        started.append({"base_url": base_url, "response": response})
    return started


def _stop_profiles(profile_urls: list[str], run_id: str) -> list[dict[str, Any]]:
    stopped = []
    for base_url in profile_urls:
        try:
            response = _json_request(
                "POST",
                f"{base_url.rstrip('/')}/stop_profile",
                {"run_id": run_id},
            )
            stopped.append({"base_url": base_url, "response": response})
        except Exception as exc:
            stopped.append({"base_url": base_url, "error": repr(exc)})
    return stopped


def _write_summary(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    profile_urls: list[str],
    started: list[dict[str, Any]],
    stopped: list[dict[str, Any]],
    results: dict[str, Any],
    elapsed_s: float,
) -> None:
    summary = results.get("summary", {})
    speed = results.get("speed", {})
    text = f"""# Video-AMME Profile Run

run_id: `{args.run_id}`
scenario: `{args.scenario}`
traffic_base_url: `{args.traffic_base_url}`
profile_urls: `{", ".join(profile_urls)}`
audio_output: `{args.audio_output}`
max_samples: `{args.max_samples}`
max_concurrency: `{args.max_concurrency}`
elapsed_s: `{elapsed_s:.3f}`

## Benchmark Summary

```json
{json.dumps(summary, indent=2)}
```

## Speed Summary

```json
{json.dumps(speed, indent=2)}
```

## Profiler Control

started:
```json
{json.dumps(started, indent=2)}
```

stopped:
```json
{json.dumps(stopped, indent=2)}
```

## Key Artifacts

- `events/`: request-level JSONL events from coordinator and stages
- `traces/`: torch profiler Chrome traces, one set per profiled server process
- `profiler_report.txt`: aggregated stage and hop timing table
- `profiler_report.json`: full timelines, stage breakdown, and hop breakdown
- `nvidia_smi/gpu_memory_util.csv`: GPU memory/utilization samples
- `nvidia_smi/compute_apps.csv`: per-process GPU memory samples
- `benchmark/videoamme_results.json`: raw benchmark output
- `router_workers_before.json` / `router_workers_after.json`: router state if available
- `system/`: git, environment, torch/CUDA, and nvidia-smi metadata
"""
    (out_dir / "SUMMARY.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("stage9-text", "stage10-router", "stage11-tp2", "custom"),
        default="custom",
    )
    parser.add_argument("--traffic-base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--profile-base-url",
        action="append",
        help=(
            "Server URL to profile. Repeat for router workers. If omitted, "
            "the traffic URL is profiled unless --discover-router-workers is set."
        ),
    )
    parser.add_argument("--discover-router-workers", action="store_true")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--video-fps", type=float, default=2)
    parser.add_argument("--video-max-frames", type=int, default=128)
    parser.add_argument("--video-max-pixels", type=int, default=401408)
    parser.add_argument("--timeout-s", type=int, default=500)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--audio-output", dest="audio_output", action="store_true")
    parser.add_argument("--no-audio-output", dest="audio_output", action="store_false")
    parser.set_defaults(audio_output=None)
    parser.add_argument("--request-events-only", action="store_true")
    parser.add_argument("--nvidia-smi-interval-s", type=float, default=1.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to /tmp/sglang_omni_issue765_profiles/<run_id>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _scenario_defaults(args)
    if args.run_id is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"videoamme_{args.scenario}_{stamp}"
    out_dir = Path(args.out_dir or f"/tmp/sglang_omni_issue765_profiles/{args.run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    event_dir = out_dir / "events"
    trace_dir = out_dir / "traces"
    profile_urls = _discover_profile_urls(args)

    _write_system_artifacts(out_dir, args)
    _maybe_router_snapshot(
        args.traffic_base_url, out_dir / "router_workers_before.json"
    )

    poller = NvidiaSmiPoller(out_dir / "nvidia_smi", args.nvidia_smi_interval_s)
    poller.start()
    started: list[dict[str, Any]] = []
    stopped: list[dict[str, Any]] = []
    started_at = time.monotonic()
    try:
        started = _start_profiles(
            profile_urls,
            run_id=args.run_id,
            event_dir=event_dir,
            trace_dir=trace_dir,
            enable_torch=not args.request_events_only,
        )
        results = asyncio.run(_run_videoamme(args, out_dir))
    finally:
        stopped = _stop_profiles(profile_urls, args.run_id)
        poller.stop()
    elapsed_s = time.monotonic() - started_at

    _maybe_router_snapshot(args.traffic_base_url, out_dir / "router_workers_after.json")
    (out_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    _render_profiler_report(event_dir, out_dir)
    _write_summary(
        out_dir,
        args=args,
        profile_urls=profile_urls,
        started=started,
        stopped=stopped,
        results=results,
        elapsed_s=elapsed_s,
    )
    print(f"profile artifacts: {out_dir}")
    print(f"summary: {out_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
