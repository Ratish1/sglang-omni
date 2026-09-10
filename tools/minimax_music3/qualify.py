"""Run MiniMax cookbook, serving A/B and untimed latent qualification remotely."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command_output(args, *, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def repository_identity(path):
    return {
        "path": str(path),
        "commit": command_output(["git", "rev-parse", "HEAD"], cwd=path),
        "status": command_output(["git", "status", "--porcelain"], cwd=path),
        "diff": command_output(["git", "diff", "HEAD"], cwd=path),
    }


def stop_process(process):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def checkpoint_view(source, destination):
    destination.mkdir()
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.suffix == ".json":
            shutil.copy2(path, target)
        else:
            target.symlink_to(path.resolve())


def checkpoint_identity(source):
    files = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        entry = {
            "resolved": str(path.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if path.suffix == ".json":
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        files[str(path.relative_to(source))] = entry
    return files


@contextlib.contextmanager
def server(args, repo, output, devices, *, capture=False):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo), env.get("PYTHONPATH")])
    )
    for name in (
        "MINIMAX_MUSIC3_HIDDEN_DUMP",
        "MINIMAX_MUSIC3_FORCED_CODES",
        "MINIMAX_MUSIC3_QUALIFICATION_CAPTURE",
        "SGLANG_TEST_RETRACT",
    ):
        env.pop(name, None)
    entry = [sys.executable, "-m", "sglang_omni.cli"]
    if capture:
        capture_dir = output / "tensors"
        env["MINIMAX_MUSIC3_HIDDEN_DUMP"] = str(capture_dir)
        env["MINIMAX_MUSIC3_QUALIFICATION_CAPTURE"] = str(capture_dir)
        entry = [sys.executable, str(HERE / "capture_server.py")]
    checkpoint = output / "checkpoint"
    checkpoint_view(args.model_path, checkpoint)
    cmd = entry + [
        "serve",
        "--model-path",
        str(checkpoint),
        "--port",
        str(args.port),
        "--minimax_music3_ar.engine.max_running_requests",
        str(args.max_running_requests),
        "--dit_dav.factory.dtype",
        args.dtype,
    ]
    write_json(
        output / "launch.json",
        {
            "argv": cmd,
            "repo": repository_identity(repo),
            "devices": devices,
            "capture": capture,
        },
    )
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            cmd,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited with {process.returncode}; see {output / 'server.log'}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=5
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Server startup timed out")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/v1/models", timeout=10
            ) as response:
                models = json.load(response)["data"]
            if len(models) != 1:
                raise RuntimeError(f"Expected one served model, got {models}")
            yield process, models[0]["id"]
        finally:
            stop_process(process)


def render(args, case, model, destination):
    import numpy as np
    import soundfile as sf

    payload = dict(case["request"], model=model)
    encoded = json.dumps(payload).encode()
    record = {"name": case["name"], "request": payload, "status": "error"}
    started = time.perf_counter()
    record["started_s"] = started
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/audio/speech",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            body = response.read()
            record["http_status"] = response.status
        record["received_s"] = time.perf_counter()
        record["latency_s"] = record["received_s"] - started
        destination.write_bytes(body)
        waveform, rate = sf.read(io.BytesIO(body), always_2d=True, dtype="float32")
        if rate != 32000 or waveform.shape[1] != 2 or not waveform.size:
            raise ValueError(f"Invalid waveform: {waveform.shape}, sample rate {rate}")
        if not np.isfinite(waveform).all():
            raise ValueError("Non-finite waveform")
        record.update(
            status="ok",
            audio_s=len(waveform) / rate,
            samples=len(waveform),
            sample_rate=rate,
            peak=float(np.abs(waveform).max()),
            rms=float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
            wav_sha256=hashlib.sha256(body).hexdigest(),
        )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record.setdefault("latency_s", time.perf_counter() - started)
    return record


def render_group(args, cases, model, output, concurrency):
    output.mkdir()
    started = time.perf_counter()
    records = []
    with (output / "events.jsonl").open("w") as events:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(render, args, case, model, output / f"{case['name']}.wav")
                for case in cases
            ]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                records.append(record)
                events.write(json.dumps(record) + "\n")
                events.flush()
    elapsed = time.perf_counter() - started
    successful = [r for r in records if r["status"] == "ok"]
    valid = len(successful) == len(cases)
    total_audio = sum(r["audio_s"] for r in successful)
    latencies = [r["latency_s"] for r in successful]
    p95 = (
        statistics.quantiles(latencies, n=100, method="inclusive")[94]
        if len(latencies) > 1
        else (latencies[0] if latencies else None)
    )
    edges = sorted(
        [(r["started_s"], 1) for r in successful]
        + [(r["received_s"], -1) for r in successful]
    )
    active = peak_active = 0
    for _, change in edges:
        active += change
        peak_active = max(active, peak_active)
    summary = {
        "harness_status": "ok" if valid else "error",
        "concurrency": concurrency,
        "submitted": len(cases),
        "completed": len(successful),
        "wall_s": elapsed,
        "audio_s": total_audio,
        "audio_seconds_per_wall_second": total_audio / elapsed if valid else None,
        "requests_per_second": len(successful) / elapsed if valid else None,
        "mean_latency_s": (
            statistics.mean(r["latency_s"] for r in successful) if valid else None
        ),
        "peak_client_requests_in_flight": peak_active,
        "p50_latency_s": statistics.median(latencies) if valid else None,
        "p95_latency_s": p95 if valid else None,
    }
    write_json(output / "summary.json", summary)
    if not valid:
        raise RuntimeError(
            f"{len(cases) - len(successful)} requests failed; see {output}"
        )
    return summary


def benchmark_cases(case, count):
    return [
        {
            "name": f"request_{index:03d}",
            "request": dict(case["request"], seed=10000 + index),
        }
        for index in range(count)
    ]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((HERE / "cookbook_requests.json").read_text())
    write_json(args.output / "cookbook_requests.json", manifest)
    checkpoint = checkpoint_identity(args.model_path)
    write_json(args.output / "checkpoint.json", checkpoint)
    identities = {
        name: repository_identity(repo)
        for name, repo in (("base", args.base_repo), ("candidate", args.candidate_repo))
    }
    sglang_source = importlib.util.find_spec("sglang").origin
    sglang_git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(sglang_source).parent,
        text=True,
        capture_output=True,
    )
    write_json(
        args.output / "environment.json",
        {
            "repositories": identities,
            "python": sys.version,
            "executable": sys.executable,
            "model_path": str(args.model_path),
            "dtype": args.dtype,
            "concurrency": args.concurrency,
            "packages": sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
            ),
            "sglang_source": sglang_source,
            "sglang_commit": (
                sglang_git.stdout.strip() if sglang_git.returncode == 0 else None
            ),
            "gpu": command_output(["nvidia-smi", "-q"]),
            "harness": repository_identity(HERE.parent.parent),
        },
    )
    status = {"harness_status": "error", "completed_runs": []}
    try:
        profiles = {"single": args.single_gpu, "dual": args.dual_gpus}
        for layout in args.layouts:
            devices = profiles[layout]
            rounds = args.rounds if args.mode == "performance" else 1
            for repeat in range(rounds):
                arms = [("base", args.base_repo), ("candidate", args.candidate_repo)]
                if repeat % 2:
                    arms.reverse()
                for arm, repo in arms:
                    if checkpoint_identity(args.model_path) != checkpoint:
                        raise RuntimeError(
                            "Checkpoint files changed during the comparison"
                        )
                    name = f"{layout}_round{repeat}_{arm}"
                    output = args.output / name
                    output.mkdir()
                    with server(
                        args, repo, output, devices, capture=args.mode == "accuracy"
                    ) as (process, model):
                        if args.mode == "cookbook":
                            render_group(
                                args,
                                manifest["cases"][:-3],
                                model,
                                output / "cookbook",
                                1,
                            )
                            render_group(
                                args,
                                manifest["cases"][-3:],
                                model,
                                output / "cookbook_parallel",
                                4,
                            )
                        elif args.mode == "accuracy":
                            render_group(
                                args,
                                manifest["cases"][:1],
                                model,
                                output / "accuracy",
                                1,
                            )
                        else:
                            for concurrency in args.concurrency:
                                cases = benchmark_cases(
                                    manifest["cases"][0], args.requests
                                )
                                render_group(
                                    args,
                                    cases[:concurrency],
                                    model,
                                    output / f"c{concurrency}_warmup",
                                    concurrency,
                                )
                                with (output / f"c{concurrency}_gpu.csv").open(
                                    "w"
                                ) as gpu_log:
                                    monitor = subprocess.Popen(
                                        [
                                            "nvidia-smi",
                                            f"--id={devices}",
                                            "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm",
                                            "--format=csv",
                                            "--loop-ms=1000",
                                        ],
                                        stdout=gpu_log,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True,
                                    )
                                    try:
                                        log_start = (
                                            (output / "server.log").stat().st_size
                                        )
                                        result = render_group(
                                            args,
                                            cases,
                                            model,
                                            output / f"c{concurrency}",
                                            concurrency,
                                        )
                                        with (output / "server.log").open("rb") as log:
                                            log.seek(log_start)
                                            completions = re.findall(
                                                rb"MiniMax Music 3 AR done request=\S+ frames=(\d+)",
                                                log.read(),
                                            )
                                        frames = (
                                            sum(map(int, completions))
                                            if len(completions) == len(cases)
                                            else None
                                        )
                                        result["ar_completion_logs"] = len(completions)
                                        result["ar_frames"] = frames
                                        result["ar_frames_per_wall_second"] = (
                                            frames / result["wall_s"]
                                            if frames is not None
                                            else None
                                        )
                                        write_json(
                                            output / f"c{concurrency}" / "summary.json",
                                            result,
                                        )
                                    finally:
                                        stop_process(monitor)
                        if process.poll() is not None:
                            raise RuntimeError("Server exited during qualification")
                    status["completed_runs"].append(name)
                    write_json(args.output / "status.json", status)
        status["harness_status"] = "ok"
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "status.json", status)


def tensor_metrics(left, right):
    import torch

    if left.shape != right.shape:
        return {
            "match": False,
            "error": "shape mismatch",
            "base_shape": list(left.shape),
            "candidate_shape": list(right.shape),
        }
    a, b = left.double().flatten(), right.double().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite or not a.numel():
        return {
            "match": False,
            "finite": finite,
            "elements": a.numel(),
            "error": "non-finite or empty tensor",
        }
    delta = a - b
    norm = a.norm().item()
    denominator = norm * b.norm().item()
    return {
        "match": bool(left.dtype == right.dtype and torch.equal(left, right)),
        "finite": True,
        "shape": list(left.shape),
        "base_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": delta.norm().item() / norm if norm else None,
        "cosine": torch.dot(a, b).item() / denominator if denominator else None,
    }


def compare(args):
    import torch

    left = {p.name: p for p in args.base.glob("*.pt")}
    right = {p.name: p for p in args.candidate.glob("*.pt")}
    names = sorted(left.keys() | right.keys())
    records = {}
    for name in names:
        if name not in left or name not in right:
            records[name] = {"match": False, "error": "missing tensor"}
        else:
            records[name] = tensor_metrics(
                torch.load(left[name], map_location="cpu", weights_only=True),
                torch.load(right[name], map_location="cpu", weights_only=True),
            )
    complete = bool(names) and left.keys() == right.keys()
    groups = [
        {name.removesuffix(suffix) for name in names if name.endswith(suffix)}
        for suffix in ("_hidden.pt", "_condition.pt", "_latent.pt")
    ]
    complete = complete and bool(groups[0]) and groups[0] == groups[1] == groups[2]
    exact = complete and all(r["match"] for r in records.values())
    write_json(args.output, {"complete": complete, "exact": exact, "tensors": records})
    if not complete or any("error" in r for r in records.values()):
        raise RuntimeError("Incomplete or invalid tensor comparison")
    if args.require_exact and not exact:
        raise RuntimeError("Tensor comparison differs; see the report")


def summarize(args):
    status = json.loads((args.results / "status.json").read_text())
    environment = json.loads((args.results / "environment.json").read_text())
    if status["harness_status"] != "ok":
        raise RuntimeError("Incomplete runs cannot be performance evidence")
    rows = []
    for name in status["completed_runs"]:
        if not name.endswith("_base"):
            continue
        candidate = name.removesuffix("_base") + "_candidate"
        for concurrency in environment["concurrency"]:
            base = json.loads(
                (args.results / name / f"c{concurrency}" / "summary.json").read_text()
            )
            head = json.loads(
                (
                    args.results / candidate / f"c{concurrency}" / "summary.json"
                ).read_text()
            )
            if base["harness_status"] != "ok" or head["harness_status"] != "ok":
                raise RuntimeError(
                    "Failed request group cannot be performance evidence"
                )
            rows.append(
                {
                    "run": name.removesuffix("_base"),
                    "concurrency": concurrency,
                    "base": base,
                    "candidate": head,
                    "wall_change_percent": 100 * (head["wall_s"] / base["wall_s"] - 1),
                    "audio_throughput_change_percent": 100
                    * (
                        head["audio_seconds_per_wall_second"]
                        / base["audio_seconds_per_wall_second"]
                        - 1
                    ),
                    "equal_generated_frames": base.get("ar_frames") is not None
                    and base["ar_frames"] == head.get("ar_frames"),
                }
            )
    if not rows:
        raise RuntimeError("No paired performance rounds found")
    write_json(args.output, rows)
    print("run | c | base seconds | candidate seconds | wall change | equal AR work")
    for row in rows:
        print(
            f"{row['run']} | {row['concurrency']} | {row['base']['wall_s']:.3f} | "
            f"{row['candidate']['wall_s']:.3f} | {row['wall_change_percent']:+.2f}% | "
            f"{row['equal_generated_frames']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--mode", choices=("cookbook", "performance", "accuracy"), required=True
    )
    run_parser.add_argument("--base-repo", type=Path, required=True)
    run_parser.add_argument("--candidate-repo", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--single-gpu", default="0")
    run_parser.add_argument("--dual-gpus", default="0,1")
    run_parser.add_argument(
        "--layouts", nargs="+", choices=("single", "dual"), default=["single", "dual"]
    )
    run_parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    run_parser.add_argument("--port", type=int, default=18000)
    run_parser.add_argument("--requests", type=int, default=32)
    run_parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    run_parser.add_argument("--max-running-requests", type=int, default=16)
    run_parser.add_argument("--rounds", type=int, default=3)
    run_parser.add_argument("--startup-timeout", type=float, default=1800)
    run_parser.add_argument("--request-timeout", type=float, default=1800)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--base", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--require-exact", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--results", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if (
            min(args.concurrency) < 1
            or args.requests < max(args.concurrency)
            or args.rounds < 1
        ):
            parser.error(
                "Concurrency and rounds must be positive; requests must cover concurrency"
            )
        if len(set(args.concurrency)) != len(args.concurrency):
            parser.error("Concurrency values must be distinct")
        for field in ("base_repo", "candidate_repo", "model_path", "output"):
            setattr(args, field, getattr(args, field).resolve())
        if not args.model_path.is_dir():
            parser.error("--model-path must be a complete local checkpoint snapshot")
        run(args)
    elif args.command == "compare":
        compare(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
