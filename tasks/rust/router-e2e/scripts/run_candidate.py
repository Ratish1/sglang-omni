#!/usr/bin/env python3
"""Run one benchmark trial against a temporary router and persistent workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from common import (
    ProcessGroupSampler,
    assert_workers_healthy,
    assert_zero_in_flight,
    benchmark_process_env,
    counter_moved,
    fetch_json,
    fetch_text,
    local_process_env,
    metrics_dict,
    request_counters,
    substitute_placeholders,
    terminate_process_group,
    wait_http,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("rust", "python"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--worker-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy",
        choices=("round_robin", "least_requests", "least_request"),
        required=True,
    )
    parser.add_argument("--rust-config", type=Path)
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("sglang-omni-router/target/release/sgl-omni-router"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    parser.add_argument("benchmark", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.benchmark and args.benchmark[0] == "--":
        args.benchmark = args.benchmark[1:]
    if not args.benchmark:
        parser.error("benchmark command is required after --")
    if len(args.worker_url) != 2:
        parser.error("exactly two --worker-url values are required")
    if args.candidate == "rust" and args.rust_config is None:
        parser.error("--rust-config is required for the Rust candidate")
    if args.candidate == "rust" and args.policy == "least_request":
        parser.error("Rust policy spelling is least_requests")
    if args.candidate == "python" and args.policy == "least_requests":
        parser.error("Python policy spelling is least_request")
    if args.candidate == "rust":
        validate_rust_contract(args, parser)
    return args


def validate_rust_contract(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    try:
        with args.rust_config.open("rb") as config_file:
            config = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"cannot read Rust config: {exc}")
    expected_listen = f"127.0.0.1:{args.router_port}"
    if config.get("server", {}).get("listen") != expected_listen:
        parser.error(f"Rust config must listen on {expected_listen}")
    if config.get("router", {}).get("strategy") != args.policy:
        parser.error("Rust config strategy does not match --policy")
    workers = config.get("workers")
    if not isinstance(workers, list) or len(workers) != 2:
        parser.error("Rust config must contain exactly two workers")
    configured_urls = {
        str(worker.get("base_url", "")).rstrip("/") for worker in workers
    }
    if configured_urls != {url.rstrip("/") for url in args.worker_url}:
        parser.error("Rust config worker URLs do not match --worker-url")
    if any(worker.get("default_model_id") != args.model for worker in workers):
        parser.error("Rust config worker models do not match --model")


def build_router_command(args: argparse.Namespace) -> list[str]:
    if args.candidate == "rust":
        return [str(args.rust_binary), "--config", str(args.rust_config)]
    return [
        args.python,
        "-m",
        "sglang_omni_router.serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
        "--worker-urls",
        *args.worker_url,
        "--model",
        args.model,
        "--policy",
        args.policy,
        "--health-success-threshold",
        "1",
        "--health-failure-threshold",
        "3",
        "--health-check-interval-secs",
        "2",
        "--log-level",
        "info",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    router_url = f"http://127.0.0.1:{args.router_port}"
    command = build_router_command(args)
    benchmark = substitute_placeholders(
        args.benchmark,
        {
            "router_port": str(args.router_port),
            "router_url": router_url,
            "output_dir": str(args.output_dir),
        },
    )
    log_path = args.output_dir / "router.log"
    benchmark_log_path = args.output_dir / "benchmark.log"
    metadata: dict[str, object] = {
        "candidate": args.candidate,
        "policy": args.policy,
        "router_command": command,
        "benchmark_command": benchmark,
        "benchmark_log": str(benchmark_log_path),
        "worker_urls": args.worker_url,
        "started_unix_s": time.time(),
    }
    process: subprocess.Popen[bytes] | None = None
    sampler: ProcessGroupSampler | None = None
    exit_code = 1
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=local_process_env(),
            )
            ready_path = "/ready" if args.candidate == "rust" else "/health"
            wait_http(router_url + ready_path, args.startup_timeout_s)
            metadata["pre_diagnostics"] = fetch_json(
                router_url
                + ("/diagnostics" if args.candidate == "rust" else "/workers")
            )
            worker_counters_before = {
                url: request_counters(fetch_text(url.rstrip("/") + "/metrics"))
                for url in args.worker_url
            }
            metadata["worker_request_counters_before"] = worker_counters_before
            sampler = ProcessGroupSampler(process.pid)
            sampler.start()
            benchmark_started = time.monotonic()
            with benchmark_log_path.open("wb") as benchmark_log:
                completed = subprocess.run(
                    benchmark,
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    env=benchmark_process_env(),
                    check=False,
                )
            metadata["benchmark_wall_s"] = time.monotonic() - benchmark_started
            metadata["benchmark_exit_code"] = completed.returncode
            metrics = sampler.stop()
            sampler = None
            metadata["router_process_group"] = metrics_dict(metrics)
            post_url = router_url + (
                "/diagnostics" if args.candidate == "rust" else "/workers"
            )
            post = fetch_json(post_url)
            metadata["post_diagnostics"] = post
            worker_counters_after = {
                url: request_counters(fetch_text(url.rstrip("/") + "/metrics"))
                for url in args.worker_url
            }
            metadata["worker_request_counters_after"] = worker_counters_after
            traffic = {
                url: counter_moved(
                    worker_counters_before[url], worker_counters_after[url]
                )
                for url in args.worker_url
            }
            metadata["worker_request_counter_moved"] = traffic
            if all(value is not None for value in traffic.values()) and not all(
                traffic.values()
            ):
                raise AssertionError(
                    f"both worker request counters did not advance: {traffic}"
                )
            if args.candidate == "rust":
                if post is None:
                    raise AssertionError(
                        "Rust /diagnostics was unavailable after benchmark"
                    )
                assert_zero_in_flight(post)
            if post is None:
                raise AssertionError("post-trial router diagnostics were unavailable")
            assert_workers_healthy(post, args.candidate)
            if completed.returncode != 0:
                raise RuntimeError(f"benchmark exited with {completed.returncode}")
            exit_code = 0
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if sampler is not None:
            metadata["router_process_group"] = metrics_dict(sampler.stop())
        if process is not None:
            terminate_process_group(process)
            metadata["router_exit_code"] = process.returncode
        metadata["finished_unix_s"] = time.time()
        metadata["exit_code"] = exit_code
        (args.output_dir / "trial.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
