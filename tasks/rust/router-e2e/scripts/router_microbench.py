#!/usr/bin/env python3
"""Real-socket router microbenchmarks using synthetic workers and oha."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from common import (
    ProcessGroupSampler,
    assert_zero_in_flight,
    fetch_json,
    local_process_env,
    metrics_dict,
    terminate_process_group,
    wait_http,
)

CONCURRENCIES = (1, 8, 32, 128, 512)
SCENARIOS = {
    "small_json": {
        "payload_bytes": 128,
        "requests_min": 5_000,
        "requests_per_c": 200,
    },
    "large_json": {
        "payload_bytes": 1_048_576,
        "requests_min": 128,
        "requests_per_c": 4,
    },
    "sse": {"payload_bytes": 128, "requests_min": 500, "requests_per_c": 100},
}


class SyntheticHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "router-qualification-worker"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path in ("/health", "/ready"):
            self._fixed(200, b"ok\n", "text/plain")
            return
        self._fixed(404, b"not found\n", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        remaining = length
        while remaining:
            chunk = self.rfile.read(min(65_536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        scenario = self.headers.get("x-router-bench-scenario", "small_json")
        if scenario == "variable":
            counter = self.server.next_request()  # type: ignore[attr-defined]
            time.sleep(0.002 if counter % 4 else 0.040)
        if scenario == "sse":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for data in (
                b'data: {"delta":"a"}\n\n',
                b'data: {"delta":"b"}\n\n',
                b"data: [DONE]\n\n",
            ):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                time.sleep(0.005)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self._fixed(
            200,
            b'{"id":"synthetic","choices":[{"message":{"content":"ok"}}]}',
            "application/json",
        )

    def _fixed(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SyntheticServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, SyntheticHandler)
        self._counter = 0
        self._lock = threading.Lock()

    def next_request(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def worker_main(port: int) -> int:
    server = SyntheticServer(("127.0.0.1", port))
    stop = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("sglang-omni-router/target/release/sgl-omni-router"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--oha", default="oha")
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--worker-port", type=int, default=18011)
    parser.add_argument("--max-requests", type=int, default=100_000)
    parser.add_argument("--oha-timeout-s", type=float, default=90.0)
    parser.add_argument("--worker-mode", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_oha(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"oha executable not found: {executable}")
    help_result = subprocess.run(
        [resolved, "--help"],
        capture_output=True,
        text=True,
        env=local_process_env(),
        check=False,
    )
    if help_result.returncode != 0 or not all(
        option in help_result.stdout for option in ("--output-format", "-D")
    ):
        raise RuntimeError("oha must support --output-format json and -D BODY_PATH")
    return resolved


def payload(size: int) -> str:
    prefix = '{"model":"bench-model","messages":[{"role":"user","content":"'
    suffix = '"}]}'
    return prefix + ("x" * max(0, size - len(prefix) - len(suffix))) + suffix


def write_rust_config(
    path: Path,
    worker_ports: tuple[int, int],
    router_port: int,
    policy: str,
    log_filter: str,
) -> None:
    worker_blocks = []
    for index, port in enumerate(worker_ports, start=1):
        worker_blocks.append(
            f"""[[workers]]
worker_id = "worker-{index}"
base_url = "http://127.0.0.1:{port}/"
trust_domain = "local"
default_model_id = "bench-model"
health_path = "/health"

[workers.capacity]
generation_http = 1024

[[workers.service_profiles]]
service = "generation_http"
model_ids = ["bench-model"]
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming", "streaming"]
"""
        )
    path.write_text(
        f"""schema_version = 1

[server]
listen = "127.0.0.1:{router_port}"
max_connections = 4096

[shutdown]
drain_timeout_ms = 15000

[logging]
format = "json"
filter = "{log_filter}"

[router]
strategy = "{policy}"
max_concurrent_classifications = 4

[admission]
global = 2048
generation_http = 2048

[health]
interval_ms = 1000
timeout_ms = 500
success_threshold = 1
failure_threshold = 3
max_concurrent_probes = 2

[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 2097152
buffered_request_total_bytes = 1073741824
streamed_request_max_bytes = 2097152
connect_timeout_ms = 2000
request_timeout_ms = 30000
pool_idle_timeout_ms = 90000
pool_max_idle_per_host = 1024

{"".join(worker_blocks)}""",
        encoding="utf-8",
    )


def router_command(
    candidate: str, args: argparse.Namespace, config: Path, policy: str
) -> list[str]:
    if candidate == "rust":
        return [str(args.rust_binary), "--config", str(config)]
    return [
        args.python,
        "-m",
        "sglang_omni_router.serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
        "--worker-urls",
        f"http://127.0.0.1:{args.worker_port}",
        f"http://127.0.0.1:{args.worker_port + 1}",
        "--model",
        "bench-model",
        "--policy",
        policy,
        "--health-success-threshold",
        "1",
        "--health-failure-threshold",
        "3",
        "--health-check-interval-secs",
        "1",
        "--max-connections",
        "2048",
        "--max-inflight",
        "2048",
        "--log-level",
        "info",
    ]


def oha_command(
    oha: str, url: str, concurrency: int, requests: int, scenario: str, body_file: Path
) -> list[str]:
    return [
        oha,
        "--output-format",
        "json",
        "--no-tui",
        "-n",
        str(requests),
        "-c",
        str(concurrency),
        "-m",
        "POST",
        "-H",
        "content-type: application/json",
        "-H",
        f"x-router-bench-scenario: {scenario}",
        "-D",
        str(body_file),
        url,
    ]


def parse_oha(raw: dict[str, Any]) -> dict[str, Any]:
    summary = raw.get("summary", {}) if isinstance(raw.get("summary"), dict) else {}
    metrics = raw.get("metrics", {}) if isinstance(raw.get("metrics"), dict) else {}
    latency_ms = (
        metrics.get("latency_ms", {})
        if isinstance(metrics.get("latency_ms"), dict)
        else {}
    )
    percentiles = (
        raw.get("latencyPercentiles", {})
        if isinstance(raw.get("latencyPercentiles"), dict)
        else {}
    )
    status = (
        raw.get("statusCodeDistribution", {})
        if isinstance(raw.get("statusCodeDistribution"), dict)
        else {}
    )
    errors = (
        raw.get("errorDistribution", {})
        if isinstance(raw.get("errorDistribution"), dict)
        else {}
    )
    requests_per_second = metrics.get(
        "requests_per_sec", summary.get("requestsPerSec", raw.get("requestsPerSec"))
    )
    p50_ms = latency_ms.get("p50")
    p95_ms = latency_ms.get("p95")
    p99_ms = latency_ms.get("p99")
    if not latency_ms:
        p50_ms = _seconds_to_ms(percentiles.get("p50"))
        p95_ms = _seconds_to_ms(percentiles.get("p95"))
        p99_ms = _seconds_to_ms(percentiles.get("p99"))
    return {
        "requests_per_second": requests_per_second,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "status_codes": status,
        "errors": errors,
        "success_rate": metrics.get(
            "success_rate", summary.get("successRate", raw.get("successRate"))
        ),
    }


def _seconds_to_ms(value: object) -> float | None:
    return float(value) * 1000.0 if isinstance(value, (int, float)) else None


def run_oha(command: list[str], timeout_s: float) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=local_process_env(),
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(f"oha failed ({result.returncode}): {result.stderr.strip()}")
    try:
        raw = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("oha did not emit valid JSON") from exc
    if not isinstance(raw, dict):
        raise TypeError("oha JSON root is not an object")
    return raw, elapsed


def run_direct_pair(
    oha: str,
    args: argparse.Namespace,
    concurrency: int,
    requests: int,
    scenario: str,
    body_file: Path,
) -> dict[str, Any]:
    clients = 1 if concurrency == 1 else 2
    split_requests = (
        (requests,) if clients == 1 else (requests // 2, requests - requests // 2)
    )
    split_concurrency = (
        (concurrency,)
        if clients == 1
        else (concurrency // 2, concurrency - concurrency // 2)
    )
    commands = [
        oha_command(
            oha,
            f"http://127.0.0.1:{args.worker_port + index}/v1/chat/completions",
            split_concurrency[index],
            split_requests[index],
            scenario,
            body_file,
        )
        for index in range(clients)
    ]
    started = time.monotonic()
    processes = [
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=local_process_env(),
        )
        for command in commands
    ]
    raws: list[dict[str, Any]] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=args.oha_timeout_s)
            if process.returncode != 0:
                raise RuntimeError(
                    f"direct oha failed ({process.returncode}): {stderr.strip()}"
                )
            parsed = json.loads(stdout)
            if not isinstance(parsed, dict):
                raise TypeError("direct oha JSON root is not an object")
            raws.append(parsed)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    elapsed = time.monotonic() - started
    children = [parse_oha(raw) for raw in raws]
    child_rps = [child["requests_per_second"] for child in children]
    if not all(isinstance(value, (int, float)) for value in child_rps):
        raise TypeError("direct oha JSON did not contain numeric requests/sec")
    return {
        "kind": "direct_pair",
        "scenario": scenario,
        "concurrency": concurrency,
        "requests": requests,
        "elapsed_s": elapsed,
        "requests_per_second": requests / elapsed,
        "child_requests_per_second_sum": sum(float(value) for value in child_rps),
        "children": children,
        "raw": raws,
    }


def run_router_case(
    candidate: str,
    policy: str,
    log_filter: str,
    oha: str,
    args: argparse.Namespace,
    work: Path,
    scenario: str,
    concurrency: int,
    requests: int,
    body_file: Path,
) -> dict[str, Any]:
    config = work / f"{candidate}-{policy}-{log_filter}.toml"
    write_rust_config(
        config,
        (args.worker_port, args.worker_port + 1),
        args.router_port,
        policy,
        log_filter,
    )
    command = router_command(
        candidate,
        args,
        config,
        "least_request" if policy == "least_requests" else policy,
    )
    log = (
        work
        / f"{candidate}-{policy}-{log_filter}-{scenario}-c{concurrency}-{time.time_ns()}.log"
    )
    with log.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=local_process_env(),
        )
        sampler = ProcessGroupSampler(process.pid)
        try:
            wait_http(
                f"http://127.0.0.1:{args.router_port}/{'ready' if candidate == 'rust' else 'health'}",
                30.0,
            )
            sampler.start()
            raw, elapsed = run_oha(
                oha_command(
                    oha,
                    f"http://127.0.0.1:{args.router_port}/v1/chat/completions",
                    concurrency,
                    requests,
                    scenario,
                    body_file,
                ),
                args.oha_timeout_s,
            )
            if candidate == "rust":
                diagnostics = fetch_json(
                    f"http://127.0.0.1:{args.router_port}/diagnostics"
                )
                if diagnostics is None:
                    raise AssertionError(
                        "Rust diagnostics unavailable after microbench"
                    )
                assert_zero_in_flight(diagnostics)
            metrics = sampler.stop()
            sampler = None  # type: ignore[assignment]
        finally:
            if sampler is not None:
                sampler.stop()
            terminate_process_group(process)
    return {
        "kind": "router",
        "candidate": candidate,
        "policy": policy,
        "log_filter": log_filter,
        "scenario": scenario,
        "concurrency": concurrency,
        "requests": requests,
        "elapsed_s": elapsed,
        "oha": parse_oha(raw),
        "process_group": metrics_dict(metrics),
        "log_file": str(log),
        "raw": raw,
    }


def numeric_rps(result: dict[str, Any]) -> float:
    value = (
        result.get("requests_per_second")
        if result["kind"] == "direct_pair"
        else result["oha"].get("requests_per_second")
    )
    if not isinstance(value, (int, float)):
        raise TypeError("oha JSON did not contain numeric requestsPerSec")
    return float(value)


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for concurrency in CONCURRENCIES:
            group = [
                r
                for r in results
                if r.get("scenario") == scenario
                and r.get("concurrency") == concurrency
                and (
                    r.get("kind") == "direct_pair"
                    or (
                        r.get("policy") == "round_robin"
                        and r.get("log_filter") == "error"
                    )
                )
            ]
            keyed = {
                "direct": next(
                    (r for r in group if r.get("kind") == "direct_pair"), None
                ),
                "python": next(
                    (r for r in group if r.get("candidate") == "python"), None
                ),
                "rust": next((r for r in group if r.get("candidate") == "rust"), None),
            }
            if any(value is None for value in keyed.values()):
                continue
            direct_rps = numeric_rps(keyed["direct"])
            python_rps = numeric_rps(keyed["python"])
            rust_rps = numeric_rps(keyed["rust"])
            headroom = direct_rps / max(python_rps, rust_rps)
            python_cpu = keyed["python"]["process_group"].get("cpu_seconds")
            rust_cpu = keyed["rust"]["process_group"].get("cpu_seconds")
            cpu_ratio = None
            if (
                isinstance(python_cpu, (int, float))
                and isinstance(rust_cpu, (int, float))
                and rust_cpu > 0
            ):
                cpu_ratio = float(python_cpu) / float(rust_cpu)
            python_p99 = keyed["python"]["oha"].get("p99_ms")
            rust_p99 = keyed["rust"]["oha"].get("p99_ms")
            tail_ratio = None
            if (
                isinstance(python_p99, (int, float))
                and isinstance(rust_p99, (int, float))
                and python_p99 > 0
            ):
                tail_ratio = float(rust_p99) / float(python_p99)
            rust_errors = keyed["rust"]["oha"].get("errors")
            python_errors = keyed["python"]["oha"].get("errors")
            rust_success = keyed["rust"]["oha"].get("success_rate")
            python_success = keyed["python"]["oha"].get("success_rate")
            direct_valid = all(
                child.get("success_rate") == 1.0 and not child.get("errors")
                for child in keyed["direct"]["children"]
            )
            if not direct_valid:
                status = "fail"
            elif (
                direct_rps < python_rps * 1.20
                or cpu_ratio is None
                or tail_ratio is None
            ):
                status = "inconclusive"
            elif (
                rust_errors
                or python_errors
                or rust_success != 1.0
                or python_success != 1.0
                or tail_ratio > 1.10
            ):
                status = "fail"
            elif rust_rps >= python_rps * 1.15 and cpu_ratio >= 1.15:
                status = "pass"
            else:
                status = "fail"
            decisions.append(
                {
                    "scenario": scenario,
                    "concurrency": concurrency,
                    "status": status,
                    "direct_over_faster_router_ratio": headroom,
                    "direct_over_python_ratio": direct_rps / python_rps,
                    "rust_over_python_throughput_ratio": rust_rps / python_rps,
                    "python_over_rust_cpu_seconds_ratio": cpu_ratio,
                    "rust_over_python_p99_ratio": tail_ratio,
                    "thresholds": {
                        "minimum_direct_over_python_ratio": 1.20,
                        "minimum_rust_throughput_ratio": 1.15,
                        "minimum_cpu_efficiency_ratio": 1.15,
                        "maximum_p99_ratio": 1.10,
                    },
                }
            )
    return {
        "comparisons": decisions,
        "note": "Tail/error and CPU-efficiency gates must also pass; RESULTS_TEMPLATE.md records the final decision.",
    }


def main() -> int:
    args = parse_args()
    if args.worker_mode is not None:
        return worker_main(args.worker_mode)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if not args.rust_binary.is_file() or not os.access(args.rust_binary, os.X_OK):
        raise RuntimeError(
            f"release Rust binary is missing or not executable: {args.rust_binary}"
        )
    oha = validate_oha(args.oha)
    results: list[dict[str, Any]] = []
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output-dir",
                str(args.output_dir / "unused"),
                "--worker-mode",
                str(args.worker_port + index),
            ],
            start_new_session=True,
            env=local_process_env(),
        )
        for index in range(2)
    ]
    try:
        for index in range(2):
            wait_http(f"http://127.0.0.1:{args.worker_port + index}/health", 10.0)
        work = args.output_dir / "support"
        work.mkdir()
        for scenario, settings in SCENARIOS.items():
            body_file = work / f"{scenario}.json"
            body_file.write_text(payload(settings["payload_bytes"]), encoding="utf-8")
            for concurrency in CONCURRENCIES:
                requests = min(
                    args.max_requests,
                    max(
                        settings["requests_min"],
                        concurrency * settings["requests_per_c"],
                    ),
                )
                results.append(
                    run_direct_pair(
                        oha, args, concurrency, requests, scenario, body_file
                    )
                )
                for candidate in ("python", "rust"):
                    results.append(
                        run_router_case(
                            candidate,
                            "round_robin",
                            "error",
                            oha,
                            args,
                            work,
                            scenario,
                            concurrency,
                            requests,
                            body_file,
                        )
                    )
        sentinel_requests = min(args.max_requests, 40_000)
        variable_body = work / "variable.json"
        variable_body.write_text(payload(128), encoding="utf-8")
        for policy in ("round_robin", "least_requests"):
            results.append(
                run_router_case(
                    "rust",
                    policy,
                    "error",
                    oha,
                    args,
                    work,
                    "variable",
                    128,
                    sentinel_requests,
                    variable_body,
                )
            )
        for repetition in range(1, 4):
            for log_filter in ("error", "info"):
                sentinel = run_router_case(
                    "rust",
                    "round_robin",
                    log_filter,
                    oha,
                    args,
                    work,
                    "small_json",
                    128,
                    sentinel_requests,
                    variable_body,
                )
                sentinel["tracing_sentinel_repetition"] = repetition
                results.append(sentinel)
    finally:
        for worker in workers:
            terminate_process_group(worker)
    output = {
        "schema_version": 1,
        "concurrencies": CONCURRENCIES,
        "results": results,
        "decision": decide(results),
    }
    (args.output_dir / "microbench.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "microbench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
