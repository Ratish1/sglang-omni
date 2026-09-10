#!/usr/bin/env python3
"""Run a direct, Python-router, and Rust-router concurrency matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from common import substitute_placeholders  # noqa: E402

from tests.test_model.rust_router_config import (  # noqa: E402
    CiRouterTopology,
    render_router_config,
)

CANDIDATES = ("direct", "python-rr", "rust-rr", "python-lr", "rust-lr")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topology", choices=tuple(CiRouterTopology), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-url", action="append", required=True)
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("sglang_omni_router/rust/target/release/sgl-omni-router"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--concurrencies", required=True)
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATES),
        help="Comma-separated ordered subset of direct, python-rr, rust-rr, python-lr, rust-lr.",
    )
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("benchmark", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.benchmark and args.benchmark[0] == "--":
        args.benchmark = args.benchmark[1:]
    if not args.benchmark:
        parser.error("benchmark command is required after --")
    if len(args.worker_url) != 2:
        parser.error("exactly two --worker-url values are required")
    args.concurrencies = _comma_separated_ints(
        args.concurrencies, parser, "--concurrencies"
    )
    args.candidates = _candidate_list(args.candidates, parser)
    if "direct" in args.candidates and any(value % 2 for value in args.concurrencies):
        parser.error("direct-pair aggregate concurrencies must be even")
    return args


def _comma_separated_ints(
    value: str, parser: argparse.ArgumentParser, option: str
) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        parser.error(f"{option} must contain comma-separated integers")
    if (
        not values
        or any(item <= 0 for item in values)
        or len(values) != len(set(values))
    ):
        parser.error(f"{option} must contain unique positive integers")
    return values


def _candidate_list(value: str, parser: argparse.ArgumentParser) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in values if item not in CANDIDATES]
    if not values or unknown or len(values) != len(set(values)):
        parser.error(
            "--candidates must contain a unique ordered subset of "
            + ", ".join(CANDIDATES)
        )
    return values


def _worker_port(url: str) -> int:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"qualification worker must be loopback HTTP: {url}")
    if parsed.port is None:
        raise ValueError(f"qualification worker URL must include a port: {url}")
    return parsed.port


def _candidate_spec(candidate: str) -> tuple[str, str] | None:
    return {
        "direct": None,
        "python-rr": ("python", "round_robin"),
        "rust-rr": ("rust", "round_robin"),
        "python-lr": ("python", "least_request"),
        "rust-lr": ("rust", "least_requests"),
    }[candidate]


def _config_path(output_dir: Path, policy: str) -> Path:
    return output_dir / "configs" / f"rust-{policy}.toml"


def _write_configs(args: argparse.Namespace) -> None:
    config_dir = args.output_dir / "configs"
    config_dir.mkdir()
    for policy in ("round_robin", "least_requests"):
        _config_path(args.output_dir, policy).write_text(
            render_router_config(
                topology=CiRouterTopology(args.topology),
                router_port=args.router_port,
                worker_urls=args.worker_url,
                model_name=args.model,
                strategy=policy,
            ),
            encoding="utf-8",
        )


def _render_benchmark(
    benchmark: list[str], *, concurrency: int, candidate: str, policy: str
) -> list[str]:
    return substitute_placeholders(
        benchmark,
        {
            "concurrency": str(concurrency),
            "worker_concurrency": str(concurrency // 2),
            "target_concurrency": str(
                concurrency // 2 if candidate == "direct" else concurrency
            ),
            "candidate": candidate,
            "policy": policy,
        },
    )


def build_trial_command(
    args: argparse.Namespace, candidate: str, concurrency: int
) -> list[str]:
    scripts = Path(__file__).resolve().parent
    output = args.output_dir / f"c{concurrency}" / candidate
    spec = _candidate_spec(candidate)
    if spec is None:
        command = [
            args.python,
            str(scripts / "run_direct_pair.py"),
            "--output-dir",
            str(output),
        ]
        for worker_url in args.worker_url:
            command.extend(["--worker-port", str(_worker_port(worker_url))])
        benchmark = _render_benchmark(
            args.benchmark,
            concurrency=concurrency,
            candidate=candidate,
            policy="direct",
        )
        return [*command, "--", *benchmark]

    implementation, policy = spec
    command = [
        args.python,
        str(scripts / "run_candidate.py"),
        "--candidate",
        implementation,
        "--policy",
        policy,
        "--router-port",
        str(args.router_port),
        "--model",
        args.model,
        "--output-dir",
        str(output),
    ]
    for worker_url in args.worker_url:
        command.extend(["--worker-url", worker_url])
    if implementation == "rust":
        command.extend(
            [
                "--rust-binary",
                str(args.rust_binary),
                "--rust-config",
                str(_config_path(args.output_dir, policy)),
            ]
        )
    benchmark = _render_benchmark(
        args.benchmark,
        concurrency=concurrency,
        candidate=candidate,
        policy=policy,
    )
    return [*command, "--", *benchmark]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trials = [
        (concurrency, candidate, build_trial_command(args, candidate, concurrency))
        for concurrency in args.concurrencies
        for candidate in args.candidates
    ]
    if args.print_plan:
        for _concurrency, _candidate, command in trials:
            print(shlex.join(command))
        return 0

    if any(candidate.startswith("rust-") for candidate in args.candidates):
        if not args.rust_binary.is_file() or not os.access(args.rust_binary, os.X_OK):
            raise RuntimeError(
                f"Rust router binary is not executable: {args.rust_binary}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_configs(args)
    manifest: dict[str, object] = {
        "repository_commit": _git_head(),
        "rust_binary": str(args.rust_binary),
        "rust_binary_sha256": (
            _sha256(args.rust_binary)
            if any(candidate.startswith("rust-") for candidate in args.candidates)
            else None
        ),
        "topology": args.topology,
        "model": args.model,
        "worker_urls": args.worker_url,
        "concurrencies": args.concurrencies,
        "candidates": args.candidates,
        "benchmark_template": args.benchmark,
        "started_unix_s": time.time(),
        "trials": [],
    }
    trial_results: list[dict[str, object]] = []
    manifest["trials"] = trial_results
    manifest_path = args.output_dir / "matrix.json"
    exit_code = 0
    try:
        for concurrency, candidate, command in trials:
            started = time.monotonic()
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            trial = {
                "concurrency": concurrency,
                "candidate": candidate,
                "command": command,
                "wall_s": time.monotonic() - started,
                "exit_code": completed.returncode,
            }
            trial_results.append(trial)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if completed.returncode != 0:
                exit_code = completed.returncode
                break
    except BaseException as exc:
        exit_code = 1
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_unix_s"] = time.time()
        manifest["exit_code"] = exit_code
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
