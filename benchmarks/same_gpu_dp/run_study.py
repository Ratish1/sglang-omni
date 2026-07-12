#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run same-GPU DP capacity calibration or a matrix from a YAML study file."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

COMMON_KEYS = {
    "gpu_uuid": "GPU_UUID",
    "numa_node": "NUMA_NODE",
    "model": "MODEL",
    "model_name": "MODEL_NAME",
    "meta": "META",
    "client_driver": "CLIENT_DRIVER",
    "tts_manifest": "TTS_MANIFEST",
    "router_cores": "ROUTER_CORES",
    "router_policy": "ROUTER_POLICY",
    "bench_lang": "BENCH_LANG",
    "ref_format": "REF_FORMAT",
    "allowed_local_media_path": "ALLOWED_LOCAL_MEDIA_PATH",
    "max_running_requests": "MAX_RUNNING_REQUESTS",
    "cuda_graph_max_bs": "CUDA_GRAPH_MAX_BS",
    "max_new_tokens": "MAX_NEW_TOKENS",
    "max_samples": "MAX_SAMPLES",
    "warmup": "WARMUP",
    "seed": "SEED",
    "kv_equality": "KV_EQUALITY",
    "require_idle_gpu": "REQUIRE_IDLE_GPU",
    "server_ready_timeout": "SERVER_READY_TIMEOUT",
    "mps_ready_timeout": "MPS_READY_TIMEOUT",
    "shutdown_timeout": "SHUTDOWN_TIMEOUT",
    "mps_tmp_root": "MPS_TMP_ROOT",
}
MATRIX_KEYS = {
    "order": "MATRIX_ORDER",
    "concurrency_values": "CONCURRENCY_VALUES",
    "repetitions": "REPETITIONS",
    "shuffle_seed": "SHUFFLE_SEED",
    "run_label": "RUN_LABEL",
    "out_root": "OUT_ROOT",
}
CALIBRATION_KEYS = {
    "dps": "CALIBRATION_DPS",
    "repetitions": "CALIBRATION_REPETITIONS",
    "mps_modes": "CALIBRATION_MPS_MODES",
    "margin_basis_points": "CALIBRATION_MARGIN_BPS",
    "label": "CALIBRATION_LABEL",
    "root": "CALIBRATION_ROOT",
}
LAYOUT_KEYS = {
    "server_core_sets": ("SERVER_CORE_SETS", ";"),
    "client_core_sets": ("CLIENT_CORE_SETS", ";"),
    "mem_fractions": ("MEM_FRACTIONS", ","),
    "max_total_tokens": ("MAX_TOTAL_TOKENS", None),
    "mps_thread_percentages": ("MPS_THREAD_PERCENTAGES", ","),
    "mps_pinned_mem_limits": ("MPS_PINNED_MEM_LIMITS", ";"),
}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _stringify(value: Any, *, separator: str | None = None) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        if separator is None:
            raise ValueError("this YAML value must be a scalar")
        rendered = separator.join(_stringify(item) for item in value)
    else:
        rendered = str(value)
    expanded = os.path.expandvars(rendered)
    if "${" in expanded:
        raise ValueError(f"unresolved environment variable in {rendered!r}")
    return expanded


def _apply_section(
    env: dict[str, str],
    section: dict[str, Any],
    mapping: dict[str, str],
    label: str,
) -> None:
    unknown = sorted(set(section) - set(mapping))
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(unknown)}")
    for key, value in section.items():
        if value is not None:
            separator = "," if isinstance(value, list) else None
            env[mapping[key]] = _stringify(value, separator=separator)


def build_environment(payload: dict[str, Any]) -> dict[str, str]:
    """Translate a strict version-1 study document into runner variables."""
    if payload.get("version") != 1:
        raise ValueError("study YAML must set version: 1")
    unknown = sorted(
        set(payload) - {"version", "common", "matrix", "calibration", "layouts"}
    )
    if unknown:
        raise ValueError(f"unknown top-level keys: {', '.join(unknown)}")

    env = dict(os.environ)
    _apply_section(
        env,
        _require_mapping(payload.get("common"), "common"),
        COMMON_KEYS,
        "common",
    )
    _apply_section(
        env,
        _require_mapping(payload.get("matrix"), "matrix"),
        MATRIX_KEYS,
        "matrix",
    )
    _apply_section(
        env,
        _require_mapping(payload.get("calibration"), "calibration"),
        CALIBRATION_KEYS,
        "calibration",
    )

    layouts = _require_mapping(payload.get("layouts"), "layouts")
    for raw_dp, raw_layout in layouts.items():
        dp = int(raw_dp)
        if dp not in {1, 2, 3, 4}:
            raise ValueError(f"layout key must be DP 1, 2, 3, or 4; got {raw_dp!r}")
        layout = _require_mapping(raw_layout, f"layouts.{dp}")
        unknown_layout = sorted(set(layout) - set(LAYOUT_KEYS))
        if unknown_layout:
            raise ValueError(f"unknown layouts.{dp} keys: {', '.join(unknown_layout)}")
        for key, value in layout.items():
            suffix, separator = LAYOUT_KEYS[key]
            env[f"DP{dp}_{suffix}"] = _stringify(value, separator=separator)
    return env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("calibrate", "matrix"), default="matrix")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.mode == "calibrate" and args.dry_run:
        raise SystemExit("--dry-run is only supported with --mode matrix")
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("study YAML root must be a mapping")
    try:
        env = build_environment(payload)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    here = Path(__file__).resolve().parent
    if args.mode == "calibrate":
        command = ["bash", str(here / "calibrate_capacity.sh")]
    else:
        command = ["bash", str(here / "run_matrix.sh")]
        if args.dry_run:
            command.append("--dry-run")
    raise SystemExit(subprocess.run(command, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
