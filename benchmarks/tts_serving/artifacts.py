# SPDX-License-Identifier: Apache-2.0
"""Artifact writing for the TTS serving benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec


class ArtifactError(RuntimeError):
    """Raised when benchmark artifacts cannot be written."""


def prepare_output_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    try:
        (out_dir / "raw").mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError(
            f"failed to create output directory {out_dir}: {exc}"
        ) from exc
    return out_dir


def write_artifacts(
    out_dir: Path,
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    results: list[ScenarioResult],
    report: dict[str, Any],
) -> None:
    _write_json(out_dir / "results.json", report)
    _write_json(
        out_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": spec.model_name,
            "base_url": spec.base_url,
            "test_type": spec.test_type,
            "profile": spec.params.profile,
            "artifacts": {
                "results": "results.json",
                "requests": "raw/requests.jsonl",
                "events": "raw/events.jsonl",
                "logs": "logs/harness.log",
            },
        },
    )
    _write_jsonl(out_dir / "raw" / "requests.jsonl", scenarios)
    _write_jsonl(out_dir / "raw" / "events.jsonl", results)


def write_harness_log(out_dir: Path, lines: list[str]) -> None:
    try:
        (out_dir / "logs" / "harness.log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise ArtifactError(f"failed to write harness log: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ArtifactError(f"failed to write {path}: {exc}") from exc


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_to_json(row), ensure_ascii=False) + "\n")
    except OSError as exc:
        raise ArtifactError(f"failed to write {path}: {exc}") from exc


def _to_json(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if is_dataclass(value):
        return asdict(value)
    return value
