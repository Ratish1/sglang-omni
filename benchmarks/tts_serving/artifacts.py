# SPDX-License-Identifier: Apache-2.0
"""Artifact writing for the TTS serving benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.scenarios import (
    SCENARIO_SCHEMA_VERSION,
    Scenario,
    scenario_set_hash,
)
from benchmarks.tts_serving.spec import BenchmarkSpec, redact_sensitive_metadata

SENSITIVE_AUDIO_REFERENCE_KEYS = {"audio", "audio_sample", "ref_audio"}


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
            "platform_metadata": redact_sensitive_metadata(spec.platform_metadata),
            "profile": spec.params.profile,
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_set_hash": scenario_set_hash(scenarios),
            "load_stages": [stage.to_json() for stage in spec.params.load_stages],
            "artifacts": {
                "results": "results.json",
                "requests": "raw/requests.jsonl",
                "events": "raw/events.jsonl",
                "logs": "logs/harness.log",
            },
        },
    )
    _write_jsonl(
        out_dir / "raw" / "requests.jsonl",
        [_sanitize_scenario_for_artifact(scenario) for scenario in scenarios],
    )
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


def _sanitize_scenario_for_artifact(scenario: Scenario) -> dict[str, Any]:
    payload = scenario.to_json()
    if isinstance(payload.get("payload"), dict):
        payload["payload"] = _sanitize_payload_value(payload["payload"])
    return payload


def _sanitize_payload_value(value: Any, *, key: str | None = None) -> Any:
    if key in SENSITIVE_AUDIO_REFERENCE_KEYS and isinstance(value, str):
        return _summarize_audio_reference(value)
    if isinstance(value, dict):
        return {
            item_key: _sanitize_payload_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_payload_value(item) for item in value]
    return value


def _summarize_audio_reference(value: str) -> dict[str, Any]:
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
    parsed = urlparse(value)
    summary: dict[str, Any] = {
        "redacted": True,
        "chars": len(value),
        "sha256_16": digest,
        "scheme": parsed.scheme or None,
    }
    if parsed.scheme in {"http", "https"}:
        netloc = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            netloc = f"{netloc}:{port}"
        summary["url"] = urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    elif parsed.scheme:
        summary["kind"] = f"{parsed.scheme}_uri"
    else:
        summary["kind"] = "inline_or_path"
    return summary
