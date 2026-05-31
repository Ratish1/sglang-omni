# SPDX-License-Identifier: Apache-2.0
"""Spec parsing for the TTS serving benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_TEST_TYPES = {"engine", "e2e", "external"}
VALID_PROFILES = {"ci", "production", "stress"}
VALID_LOAD_MODES = {"closed_loop", "open_loop", "ramp", "burst", "soak"}
VALID_ARRIVAL_DISTRIBUTIONS = {"deterministic", "poisson"}
DEFAULT_ENDPOINTS = ("speech", "speech_sse", "voices", "batch", "websocket")


class SpecError(ValueError):
    """Raised when /etc/benchmark/spec.json is invalid."""


@dataclass(frozen=True)
class AuthSpec:
    api_key_env: str | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> AuthSpec:
        if obj is None:
            return cls()
        if not isinstance(obj, dict):
            raise SpecError("auth must be an object when provided")
        api_key_env = obj.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise SpecError("auth.api_key_env must be a string")
        return cls(api_key_env=api_key_env)


@dataclass(frozen=True)
class LoadStage:
    id: str
    mode: str
    request_count: int
    max_concurrency: int
    request_rate: float = float("inf")
    start_request_rate: float | None = None
    duration_s: float | None = None
    arrival_distribution: str = "deterministic"

    @classmethod
    def from_obj(cls, obj: Any, *, index: int) -> LoadStage:
        if not isinstance(obj, dict):
            raise SpecError("params.load_stages entries must be objects")
        stage_id = _str_value(obj, "id", f"stage-{index + 1}")
        mode = _str_value(obj, "mode", "closed_loop")
        if mode not in VALID_LOAD_MODES:
            raise SpecError(
                f"params.load_stages[].mode must be one of {sorted(VALID_LOAD_MODES)}"
            )

        request_count = _positive_int(
            obj,
            "request_count",
            _positive_int(obj, "total_requests", 100),
        )
        max_concurrency = _positive_int(
            obj,
            "max_concurrency",
            _positive_int(obj, "concurrency", 8),
        )
        request_rate = _request_rate(obj.get("request_rate", float("inf")))
        start_request_rate = _optional_request_rate(obj.get("start_request_rate"))
        duration_s = _optional_positive_float(obj.get("duration_s"), "duration_s")
        arrival_distribution = _str_value(obj, "arrival_distribution", "deterministic")
        if arrival_distribution not in VALID_ARRIVAL_DISTRIBUTIONS:
            raise SpecError(
                "params.load_stages[].arrival_distribution must be one of "
                f"{sorted(VALID_ARRIVAL_DISTRIBUTIONS)}"
            )
        if mode in {"open_loop", "ramp", "soak"} and request_rate == float("inf"):
            raise SpecError(
                "params.load_stages[].request_rate must be finite for " f"{mode} stages"
            )
        if mode == "ramp" and start_request_rate is None:
            raise SpecError(
                "params.load_stages[].start_request_rate is required for ramp stages"
            )
        return cls(
            id=stage_id,
            mode=mode,
            request_count=request_count,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
            start_request_rate=start_request_rate,
            duration_s=duration_s,
            arrival_distribution=arrival_distribution,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "request_count": self.request_count,
            "max_concurrency": self.max_concurrency,
            "request_rate": (
                "inf" if self.request_rate == float("inf") else self.request_rate
            ),
            "start_request_rate": self.start_request_rate,
            "duration_s": self.duration_s,
            "arrival_distribution": self.arrival_distribution,
        }


@dataclass(frozen=True)
class BenchmarkParams:
    profile: str = "ci"
    total_requests: int = 100
    max_concurrency: int = 8
    concurrency_levels: tuple[int, ...] | None = None
    load_stages: tuple[LoadStage, ...] = field(default_factory=tuple)
    request_rate: float = float("inf")
    timeout_s: int = 120
    enabled_endpoints: tuple[str, ...] = DEFAULT_ENDPOINTS
    allow_missing_optional_endpoints: bool = True
    seedtts_ref_audio: str | None = None
    seedtts_ref_text: str | None = None
    provider_label: str | None = None
    implementation_label: str | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> BenchmarkParams:
        if obj is None:
            obj = {}
        if not isinstance(obj, dict):
            raise SpecError("params must be an object when provided")

        profile = _str_value(obj, "profile", cls.profile)
        if profile not in VALID_PROFILES:
            raise SpecError(f"params.profile must be one of {sorted(VALID_PROFILES)}")

        total_requests = _positive_int(obj, "total_requests", cls.total_requests)
        max_concurrency = _positive_int(obj, "max_concurrency", cls.max_concurrency)
        concurrency_levels = _concurrency_levels(obj.get("concurrency_levels"))
        timeout_s = _positive_int(obj, "timeout_s", cls.timeout_s)
        request_rate = _request_rate(obj.get("request_rate", cls.request_rate))
        load_stages = _load_stages(
            obj.get("load_stages"),
            total_requests=total_requests,
            max_concurrency=max_concurrency,
            concurrency_levels=concurrency_levels,
            request_rate=request_rate,
        )

        enabled = obj.get("enabled_endpoints", list(DEFAULT_ENDPOINTS))
        if not isinstance(enabled, list) or not all(
            isinstance(item, str) for item in enabled
        ):
            raise SpecError("params.enabled_endpoints must be a list of strings")
        unknown = sorted(set(enabled) - set(DEFAULT_ENDPOINTS))
        if unknown:
            raise SpecError(f"unknown enabled_endpoints: {unknown}")

        return cls(
            profile=profile,
            total_requests=total_requests,
            max_concurrency=(
                max(stage.max_concurrency for stage in load_stages)
                if load_stages
                else max_concurrency
            ),
            concurrency_levels=concurrency_levels,
            load_stages=load_stages,
            request_rate=request_rate,
            timeout_s=timeout_s,
            enabled_endpoints=tuple(enabled),
            allow_missing_optional_endpoints=_bool_value(
                obj,
                "allow_missing_optional_endpoints",
                cls.allow_missing_optional_endpoints,
            ),
            seedtts_ref_audio=_optional_str(obj, "seedtts_ref_audio"),
            seedtts_ref_text=_optional_str(obj, "seedtts_ref_text"),
            provider_label=_optional_str(obj, "provider_label"),
            implementation_label=_optional_str(obj, "implementation_label"),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    base_url: str
    model_name: str
    test_type: str = "engine"
    run_id: str | None = None
    seed: int = 601
    auth: AuthSpec = field(default_factory=AuthSpec)
    params: BenchmarkParams = field(default_factory=BenchmarkParams)

    @classmethod
    def from_obj(cls, obj: Any) -> BenchmarkSpec:
        if not isinstance(obj, dict):
            raise SpecError("spec must be a JSON object")
        base_url = _required_str(obj, "base_url").rstrip("/")
        model_name = _required_str(obj, "model_name")
        test_type = _str_value(obj, "test_type", "engine")
        if test_type not in VALID_TEST_TYPES:
            raise SpecError(f"test_type must be one of {sorted(VALID_TEST_TYPES)}")
        seed = obj.get("seed", 601)
        if not isinstance(seed, int):
            raise SpecError("seed must be an integer")
        run_id = _optional_str(obj, "run_id")
        return cls(
            base_url=base_url,
            model_name=model_name,
            test_type=test_type,
            run_id=run_id,
            seed=seed,
            auth=AuthSpec.from_obj(obj.get("auth")),
            params=BenchmarkParams.from_obj(obj.get("params")),
        )


def load_spec(path: str | Path) -> BenchmarkSpec:
    spec_path = Path(path)
    if not spec_path.is_file():
        raise SpecError(f"spec file not found: {spec_path}")
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON in spec file: {exc}") from exc
    return BenchmarkSpec.from_obj(raw)


def _required_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{key} must be a non-empty string")
    return value


def _optional_str(obj: dict[str, Any], key: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{key} must be a non-empty string when provided")
    return value


def _str_value(obj: dict[str, Any], key: str, default: str) -> str:
    value = obj.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{key} must be a non-empty string")
    return value


def _positive_int(obj: dict[str, Any], key: str, default: int) -> int:
    value = obj.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecError(f"params.{key} must be a positive integer")
    return value


def _bool_value(obj: dict[str, Any], key: str, default: bool) -> bool:
    value = obj.get(key, default)
    if not isinstance(value, bool):
        raise SpecError(f"params.{key} must be a boolean")
    return value


def _request_rate(value: Any) -> float:
    if value == "inf":
        return float("inf")
    if not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0:
        return float(value)
    raise SpecError("params.request_rate must be a positive number or 'inf'")


def _optional_request_rate(value: Any) -> float | None:
    if value is None:
        return None
    rate = _request_rate(value)
    if rate == float("inf"):
        raise SpecError("params.load_stages[].start_request_rate must be finite")
    return rate


def _optional_positive_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SpecError(f"params.load_stages[].{key} must be a positive number")
    return float(value)


def _concurrency_levels(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise SpecError("params.concurrency_levels must be a non-empty list")
    levels: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise SpecError("params.concurrency_levels must contain positive integers")
        if item not in levels:
            levels.append(item)
    return tuple(levels)


def _load_stages(
    value: Any,
    *,
    total_requests: int,
    max_concurrency: int,
    concurrency_levels: tuple[int, ...] | None,
    request_rate: float,
) -> tuple[LoadStage, ...]:
    if value is not None:
        if not isinstance(value, list) or not value:
            raise SpecError("params.load_stages must be a non-empty list")
        stages = tuple(
            LoadStage.from_obj(item, index=index) for index, item in enumerate(value)
        )
        _validate_unique_stage_ids(stages)
        return stages

    levels = concurrency_levels or (max_concurrency,)
    return tuple(
        LoadStage(
            id=f"c{level}",
            mode="closed_loop",
            request_count=total_requests,
            max_concurrency=level,
            request_rate=request_rate,
        )
        for level in levels
    )


def _validate_unique_stage_ids(stages: tuple[LoadStage, ...]) -> None:
    seen: set[str] = set()
    for stage in stages:
        if stage.id in seen:
            raise SpecError(f"duplicate params.load_stages[].id: {stage.id}")
        seen.add(stage.id)
