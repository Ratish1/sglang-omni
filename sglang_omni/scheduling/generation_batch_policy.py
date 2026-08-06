# SPDX-License-Identifier: Apache-2.0
"""Generation-stage batch policy helpers for SGLang-backed stages."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

_MISSING = object()

# Mirrors sglang's _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR: a prefill replay is
# rejected (falls back to eager) when the padded bucket exceeds twice the real
# token count, so bucket gaps beyond 2x create silent eager valleys.
_PREFILL_PADDING_FACTOR = 2


def get_decode_cuda_graph_max_bs(server_args: Any) -> Any:
    """Read the resolved SGLang decode CUDA Graph batch cap."""
    return server_args.cuda_graph_config.decode.max_bs


def get_decode_cuda_graph_bs(server_args: Any) -> Any:
    """Read the resolved SGLang decode CUDA Graph batch buckets."""
    return server_args.cuda_graph_config.decode.bs


def get_prefill_cuda_graph_backend(server_args: Any) -> str:
    """Read the resolved SGLang prefill CUDA graph backend."""
    return server_args.cuda_graph_config.prefill.backend


def build_default_cuda_graph_bs(max_bs: int) -> list[int]:
    max_bs = int(max_bs)
    if max_bs < 1:
        raise ValueError("max_bs must be >= 1")

    values = [1, 2, 4, 8, 12]
    values.extend(range(16, 257, 8))
    values.extend(range(272, 512, 16))
    values.extend(range(512, max_bs + 1, 32))
    values = [bs for bs in values if bs <= max_bs]
    if not values or values[-1] != max_bs:
        values.append(max_bs)
    return values


def build_generation_batch_overrides(
    *,
    max_running_requests: int,
    cuda_graph_max_bs: int | None = None,
    torch_compile_max_bs: int | None = None,
    server_args_overrides: Mapping[str, Any] | None = None,
    **stage_defaults: Any,
) -> dict[str, Any]:
    incoming = dict(server_args_overrides or {})
    max_running_requests = _normalize_positive_int(
        "max_running_requests",
        incoming.pop("max_running_requests", max_running_requests),
    )
    cuda_graph_max_bs = (
        max_running_requests if cuda_graph_max_bs is None else cuda_graph_max_bs
    )
    cuda_graph_max_bs = _normalize_positive_int(
        "cuda_graph_max_bs",
        incoming.pop("cuda_graph_max_bs", cuda_graph_max_bs),
    )
    torch_compile_max_bs = (
        max_running_requests if torch_compile_max_bs is None else torch_compile_max_bs
    )
    torch_compile_max_bs = _normalize_positive_int(
        "torch_compile_max_bs",
        incoming.pop("torch_compile_max_bs", torch_compile_max_bs),
    )
    cuda_graph_bs = incoming.pop("cuda_graph_bs", _MISSING)

    overrides = {
        **stage_defaults,
        **incoming,
        "max_running_requests": max_running_requests,
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "torch_compile_max_bs": torch_compile_max_bs,
    }
    if cuda_graph_bs is _MISSING:
        overrides["cuda_graph_bs"] = build_default_cuda_graph_bs(cuda_graph_max_bs)
    else:
        overrides["cuda_graph_bs"] = cuda_graph_bs

    # Prefill graph buckets are declared explicitly (no generated default);
    # derive the cap when only the bucket list is given so the resolved
    # sglang config stays coherent (max(bs) == max_bs).
    prefill_bs = overrides.get("cuda_graph_bs_prefill")
    if prefill_bs and "cuda_graph_max_bs_prefill" not in overrides:
        overrides["cuda_graph_max_bs_prefill"] = max(int(b) for b in prefill_bs)

    return overrides


def validate_generation_batch_policy(
    *,
    model_name: str,
    server_args: Any,
    model_buffer_bs: int | None = None,
) -> None:
    errors: list[str] = []

    max_running_requests = _validate_positive_int(
        "max_running_requests",
        server_args.max_running_requests,
        errors,
    )
    cuda_graph_enabled = not bool(server_args.disable_cuda_graph)

    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: tuple[int, ...] | None = None
    if cuda_graph_enabled:
        cuda_graph_max_bs = _validate_positive_int(
            "cuda_graph_max_bs",
            get_decode_cuda_graph_max_bs(server_args),
            errors,
            required=True,
        )
        cuda_graph_bs_value = get_decode_cuda_graph_bs(server_args)
        if cuda_graph_bs_value is None:
            errors.append("cuda_graph_bs must be explicit when CUDA graph is enabled")
        else:
            cuda_graph_bs = _normalize_cuda_graph_bs(cuda_graph_bs_value, errors)

        if cuda_graph_max_bs is not None and cuda_graph_bs is not None:
            if max(cuda_graph_bs) != cuda_graph_max_bs:
                errors.append(
                    "max(cuda_graph_bs) must match cuda_graph_max_bs "
                    f"({max(cuda_graph_bs)} != {cuda_graph_max_bs})"
                )

        if (
            max_running_requests is not None
            and cuda_graph_max_bs is not None
            and cuda_graph_max_bs < max_running_requests
        ):
            errors.append(
                "cuda_graph_max_bs must cover max_running_requests "
                f"({cuda_graph_max_bs} < {max_running_requests})"
            )

    _validate_prefill_graph_policy(server_args, cuda_graph_enabled, errors)

    torch_compile_enabled = bool(server_args.enable_torch_compile)
    torch_compile_max_bs = _validate_positive_int(
        "torch_compile_max_bs",
        server_args.torch_compile_max_bs,
        errors,
        required=torch_compile_enabled,
    )
    if (
        torch_compile_enabled
        and max_running_requests is not None
        and torch_compile_max_bs is not None
        and torch_compile_max_bs < max_running_requests
    ):
        errors.append(
            "torch_compile_max_bs must cover max_running_requests "
            f"({torch_compile_max_bs} < {max_running_requests})"
        )

    normalized_model_buffer_bs: int | None = None
    if model_buffer_bs is not None:
        normalized_model_buffer_bs = int(model_buffer_bs)
        if normalized_model_buffer_bs < 1:
            errors.append("model_buffer_bs must be >= 1")
        if (
            max_running_requests is not None
            and normalized_model_buffer_bs < max_running_requests
        ):
            errors.append(
                "model_buffer_bs must cover max_running_requests "
                f"({normalized_model_buffer_bs} < {max_running_requests})"
            )

    if errors:
        raise ValueError(
            f"{model_name} invalid generation batch policy: " + "; ".join(errors)
        )


def _validate_prefill_graph_policy(
    server_args: Any,
    cuda_graph_enabled: bool,
    errors: list[str],
) -> None:
    """Validate the declared prefill CUDA graph policy (R5).

    Only the breakable backend is supported: the omni prefill contract
    (payload channel, eager tail, input_embeds slot) is validated for BCG
    only. Buckets must be declared explicitly so the shape policy is a
    conscious per-model decision rather than sglang's generated ladder.
    """
    backend = get_prefill_cuda_graph_backend(server_args)
    if backend == "disabled":
        return

    if not cuda_graph_enabled:
        errors.append(
            "prefill CUDA graphs require CUDA graphs enabled "
            f"(backend={backend!r} with disable_cuda_graph)"
        )
        return
    if backend != "breakable":
        errors.append(
            "prefill CUDA graph backend must be 'breakable' or 'disabled', "
            f"got {backend!r}"
        )
        return

    if ("prefill", "bs") not in server_args._cuda_graph_config_locked:
        errors.append(
            "breakable prefill CUDA graphs require explicit "
            "cuda_graph_bs_prefill buckets (sglang's generated ladder is "
            "not an accepted shape policy)"
        )
        return

    prefill_cfg = server_args.cuda_graph_config.prefill
    buckets = _normalize_cuda_graph_bs(
        prefill_cfg.bs, errors, field="cuda_graph_bs_prefill"
    )
    if buckets is None:
        return

    max_bs = prefill_cfg.max_bs
    if max_bs is not None and max(buckets) != int(max_bs):
        errors.append(
            "max(cuda_graph_bs_prefill) must match cuda_graph_max_bs_prefill "
            f"({max(buckets)} != {max_bs})"
        )

    # Extend tokens per forward are capped by both the chunk budget and the
    # batch prefill-token budget; buckets above either cap never replay.
    for cap_name, cap_value in (
        ("chunked_prefill_size", server_args.chunked_prefill_size),
        ("max_prefill_tokens", server_args.max_prefill_tokens),
    ):
        if (
            cap_value is not None
            and int(cap_value) > 0
            and max(buckets) > int(cap_value)
        ):
            errors.append(
                f"cuda_graph_bs_prefill buckets above {cap_name} are "
                f"unreachable ({max(buckets)} > {cap_value})"
            )

    # A raw token count t replays at the smallest bucket >= t and falls back
    # to eager when that bucket exceeds t * factor, so a gap beyond the
    # factor leaves prompt lengths that never replay a graph. The largest
    # eager length under bucket nxt is (nxt - 1) // factor; a valley exists
    # only when that reaches past the previous bucket.
    valleys = []
    for prev, nxt in zip(buckets, buckets[1:]):
        eager_end = (nxt - 1) // _PREFILL_PADDING_FACTOR
        if eager_end > prev:
            valleys.append((prev + 1, eager_end))
    if valleys:
        logger.warning(
            "prefill CUDA graph bucket gaps exceed the %dx padding factor; "
            "prompt lengths inside %s fall back to eager",
            _PREFILL_PADDING_FACTOR,
            valleys,
        )


def _validate_positive_int(
    field: str,
    value: Any,
    errors: list[str],
    *,
    required: bool = True,
) -> int | None:
    if value is None:
        if required:
            errors.append(f"{field} must be explicit")
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be an integer")
        return None
    if normalized < 1:
        errors.append(f"{field} must be >= 1")
        return None
    return normalized


def _normalize_positive_int(field: str, value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if normalized < 1:
        raise ValueError(f"{field} must be >= 1")
    return normalized


def _normalize_cuda_graph_bs(
    value: Iterable[Any],
    errors: list[str],
    *,
    field: str = "cuda_graph_bs",
) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)):
        errors.append(f"{field} must be a sequence of positive integers")
        return None

    try:
        normalized = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be a sequence of positive integers")
        return None

    if not normalized:
        errors.append(f"{field} must be non-empty")
        return None
    if any(item < 1 for item in normalized):
        errors.append(f"{field} values must be >= 1")
        return None
    if tuple(sorted(set(normalized))) != normalized:
        errors.append(f"{field} must be strictly increasing")
        return None
    return normalized
