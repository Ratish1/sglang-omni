# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.hardware_backend.mlx import runtime as mlx_runtime
from sglang.srt.runtime_context import get_context

from sglang_omni.models.fun_cosyvoice3 import engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.scheduling.stage_kv_budget import (
    consume_stage_kv_cache_bytes,
    stage_kv_cache_budget,
)


def _enable_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: True)
    monkeypatch.setattr(
        engine_builder_module.current_platform,
        "is_mps",
        lambda: True,
    )


def _valid_mlx_server_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_running_requests=1,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        disable_overlap_schedule=True,
        enable_priority_scheduling=False,
        mlx_enable_sampling=True,
    )


def test_mlx_engine_profile_disables_incompatible_scheduler_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_mlx(monkeypatch)
    builder = FunCosyVoice3EngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["max_running_requests"] == 1
    assert defaults["disable_radix_cache"] is True
    assert defaults["disable_overlap_schedule"] is True
    assert defaults["chunked_prefill_size"] == -1
    assert defaults["mlx_enable_sampling"] is True
    assert builder.extra_scheduler_kwargs() == {
        "enable_async_decode": True,
        "async_decode_min_batch_size": 1,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_running_requests", 2, "max_running_requests=1"),
        ("disable_radix_cache", False, "disable_radix_cache=True"),
        ("chunked_prefill_size", 128, "chunked_prefill_size=-1"),
        ("disable_overlap_schedule", False, "disable_overlap_schedule=True"),
        ("enable_priority_scheduling", True, "priority preemption"),
        ("mlx_enable_sampling", False, "mlx_enable_sampling=True"),
    ],
)
def test_mlx_engine_rejects_unsafe_overrides(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    _enable_mlx(monkeypatch)
    server_args = _valid_mlx_server_args()
    setattr(server_args, field, value)

    with pytest.raises(ValueError, match=message):
        FunCosyVoice3EngineBuilder().validate_before_infrastructure(server_args)


def test_mlx_engine_passes_distinct_native_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_mlx(monkeypatch)
    builder = FunCosyVoice3EngineBuilder(
        mlx_model_path="mlx-org/model",
        mlx_model_revision="mlx-revision",
    )
    builder._checkpoint_root = "/official/model"

    assert builder.infra_kwargs() == {
        "mlx_model_path": "mlx-org/model",
        "mlx_model_revision": "mlx-revision",
    }


def test_torch_mps_uses_single_request_native_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "mps:0"

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["max_running_requests"] == 1

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=2))


def test_cuda_engine_leaves_the_memory_fraction_to_sglang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)

    defaults = FunCosyVoice3EngineBuilder().generation_defaults(dtype="bfloat16")

    assert "mem_fraction_static" not in defaults


@pytest.mark.parametrize(
    ("max_running_requests", "expected_max_total_tokens"),
    [(32, 131072), (8, 32768)],
)
def test_cuda_engine_caps_the_kv_pool_at_the_admission_bound(
    max_running_requests: int,
    expected_max_total_tokens: int,
) -> None:
    overrides = {"max_running_requests": max_running_requests}

    FunCosyVoice3EngineBuilder().adjust_overrides(overrides)

    assert overrides["max_total_tokens"] == expected_max_total_tokens


def test_cuda_engine_keeps_a_deployment_token_cap() -> None:
    overrides = {"max_running_requests": 32, "max_total_tokens": 4096}

    FunCosyVoice3EngineBuilder().adjust_overrides(overrides)

    assert overrides["max_total_tokens"] == 4096


def test_cuda_engine_adds_no_token_cap_under_a_stage_byte_budget() -> None:
    overrides = {"max_running_requests": 32}

    with stage_kv_cache_budget("tts_engine", 2 * 1024**3):
        FunCosyVoice3EngineBuilder().adjust_overrides(overrides)
        assert consume_stage_kv_cache_bytes() == 2 * 1024**3

    assert "max_total_tokens" not in overrides


@pytest.mark.parametrize(
    ("pool_tokens", "max_running_requests"),
    [(131072, 32), (854232, 8)],
)
def test_cuda_engine_reports_the_pool_against_the_admission_bound(
    monkeypatch: pytest.MonkeyPatch,
    pool_tokens: int,
    max_running_requests: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(mlx_runtime, "use_mlx", lambda: False)

    class FakePool:
        def get_kv_size_bytes(self) -> tuple[int, int]:
            return pool_tokens * 1024, pool_tokens * 1024

    outboxes: list[object] = []
    scheduler = SimpleNamespace(
        max_total_num_tokens=pool_tokens,
        outbox=object(),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(token_to_kv_pool=FakePool())
        ),
    )

    with (
        get_context().override_server_args(
            max_running_requests=max_running_requests,
            context_length=4096,
            mem_fraction_static=0.875,
        ),
        caplog.at_level(
            "INFO", logger="sglang_omni.models.fun_cosyvoice3.engine_builder"
        ),
    ):
        FunCosyVoice3EngineBuilder().post_scheduler_setup(
            scheduler,
            SimpleNamespace(set_stream_outbox=outboxes.append),
        )

    assert outboxes == [scheduler.outbox]
    assert caplog.messages == [
        f"Fun-CosyVoice3 KV pool holds {pool_tokens} tokens, "
        f"{pool_tokens * 2048 / 2**30:.2f} GiB, against a configured maximum "
        f"demand of {max_running_requests * 4096} "
        f"({max_running_requests} running x 4096 context), "
        "mem_fraction_static 0.875"
    ]


def test_mlx_engine_reports_no_kv_pool(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_mlx(monkeypatch)
    outboxes: list[object] = []
    scheduler = SimpleNamespace(outbox=object())

    with caplog.at_level(
        "INFO", logger="sglang_omni.models.fun_cosyvoice3.engine_builder"
    ):
        FunCosyVoice3EngineBuilder().post_scheduler_setup(
            scheduler,
            SimpleNamespace(set_stream_outbox=outboxes.append),
        )

    assert outboxes == [scheduler.outbox]
    assert caplog.messages == []
