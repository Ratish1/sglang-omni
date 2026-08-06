# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the breakable prefill CUDA graph policy and wiring."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from tests.unit_test.fakes import FakeServerArgs


def _server_args(
    *,
    prefill_backend: str = "disabled",
    prefill_bs: Any = None,
    prefill_max_bs: Any = None,
    locked: frozenset = frozenset(),
    chunked_prefill_size: int | None = 8192,
    max_prefill_tokens: int = 16384,
    disable_cuda_graph: bool = False,
) -> FakeServerArgs:
    return FakeServerArgs(
        max_running_requests=4,
        disable_cuda_graph=disable_cuda_graph,
        enable_torch_compile=False,
        torch_compile_max_bs=None,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        tp_size=1,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=4, bs=(1, 2, 4)),
            prefill=SimpleNamespace(
                backend=prefill_backend,
                bs=prefill_bs,
                max_bs=prefill_max_bs,
            ),
        ),
        _cuda_graph_config_locked=locked,
    )


_PREFILL_BS_LOCKED = frozenset({("prefill", "bs")})


def _validate(server_args: FakeServerArgs) -> None:
    validate_generation_batch_policy(
        model_name="Test TTS",
        server_args=server_args,
        model_buffer_bs=None,
    )


def test_prefill_policy_accepts_disabled_and_declared_breakable() -> None:
    # "disabled" must stay a silent no-op (every existing model ships with it)
    # even though the fake declares no buckets and holds no lock; a fully
    # declared breakable policy is the only other accepted shape.
    _validate(_server_args())
    _validate(
        _server_args(
            prefill_backend="breakable",
            prefill_bs=(128, 256, 512),
            prefill_max_bs=512,
            locked=_PREFILL_BS_LOCKED,
        )
    )


def test_breakable_requires_cuda_graphs_enabled() -> None:
    with pytest.raises(ValueError, match="require CUDA graphs enabled"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(128,),
                locked=_PREFILL_BS_LOCKED,
                disable_cuda_graph=True,
            )
        )


def test_non_breakable_prefill_backend_is_rejected() -> None:
    for backend in ("full", "tc_piecewise"):
        with pytest.raises(ValueError, match="must be 'breakable'"):
            _validate(
                _server_args(
                    prefill_backend=backend,
                    prefill_bs=(128,),
                    locked=_PREFILL_BS_LOCKED,
                )
            )


def test_breakable_requires_explicit_buckets() -> None:
    # sglang generates a ladder when the backend is enabled without buckets;
    # the omni policy refuses to treat that generated list as a declaration.
    with pytest.raises(ValueError, match="explicit"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(4, 8, 16),
                locked=frozenset(),
            )
        )


def test_breakable_rejects_non_increasing_buckets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(256, 128),
                locked=_PREFILL_BS_LOCKED,
            )
        )


def test_breakable_rejects_max_bs_mismatch() -> None:
    with pytest.raises(ValueError, match="cuda_graph_max_bs_prefill"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(128, 256),
                prefill_max_bs=512,
                locked=_PREFILL_BS_LOCKED,
            )
        )


def test_breakable_rejects_buckets_above_chunked_prefill() -> None:
    with pytest.raises(ValueError, match="chunked_prefill_size are unreachable"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(4096, 12288),
                locked=_PREFILL_BS_LOCKED,
                chunked_prefill_size=8192,
            )
        )


def test_breakable_rejects_buckets_above_max_prefill_tokens() -> None:
    # With chunking disabled (sglang normalizes that to -1), the batch
    # prefill-token budget still caps extend tokens per forward.
    with pytest.raises(ValueError, match="max_prefill_tokens are unreachable"):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(4096, 32768),
                locked=_PREFILL_BS_LOCKED,
                chunked_prefill_size=-1,
                max_prefill_tokens=16384,
            )
        )


def test_breakable_accepts_disabled_chunked_prefill() -> None:
    _validate(
        _server_args(
            prefill_backend="breakable",
            prefill_bs=(4096, 8192),
            locked=_PREFILL_BS_LOCKED,
            chunked_prefill_size=-1,
        )
    )


def test_breakable_warns_on_padding_factor_gaps(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(64, 512),
                locked=_PREFILL_BS_LOCKED,
            )
        )

    assert any("padding factor" in record.message for record in caplog.records)


def test_padding_gap_warning_requires_a_non_empty_eager_range(caplog) -> None:
    # nxt = 2 * prev + 1 leaves no eager length: every t > prev satisfies
    # nxt <= 2t and replays. The warning must not fire on an empty range;
    # the first real valley appears at nxt = 2 * prev + 3 (single length 65).
    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(64, 129),
                locked=_PREFILL_BS_LOCKED,
            )
        )
    assert not any("padding factor" in record.message for record in caplog.records)

    with caplog.at_level(logging.WARNING):
        _validate(
            _server_args(
                prefill_backend="breakable",
                prefill_bs=(64, 131),
                locked=_PREFILL_BS_LOCKED,
            )
        )
    assert any("[(65, 65)]" in record.getMessage() for record in caplog.records)


def test_overrides_derive_prefill_max_bs_from_buckets() -> None:
    overrides = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [128, 256],
        },
    )

    assert overrides["cuda_graph_backend_prefill"] == "breakable"
    assert overrides["cuda_graph_max_bs_prefill"] == 256

    explicit = build_generation_batch_overrides(
        max_running_requests=4,
        server_args_overrides={
            "cuda_graph_bs_prefill": [128, 256],
            "cuda_graph_max_bs_prefill": 256,
        },
    )

    assert explicit["cuda_graph_max_bs_prefill"] == 256


def test_prefill_graph_prereqs_reject_non_cuda_and_tp() -> None:
    from sglang_omni.scheduling.engine_factory import _validate_prefill_graph_prereqs

    with pytest.raises(RuntimeError, match="CUDA device"):
        _validate_prefill_graph_prereqs(_server_args(), "mps")
    with pytest.raises(RuntimeError, match="tp_size == 1"):
        tp2 = _server_args()
        tp2.tp_size = 2
        _validate_prefill_graph_prereqs(tp2, "cuda:0")
    _validate_prefill_graph_prereqs(_server_args(), "cuda:0")


def test_builder_wires_payload_slot_and_attestation(monkeypatch) -> None:
    """A breakable prefill policy must reach ModelWorker (input_embeds slot)
    and run the post-capture attestation; a disabled policy must do neither."""
    from sglang_omni.scheduling import bootstrap, sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
    from sglang_omni.utils import cuda_graph_batch_validator

    infra_kwargs_seen: list[dict[str, Any]] = []
    attest_calls: list[tuple[Any, Any]] = []

    def fake_build_sglang_server_args(checkpoint_dir, *, context_length, **overrides):
        del checkpoint_dir, context_length
        prefill_backend = overrides.get("cuda_graph_backend_prefill", "disabled")
        prefill_bs = overrides.get("cuda_graph_bs_prefill")
        return _server_args(
            prefill_backend=prefill_backend,
            prefill_bs=prefill_bs,
            prefill_max_bs=overrides.get("cuda_graph_max_bs_prefill"),
            locked=_PREFILL_BS_LOCKED if prefill_bs else frozenset(),
        )

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del server_args, gpu_id
        infra_kwargs_seen.append(dict(kwargs))
        model_runner = SimpleNamespace(
            model=SimpleNamespace(),
            init_cuda_graphs=lambda: None,
        )
        return (
            SimpleNamespace(model_runner=model_runner),
            "tree_cache",
            "req_pool",
            "kv_pool",
            "prefill",
            "decode",
            "model_config",
        )

    def fake_attest(model_runner, server_args) -> None:
        attest_calls.append((model_runner, server_args))

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        bootstrap, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(
        sglang_backend, "SGLangOutputProcessor", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        cuda_graph_batch_validator, "attest_prefill_cuda_graphs", fake_attest
    )

    class PolicyBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123
        supports_breakable_prefill_cuda_graph = True

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            del dtype
            return {"max_running_requests": 4}

        def setup_model(self, **kwargs: Any) -> None:
            del kwargs

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del output_proc
            return model_worker

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            return object(), object()

        def make_scheduler(self, **kwargs: Any) -> Any:
            return SimpleNamespace(outbox=None, kwargs=kwargs)

    PolicyBuilder().build(
        "model",
        server_args_overrides={
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_bs_prefill": [128, 256],
        },
    )

    assert infra_kwargs_seen[-1]["enable_prefill_input_embeds"] is True
    assert len(attest_calls) == 1

    PolicyBuilder().build("model")

    assert "enable_prefill_input_embeds" not in infra_kwargs_seen[-1]
    assert len(attest_calls) == 1


def test_builder_rejects_breakable_without_model_opt_in(monkeypatch) -> None:
    """A deployment override must not enable prefill graphs on a model whose
    builder never adopted the contract: the flag would flip is_multimodal on
    an unaudited model and capture graphs its forward never replays."""
    from sglang_omni.scheduling import sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    def fake_build_sglang_server_args(checkpoint_dir, *, context_length, **overrides):
        del checkpoint_dir, context_length
        return _server_args(
            prefill_backend="breakable",
            prefill_bs=overrides.get("cuda_graph_bs_prefill"),
            locked=_PREFILL_BS_LOCKED,
        )

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_sglang_server_args
    )

    class NonAdoptingBuilder(TtsEngineBuilder):
        model_name = "Test TTS"
        context_length = 123

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            del dtype
            return {"max_running_requests": 4}

        def setup_model(self, **kwargs: Any) -> None:
            del kwargs

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del output_proc
            return model_worker

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            return object(), object()

    with pytest.raises(RuntimeError, match="has not adopted the breakable prefill"):
        NonAdoptingBuilder().build(
            "model",
            server_args_overrides={
                "cuda_graph_backend_prefill": "breakable",
                "cuda_graph_bs_prefill": [128, 256],
            },
        )
