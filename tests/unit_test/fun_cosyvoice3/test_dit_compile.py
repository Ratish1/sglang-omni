# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages


class FakeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FakeDiTEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], bool]] = []
        self.dtypes: list[tuple[torch.dtype, ...]] = []
        self.modes: list[tuple[bool, bool]] = []

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        self.calls.append((tuple(x.shape), tuple(mask.shape), streaming))
        self.dtypes.append(
            (x.dtype, mask.dtype, mu.dtype, t.dtype, spks.dtype, cond.dtype)
        )
        self.modes.append((torch.is_inference_mode_enabled(), torch.is_inference(x)))
        for block in self.transformer_blocks:
            x = block(x)
        return x


class FakeFlow(torch.nn.Module):
    def __init__(self, estimator) -> None:
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.estimator = estimator


class NonModuleEstimator:
    pass


def test_compile_dit_backbone_compiles_each_block_and_keeps_forward_eager(
    monkeypatch,
) -> None:
    estimator = FakeDiTEstimator()
    flow = FakeFlow(estimator)
    param_names = set(dict(estimator.named_parameters()))
    compile_dynamic: list[bool | None] = []

    def fake_compile(fn, dynamic=None):
        compile_dynamic.append(dynamic)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    stages.compile_dit_backbone(flow, warmup_mel_frames=16)

    assert compile_dynamic == [True, True]
    assert "forward" not in vars(estimator)
    assert set(dict(estimator.named_parameters())) == param_names
    # Warmup runs the CFG [2, 80, T] signature for buffered and causal hops.
    assert estimator.calls == (
        [((2, 80, 16), (2, 1, 16), False)] * 3 + [((2, 80, 16), (2, 1, 16), True)] * 3
    )


def test_compile_dit_backbone_warmup_matches_serving_grad_mode(monkeypatch) -> None:
    # flow.inference is @torch.inference_mode(); warmup must match (Dynamo
    # guards on grad mode) so the first request reuses the warmed graph.
    estimator = FakeDiTEstimator()
    monkeypatch.setattr(torch, "compile", lambda fn, dynamic=None: fn)

    stages.compile_dit_backbone(
        FakeFlow(estimator), warmup_mel_frames=16, warmup_steps=2
    )

    assert estimator.modes == [(True, True)] * 4


@pytest.mark.parametrize(
    ("autocast_dtype", "parameter_dtype", "expected_dtype"),
    [
        (torch.bfloat16, torch.float32, torch.bfloat16),
        (None, torch.float64, torch.float64),
    ],
)
def test_compile_dit_backbone_warmup_uses_serving_dtype(
    monkeypatch,
    autocast_dtype: torch.dtype | None,
    parameter_dtype: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    estimator = FakeDiTEstimator().to(parameter_dtype)
    monkeypatch.setattr(torch, "compile", lambda fn, dynamic=None: fn)

    stages.compile_dit_backbone(
        FakeFlow(estimator),
        autocast_dtype=autocast_dtype,
        warmup_steps=1,
        warmup_mel_frames=16,
    )

    assert estimator.dtypes == [(expected_dtype,) * 6] * 2


def test_compile_dit_backbone_rejects_non_module_estimator(monkeypatch) -> None:
    flow = FakeFlow(NonModuleEstimator())

    def fail_compile(fn, dynamic=None):
        raise AssertionError("torch.compile must not run for a non-module estimator")

    monkeypatch.setattr(torch, "compile", fail_compile)

    with pytest.raises(
        RuntimeError,
        match="requires a PyTorch estimator",
    ):
        stages.compile_dit_backbone(flow)


def test_compile_dit_backbone_compile_failure_fails_startup(monkeypatch) -> None:
    # torch.compile is lazy: a failure surfaces on the first warmup call.
    def fail_compile(fn, dynamic=None):
        def wrapped(*args, **kwargs):
            raise RuntimeError("synthetic compile failure")

        return wrapped

    monkeypatch.setattr(torch, "compile", fail_compile)

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        stages.compile_dit_backbone(FakeFlow(FakeDiTEstimator()), warmup_mel_frames=16)


def test_compile_dit_backbone_rejects_degenerate_warmup_length(monkeypatch) -> None:
    flow = FakeFlow(FakeDiTEstimator())

    def fail_compile(fn, dynamic=None):
        raise AssertionError("torch.compile must not run for invalid warmup")

    monkeypatch.setattr(torch, "compile", fail_compile)

    with pytest.raises(ValueError, match="warmup_mel_frames"):
        stages.compile_dit_backbone(flow, warmup_mel_frames=1)


def test_vocoder_factory_exposes_dit_torch_compile_flag() -> None:
    import inspect

    signature = inspect.signature(stages.create_vocoder_executor)
    assert signature.parameters["enable_dit_torch_compile"].default is True
    assert signature.parameters["enable_flow_cuda_graph"].default is True
    assert signature.parameters["enable_flow_estimator_trt"].default is False
