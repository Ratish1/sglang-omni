# SPDX-License-Identifier: Apache-2.0
"""Base-runner _install_sampling_seeds: wire a request seed to the sampler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_model_runner import (
    register_flashinfer_seeded_sampler,
)
from sglang_omni.sampling.seed import derive_sampling_seed

VOCAB_SIZE = 3072
TOP_K = 50


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_review_unseeded_seeded_unseeded_batches_use_the_expected_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sglang.srt.layers.sampler as sampler_module

    register_flashinfer_seeded_sampler()
    flashinfer_draw = sampler_module.top_k_top_p_sampling_from_probs
    flashinfer_calls: list[torch.dtype] = []

    def traced_flashinfer_draw(
        probabilities: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        *,
        filter_apply_order: str,
    ) -> torch.Tensor:
        flashinfer_calls.append(probabilities.dtype)
        return flashinfer_draw(
            probabilities, top_ks, top_ps, filter_apply_order=filter_apply_order
        )

    monkeypatch.setattr(
        sampler_module, "top_k_top_p_sampling_from_probs", traced_flashinfer_draw
    )
    logits = torch.full((2, VOCAB_SIZE), -torch.inf, device="cuda")
    logits[0, [7, 9]] = torch.tensor([1.0, 0.0], device="cuda")
    logits[1, [11, 13]] = torch.tensor([0.0, 1.0], device="cuda")
    positions = torch.tensor([17, 23], device="cuda")
    sampling_info = sampling_info_for([11, 22])

    sampling_info.sampling_seed = None
    unseeded_tokens, _ = draw("flashinfer", logits, sampling_info, positions)
    assert flashinfer_calls == [torch.float32]
    assert unseeded_tokens[0] in (7, 9) and unseeded_tokens[1] in (11, 13)

    sampling_info.sampling_seed = torch.tensor([11, 22], device="cuda")
    seeded_tokens, _ = draw("flashinfer", logits, sampling_info, positions)
    reference_tokens, _ = draw("pytorch", logits, sampling_info, positions)
    assert seeded_tokens == reference_tokens
    assert flashinfer_calls == [torch.float32]

    sampling_info.sampling_seed = None
    unseeded_tokens, _ = draw("flashinfer", logits, sampling_info, positions)
    assert flashinfer_calls == [torch.float32, torch.float32]
    assert unseeded_tokens[0] in (7, 9) and unseeded_tokens[1] in (11, 13)


def req(seed, request_id="req"):
    sp = SimpleNamespace(sampling_seed=seed)
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(req=SimpleNamespace(sampling_params=sp)),
    )


def make_fb(sampling_seed=None, *, top_p=False, top_k=False, min_p=False):
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            device="cpu",
            sampling_seed=sampling_seed,
            need_top_p_sampling=top_p,
            need_top_k_sampling=top_k,
            need_min_p_sampling=min_p,
        )
    )


def test_installs_per_row_seeds_and_noops_without_one():
    runner = object.__new__(ModelRunner)
    # seeded rows -> per-row int64 seed tensor; mixed unseeded rows get a
    # rank-shared fallback derived from the request id.
    fb = make_fb()
    requests = [req(42, "seeded"), req(None, "unseeded")]
    runner.install_sampling_seeds(fb, requests)
    ss = fb.sampling_info.sampling_seed
    assert isinstance(ss, torch.Tensor) and ss.dtype == torch.long
    assert int(ss[0]) == 42 and ss.shape == (2,)
    assert int(ss[1]) == derive_sampling_seed("sglang-omni-unseeded-row", "unseeded")
    assert requests[1].data.req.sampling_params.sampling_seed is None
    # no seed anywhere -> left unseeded (preserves random sampling)
    fb2 = make_fb()
    runner.install_sampling_seeds(fb2, [req(None), req(None)])
    assert fb2.sampling_info.sampling_seed is None


def test_does_not_clobber_subclass_installed_seed():
    runner = object.__new__(ModelRunner)
    preset = torch.tensor([1, 2, 3])
    fb = make_fb(sampling_seed=preset)
    runner.install_sampling_seeds(fb, [req(42), req(42), req(42)])
    assert fb.sampling_info.sampling_seed is preset


def test_preinstalled_seed_requires_sampling_mode_contract():
    runner = object.__new__(ModelRunner)
    preset = torch.tensor([1, 2])
    fb = SimpleNamespace(sampling_info=SimpleNamespace(sampling_seed=preset))
    with pytest.raises(AttributeError):
        runner.install_sampling_seeds(fb, [req(42), req(42)])


def test_unseeded_row_in_seeded_batch_uses_rank_shared_fallback():
    runner = object.__new__(ModelRunner)
    requests = [req(42, "seeded"), req(None, "unseeded")]
    fb = make_fb()
    runner.install_sampling_seeds(fb, requests)
    fallback = int(fb.sampling_info.sampling_seed[1])
    assert requests[1].data.req.sampling_params.sampling_seed is None

    fb_next = make_fb()
    runner.install_sampling_seeds(fb_next, requests)
    assert int(fb_next.sampling_info.sampling_seed[1]) == fallback
    assert requests[1].data.req.sampling_params.sampling_seed is None


def test_rejects_seeded_min_p_before_upstream_sampler():
    runner = object.__new__(ModelRunner)
    with pytest.raises(ValueError, match="min_p"):
        runner.install_sampling_seeds(make_fb(min_p=True), [req(42)])


def test_installs_seeded_top_k_and_top_p_on_flashinfer():
    runner = object.__new__(ModelRunner)
    fb = make_fb(top_p=True, top_k=True)
    with get_context().override_server_args(sampling_backend="flashinfer"):
        runner.install_sampling_seeds(fb, [req(42)])
    assert int(fb.sampling_info.sampling_seed[0]) == 42


def sampling_info_for(
    sampling_seeds: list[int], sampling_mask_rows: list[int] | None = None
) -> SamplingBatchInfo:
    rows = len(sampling_seeds)
    return SamplingBatchInfo(
        temperatures=torch.full((rows, 1), 0.9, device="cuda"),
        top_ps=torch.ones(rows, device="cuda"),
        top_ks=torch.full((rows,), TOP_K, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(rows, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        sampling_seed=torch.tensor(sampling_seeds, dtype=torch.long, device="cuda"),
        sampling_mask_batch_indices=(
            None
            if sampling_mask_rows is None
            else torch.tensor(sampling_mask_rows, dtype=torch.long, device="cuda")
        ),
        device="cuda",
    )


def draw(
    sampling_backend: str,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    positions: torch.Tensor,
) -> tuple[list[int], LogitsProcessorOutput]:
    rows = logits.shape[0]
    logits_output = LogitsProcessorOutput(next_token_logits=logits.clone())
    with (
        get_context().override_server_args(sampling_backend=sampling_backend),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
    ):
        token_ids = create_sampler()(
            logits_output,
            sampling_info,
            False,
            [0] * rows,
            [[] for _ in range(rows)],
            positions,
        )
    return token_ids.view(-1).tolist(), logits_output


def served_logits(rows: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(0)
    return (torch.randn(rows, VOCAB_SIZE, generator=generator) * 4).to(
        "cuda", torch.bfloat16
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_rows_draw_the_pytorch_backend_tokens_in_any_order():
    register_flashinfer_seeded_sampler()
    logits = served_logits(4)
    seeds = [11, 22, 33, 44]
    positions = torch.tensor([3, 17, 5, 40], device="cuda")
    reversed_rows = torch.tensor([3, 2, 1, 0], device="cuda")

    pytorch_tokens, _ = draw("pytorch", logits, sampling_info_for(seeds), positions)
    flashinfer_tokens, _ = draw(
        "flashinfer", logits, sampling_info_for(seeds), positions
    )
    reversed_tokens, _ = draw(
        "flashinfer",
        logits[reversed_rows],
        sampling_info_for(seeds[::-1]),
        positions[reversed_rows],
    )

    assert flashinfer_tokens == pytorch_tokens
    assert reversed_tokens == pytorch_tokens[::-1]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_rows_return_the_pytorch_backend_sampling_mask():
    """A seeded FlashInfer batch that asks for the sampling mask gets the PyTorch
    backend's mask."""
    register_flashinfer_seeded_sampler()
    logits = served_logits(3)
    seeds = [11, 22, 33]
    positions = torch.tensor([3, 17, 5], device="cuda")

    pytorch_tokens, pytorch_output = draw(
        "pytorch", logits, sampling_info_for(seeds, [0, 2]), positions
    )
    flashinfer_tokens, flashinfer_output = draw(
        "flashinfer", logits, sampling_info_for(seeds, [0, 2]), positions
    )

    assert flashinfer_tokens == pytorch_tokens
    for field in ("token_ids", "lengths", "selected_logprobs", "statuses"):
        assert torch.equal(
            getattr(flashinfer_output.sampling_mask_output, field),
            getattr(pytorch_output.sampling_mask_output, field),
        )
