# SPDX-License-Identifier: Apache-2.0
"""Differential tests: vectorized M-RoPE positions vs sglang HF port oracle."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    get_rope_index_qwen3_omni,
)

from sglang_omni.models.qwen3_omni.mrope_positions import (
    feat_extract_output_lengths,
    get_rope_index_qwen3_omni_vectorized,
)

# Defaults from Qwen3OmniMoeThinkerConfig
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
AUDIO_TOKEN_ID = 151646
AUDIO_START_TOKEN_ID = 151647
AUDIO_END_TOKEN_ID = 151648
POSITION_ID_PER_SECONDS = 25
SPATIAL_MERGE_SIZE = 2


def oracle_and_fast(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.Tensor | None = None,
    use_audio_in_video: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    kwargs: dict[str, Any] = {
        "audio_token_id": AUDIO_TOKEN_ID,
        "audio_start_token_id": AUDIO_START_TOKEN_ID,
        "position_id_per_seconds": POSITION_ID_PER_SECONDS,
        "use_audio_in_video": use_audio_in_video,
        "audio_seqlens": audio_seqlens,
    }
    common = dict(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        **kwargs,
    )
    return get_rope_index_qwen3_omni(**common), get_rope_index_qwen3_omni_vectorized(
        **common
    )


def assert_bit_identical(
    oracle: tuple[torch.Tensor, torch.Tensor],
    fast: tuple[torch.Tensor, torch.Tensor],
) -> None:
    o_pos, o_delta = oracle
    f_pos, f_delta = fast
    assert o_pos.shape == f_pos.shape, (o_pos.shape, f_pos.shape)
    assert torch.equal(o_pos.float(), f_pos.float()), (
        f"position mismatch: max abs diff="
        f"{(o_pos.float() - f_pos.float()).abs().max().item()}"
    )
    assert o_delta.shape == f_delta.shape, (o_delta.shape, f_delta.shape)
    assert torch.equal(
        o_delta.float(), f_delta.float()
    ), f"delta mismatch: oracle={o_delta} fast={f_delta}"


def image_span(grid_thw: list[int]) -> list[int]:
    t, h, w = grid_thw
    image_len = (t * h * w) // (SPATIAL_MERGE_SIZE**2)
    return (
        [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * image_len + [VISION_END_TOKEN_ID]
    )


def video_span(grid_thw: list[int]) -> list[int]:
    t, h, w = grid_thw
    video_len = (t * h * w) // (SPATIAL_MERGE_SIZE**2)
    return (
        [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * video_len + [VISION_END_TOKEN_ID]
    )


def audio_span(audio_seqlen: int) -> list[int]:
    audio_len = feat_extract_output_lengths(audio_seqlen)
    return [AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * audio_len + [AUDIO_END_TOKEN_ID]


def audio_in_video_span(grid_thw: list[int], audio_seqlen: int) -> list[int]:
    t, h, w = grid_thw
    video_len = (t * h * w) // (SPATIAL_MERGE_SIZE**2)
    audio_len = feat_extract_output_lengths(audio_seqlen)
    # bos_len=2: vision_start, audio_start; then interleaved placeholders;
    # eos_len=2: vision_end, audio_end (order only affects st cursor length)
    return (
        [VISION_START_TOKEN_ID, AUDIO_START_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * video_len
        + [AUDIO_TOKEN_ID] * audio_len
        + [VISION_END_TOKEN_ID, AUDIO_END_TOKEN_ID]
    )


def test_feat_extract_lengths_matches_sglang() -> None:
    from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
        _get_feat_extract_output_lengths,
    )

    for n in (1, 50, 100, 101, 250, 1000):
        assert feat_extract_output_lengths(n) == int(
            _get_feat_extract_output_lengths(torch.tensor(n)).item()
        )


def test_text_only_no_grids_falls_back_to_arange() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    oracle, fast = oracle_and_fast(ids)
    assert_bit_identical(oracle, fast)


def test_single_image() -> None:
    grid = [1, 4, 4]
    tokens = [10, 11] + image_span(grid) + [12, 13]
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([grid], dtype=torch.long)
    oracle, fast = oracle_and_fast(ids, image_grid_thw=image_grid)
    assert_bit_identical(oracle, fast)


def test_single_video() -> None:
    grid = [2, 4, 4]
    tokens = [10] + video_span(grid) + [11]
    ids = torch.tensor([tokens], dtype=torch.long)
    video_grid = torch.tensor([grid], dtype=torch.long)
    seconds = torch.tensor([0.5], dtype=torch.float)
    oracle, fast = oracle_and_fast(
        ids, video_grid_thw=video_grid, second_per_grid_ts=seconds
    )
    assert_bit_identical(oracle, fast)


def test_image_then_audio() -> None:
    grid = [1, 8, 8]
    audio_seqlen = 200
    tokens = [1, 2] + image_span(grid) + [3] + audio_span(audio_seqlen) + [4]
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([grid], dtype=torch.long)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids, image_grid_thw=image_grid, audio_seqlens=audio_seqlens
    )
    assert_bit_identical(oracle, fast)


def test_audio_then_video() -> None:
    grid = [3, 4, 4]
    audio_seqlen = 150
    tokens = audio_span(audio_seqlen) + video_span(grid) + [99]
    ids = torch.tensor([tokens], dtype=torch.long)
    video_grid = torch.tensor([grid], dtype=torch.long)
    seconds = torch.tensor([1.0], dtype=torch.float)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_seqlens,
    )
    assert_bit_identical(oracle, fast)


def test_two_images() -> None:
    g0, g1 = [1, 4, 4], [1, 6, 6]
    tokens = [0] + image_span(g0) + [1, 2] + image_span(g1) + [3]
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([g0, g1], dtype=torch.long)
    oracle, fast = oracle_and_fast(ids, image_grid_thw=image_grid)
    assert_bit_identical(oracle, fast)


def test_audio_in_video_interleaved() -> None:
    grid = [4, 4, 4]
    audio_seqlen = 300
    tokens = [7, 8] + audio_in_video_span(grid, audio_seqlen) + [9]
    ids = torch.tensor([tokens], dtype=torch.long)
    video_grid = torch.tensor([grid], dtype=torch.long)
    seconds = torch.tensor([0.25], dtype=torch.float)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_seqlens,
        use_audio_in_video=True,
    )
    assert_bit_identical(oracle, fast)


def test_mixed_image_video_audio_in_video() -> None:
    img = [1, 4, 4]
    vid = [2, 4, 4]
    audio_seqlen = 180
    tokens = (
        [1] + image_span(img) + [2] + audio_in_video_span(vid, audio_seqlen) + [3, 4]
    )
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([img], dtype=torch.long)
    video_grid = torch.tensor([vid], dtype=torch.long)
    seconds = torch.tensor([0.5], dtype=torch.float)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_seqlens,
        use_audio_in_video=True,
    )
    assert_bit_identical(oracle, fast)


@pytest.mark.parametrize(
    "grid,audio_seqlen,seconds",
    [
        ([1, 4, 4], 50, 1.0),
        ([8, 2, 2], 400, 0.125),
        ([2, 8, 8], 100, 0.5),
    ],
)
def test_audio_in_video_parametrized(
    grid: list[int], audio_seqlen: int, seconds: float
) -> None:
    tokens = audio_in_video_span(grid, audio_seqlen)
    ids = torch.tensor([tokens], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids,
        video_grid_thw=torch.tensor([grid], dtype=torch.long),
        second_per_grid_ts=torch.tensor([seconds], dtype=torch.float),
        audio_seqlens=torch.tensor([audio_seqlen], dtype=torch.long),
        use_audio_in_video=True,
    )
    assert_bit_identical(oracle, fast)


def test_video_non_integer_timescale_25fps() -> None:
    """(arange * sec) * pps order must match oracle (25 FPS)."""
    grid = [12, 4, 4]
    tokens = [1] + video_span(grid) + [2]
    ids = torch.tensor([tokens], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids,
        video_grid_thw=torch.tensor([grid], dtype=torch.long),
        second_per_grid_ts=torch.tensor([1.0 / 25.0], dtype=torch.float),
    )
    assert_bit_identical(oracle, fast)


def test_audio_in_video_eos_uses_last_emitted_column() -> None:
    """AIV eos st_idx follows last emitted column, not max(merged)."""
    grid = [1, 56, 56]
    audio_seqlen = 100
    tokens = [1] + audio_in_video_span(grid, audio_seqlen) + [2]
    ids = torch.tensor([tokens], dtype=torch.long)
    oracle, fast = oracle_and_fast(
        ids,
        video_grid_thw=torch.tensor([grid], dtype=torch.long),
        second_per_grid_ts=torch.tensor([1.0], dtype=torch.float),
        audio_seqlens=torch.tensor([audio_seqlen], dtype=torch.long),
        use_audio_in_video=True,
    )
    assert_bit_identical(oracle, fast)


def test_compute_mrope_positions_wires_vectorized_path(monkeypatch) -> None:
    """_compute_mrope_positions calls the vectorized path, [3, seq] layout."""
    from types import SimpleNamespace

    from sglang_omni.models.qwen3_omni import mrope_positions as mp
    from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions

    real_vectorized = mp.get_rope_index_qwen3_omni_vectorized
    calls: list[int] = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real_vectorized(*args, **kwargs)

    monkeypatch.setattr(mp, "get_rope_index_qwen3_omni_vectorized", spy)

    grid = [1, 4, 4]
    tokens = [10] + image_span(grid) + [11]
    input_ids = torch.tensor(tokens, dtype=torch.long)
    model_inputs = {"image_grid_thw": torch.tensor([grid], dtype=torch.long)}
    thinker_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=None),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
    )
    result = compute_mrope_positions(input_ids, model_inputs, thinker_config)
    assert calls, "hot path must call get_rope_index_qwen3_omni_vectorized"
    assert result is not None
    positions, delta = result
    assert positions.shape == (3, len(tokens))
    oracle_pos, oracle_delta = get_rope_index_qwen3_omni(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs["image_grid_thw"],
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        use_audio_in_video=False,
        audio_seqlens=None,
    )
    assert torch.equal(positions.float(), oracle_pos.squeeze(1).float())
    assert torch.equal(delta.float(), oracle_delta.float())
