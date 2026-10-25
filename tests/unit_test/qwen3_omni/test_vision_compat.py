# SPDX-License-Identifier: Apache-2.0
"""Contracts for Qwen3-Omni vision encoding."""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

import sglang_omni.models.qwen3_omni.components.vision_encoder as vision_runtime
from sglang_omni.models.qwen3_omni.components import image_encoder, vision_compat
from sglang_omni.models.qwen3_omni.components.vision_encoder import (
    Qwen3OmniVisionEncoder,
)

GRID_CASES = {
    "single": [[1, 16, 16]],
    "single-large": [[1, 64, 98]],
    "multi-mixed": [[1, 16, 24], [1, 32, 12], [1, 8, 40]],
    "video": [[4, 16, 20]],
    "video-plus-image": [[3, 12, 16], [1, 20, 28], [2, 8, 8]],
    "many": [[1, 24, 24]] * 8,
}
BILINEAR_TOLERANCES = {torch.bfloat16: 2**-9 + 1e-5, torch.float32: 1e-5}
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.accelerator,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ],
    ),
]


def make_encoder(
    dtype: torch.dtype, device: str
) -> vision_compat.Qwen3OmniMoeVisionEncoderCompat:
    encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat.__new__(
        vision_compat.Qwen3OmniMoeVisionEncoderCompat
    )
    torch.nn.Module.__init__(encoder)
    encoder.spatial_merge_size = 2
    encoder.num_grid_per_side = 48
    encoder.pos_embed = torch.nn.Embedding(
        48 * 48,
        16,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        values = torch.arange(
            encoder.pos_embed.weight.numel(),
            dtype=torch.float32,
        ).reshape_as(encoder.pos_embed.weight)
        encoder.pos_embed.weight.copy_(torch.sin(values / 113))
    return encoder


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", list(BILINEAR_TOLERANCES), ids=["bf16", "fp32"])
@pytest.mark.parametrize("grid_thw", list(GRID_CASES.values()), ids=list(GRID_CASES))
def test_position_interpolation_matches_float64_bilinear(
    grid_thw: list[list[int]],
    dtype: torch.dtype,
    device: str,
) -> None:
    encoder = make_encoder(dtype, device)
    table = encoder.pos_embed.weight.detach().cpu().double().T.reshape(1, -1, 48, 48)
    expected = torch.cat(
        [
            F.interpolate(table, size=(h, w), mode="bilinear", align_corners=True)[0]
            .reshape(-1, h // 2, 2, w // 2, 2)
            .permute(1, 3, 2, 4, 0)
            .reshape(h * w, -1)
            .repeat(t, 1)
            for t, h, w in grid_thw
        ]
    )

    actual = encoder.interpolate_position_embeddings(grid_thw)

    assert actual.device == encoder.pos_embed.weight.device
    torch.testing.assert_close(
        actual.cpu().double(), expected, rtol=0, atol=BILINEAR_TOLERANCES[dtype]
    )


def apply_cpu_joint_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> None:
    assert is_neox
    assert cos_sin_cache.dtype == torch.float32
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    rotated_query, rotated_key = hf_modeling.apply_rotary_pos_emb_vision(
        query, key, torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
    )
    query.copy_(rotated_query)
    key.copy_(rotated_key)


@pytest.fixture
def vision_encoder() -> vision_compat.Qwen3OmniMoeVisionEncoderCompat:
    # note (yzxiao): Keep the serving head dimension (72) in the small CPU model.
    config = hf_modeling.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=144,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=(0, 1),
    )
    previous_dtype = torch.get_default_dtype()
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(734)
        torch.set_default_dtype(torch.bfloat16)
        try:
            encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat(config)
        finally:
            torch.set_default_dtype(previous_dtype)
    return encoder.eval()


@pytest.mark.parametrize(
    ("platform_device", "has_provider", "should_use_joint_kernel"),
    [("cpu", True, True), ("cpu", False, False), ("cuda", True, False)],
    ids=["joint-provider", "native-rope", "model-platform-mismatch"],
)
def test_image_encoder_preserves_outputs_with_platform_rotary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    vision_encoder: vision_compat.Qwen3OmniMoeVisionEncoderCompat,
    platform_device: Literal["cpu", "cuda"],
    has_provider: bool,
    should_use_joint_kernel: bool,
) -> None:
    save_file(
        {
            f"thinker.visual.{name}": tensor
            for name, tensor in vision_encoder.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )
    kernel = Mock(side_effect=apply_cpu_joint_rope)
    provider = Mock(return_value=kernel if has_provider else None)
    monkeypatch.setattr(
        vision_runtime,
        "current_platform",
        SimpleNamespace(
            device_type=platform_device, get_joint_rope_inplace_kernel=provider
        ),
    )
    monkeypatch.setattr(
        image_encoder,
        "load_thinker_config",
        Mock(return_value=SimpleNamespace(vision_config=vision_encoder.config)),
    )
    model = image_encoder.Qwen3OmniImageEncoder(
        str(tmp_path), device="cpu", dtype="bf16"
    )
    image_encoder.optimize_patch_embed(vision_encoder)
    image_grid = torch.tensor([[1, 4, 4], [1, 4, 6]], dtype=torch.long, device="cpu")
    video_grid = torch.tensor([[2, 6, 4]], dtype=torch.long, device="cpu")
    image_pixels = torch.randn(40, 24, dtype=torch.bfloat16, device="cpu")
    video_pixels = torch.randn(48, 24, dtype=torch.bfloat16, device="cpu")

    with torch.no_grad():
        actual = model(
            pixel_values=image_pixels,
            image_grid_thw=image_grid,
            pixel_values_videos=video_pixels,
            video_grid_thw=video_grid,
        )
        expected_image = vision_encoder(
            image_pixels, grid_thw=image_grid, return_dict=True
        )
        expected_video = vision_encoder(
            video_pixels, grid_thw=video_grid, return_dict=True
        )

    torch.testing.assert_close(
        actual,
        {
            "image_embeds": expected_image.pooler_output,
            "image_grid_thw": image_grid,
            "image_token_counts": torch.tensor([4, 6], device="cpu"),
            "deepstack_visual_embeds_image": expected_image.deepstack_features,
            "video_embeds": expected_video.pooler_output,
            "video_grid_thw": video_grid,
            "video_token_counts": torch.tensor([12], device="cpu"),
            "deepstack_visual_embeds_video": expected_video.deepstack_features,
        },
        rtol=0,
        atol=0,
    )
    expected_kernel_calls = (
        2 * vision_encoder.config.depth if should_use_joint_kernel else 0
    )
    assert kernel.call_count == expected_kernel_calls
    assert provider.call_count == (1 if platform_device == "cpu" else 0)


@pytest.mark.parametrize(
    "has_joint_kernel", [False, True], ids=["native-rope", "joint-rope"]
)
def test_flash_backend_receives_host_max_sequence_lengths(
    monkeypatch: pytest.MonkeyPatch,
    vision_encoder: vision_compat.Qwen3OmniMoeVisionEncoderCompat,
    has_joint_kernel: bool,
) -> None:
    kernel = Mock(side_effect=apply_cpu_joint_rope)
    encoder = (
        Qwen3OmniVisionEncoder(
            vision_encoder.config,
            joint_rope_kernel=kernel if has_joint_kernel else None,
        )
        .eval()
        .to(torch.bfloat16)
    )
    encoder.load_state_dict(vision_encoder.state_dict(), strict=True)
    grid = torch.tensor([[1, 4, 4], [2, 4, 6]], dtype=torch.long, device="cpu")
    pixels = torch.randn(64, 24, dtype=torch.bfloat16, device="cpu")
    attention_output = pixels.new_zeros(
        1,
        pixels.shape[0],
        encoder.config.num_heads,
        encoder.config.hidden_size // encoder.config.num_heads,
    )
    flash_backend = Mock(return_value=(attention_output, None))
    monkeypatch.setattr(
        hf_modeling, "is_flash_attention_requested", Mock(return_value=True)
    )
    monkeypatch.setattr(
        hf_modeling.ALL_ATTENTION_FUNCTIONS,
        "get_interface",
        Mock(return_value=flash_backend),
    )

    with torch.no_grad():
        encoder(pixels, grid_thw=grid)

    assert flash_backend.call_count == vision_encoder.config.depth
    for backend_call in flash_backend.call_args_list:
        assert isinstance(backend_call.kwargs["max_length_q"], int)
        assert isinstance(backend_call.kwargs["max_length_k"], int)
        assert backend_call.kwargs["max_length_q"] == 24
        assert backend_call.kwargs["max_length_k"] == 24
