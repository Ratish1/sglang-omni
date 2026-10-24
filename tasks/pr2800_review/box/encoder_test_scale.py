"""Output scale and failure sensitivity of the #3131 public CUDA encoder test cases, to set their tolerances.

Run on the box from the review fix tree (cwd and PYTHONPATH the tree): python3 encoder_test_scale.py --out <dir>
Rebuilds test_cuda_image_encoder_matches_the_fp32_transformers_encoder's and
test_cuda_audio_encoder_matches_the_fp32_transformers_tower's inputs exactly (same fixtures, seeds, checkpoints), and
per row prints the fp32 reference's largest and mean absolute value, the largest absolute difference of the served
encoder to it, and, for the bf16 rows, the largest difference when the packed attention is replaced by one that
ignores sequence bounds (every token attends to every token): the error a broken segment layout would show.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components import audio_encoder, image_encoder, vision_compat


class UnboundedAttention(nn.Module):
    """Attention over the whole packed sequence, as if every token were one sequence."""

    def forward(self, q, k, v, cu_seqlens, bsz, seq_len, softmax_scale, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), scale=softmax_scale
        ).transpose(0, 1)


def vision_fixture() -> vision_compat.Qwen3OmniMoeVisionEncoderCompat:
    config = hf_modeling.Qwen3OmniMoeVisionEncoderConfig(
        depth=2, hidden_size=144, intermediate_size=32, num_heads=2, in_channels=3, patch_size=2,
        spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=16, num_position_embeddings=64,
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


def report(name: str, actual: list[torch.Tensor], expected: list[torch.Tensor], broken: list[torch.Tensor] | None) -> None:
    for index, (got, want) in enumerate(zip(actual, expected)):
        line = (f"{name}[{index}] reference max {float(want.abs().max()):.4g} mean {float(want.abs().mean()):.4g}; "
                f"served largest difference {float((got.float() - want).abs().max()):.4g}")
        if broken is not None:
            line += f"; unbounded attention largest difference {float((broken[index].float() - want).abs().max()):.4g}"
        else:
            pass
        print(line, flush=True)


def vision(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = vision_fixture()
    save_file({f"thinker.visual.{name}": tensor for name, tensor in fixture.state_dict().items()},
              tmp / "model.safetensors")
    monkeypatch.setattr(image_encoder, "load_thinker_config",
                        Mock(return_value=SimpleNamespace(vision_config=fixture.config)))
    reference = copy.deepcopy(fixture).to("cuda", torch.float32)
    grid = torch.tensor([[1, 4, 4], [2, 4, 6]], dtype=torch.long)
    torch.manual_seed(0)
    pixels = torch.randn(64, 24)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = reference(pixels.to("cuda"), grid_thw=grid.to("cuda"), return_dict=True)
    expected_rows = [expected.pooler_output, *expected.deepstack_features]
    for dtype in ("bf16", "fp32"):
        model = image_encoder.Qwen3OmniImageEncoder(str(tmp), device="cuda", dtype=dtype)
        with torch.no_grad():
            output = model(pixel_values=pixels, image_grid_thw=grid)
        actual = [output["image_embeds"], *output["deepstack_visual_embeds_image"]]
        broken = None
        if dtype == "bf16":
            for block in model.visual.blocks:
                block.attn.packed_attention = UnboundedAttention()
            with torch.no_grad():
                output = model(pixel_values=pixels, image_grid_thw=grid)
            broken = [output["image_embeds"], *output["deepstack_visual_embeds_image"]]
        else:
            pass
        report(f"vision {dtype}", actual, expected_rows, broken)


def audio(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=256, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=512, downsample_hidden_size=16,
        output_dim=64,
    )
    torch.manual_seed(0)
    reference = hf_modeling.Qwen3OmniMoeAudioEncoder(config).eval()
    save_file({f"thinker.audio_tower.{name}": tensor.contiguous() for name, tensor in reference.state_dict().items()},
              tmp / "model.safetensors")
    monkeypatch.setattr(audio_encoder, "load_thinker_config", Mock(return_value=SimpleNamespace(audio_config=config)))
    reference = reference.to("cuda", torch.float32)
    lengths = torch.tensor([400, 850, 99])
    features = torch.randn(3, config.num_mel_bins, 850)
    mask = torch.arange(850) < lengths.unsqueeze(1)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = reference(
            torch.cat([row[:, :length] for row, length in zip(features, lengths)], dim=-1).to("cuda"),
            feature_lens=lengths.to("cuda"),
        ).last_hidden_state
    for dtype, graphs in (("bf16", True), ("bf16", False), ("fp32", False)):
        model = audio_encoder.Qwen3OmniAudioEncoder(str(tmp), device="cuda", dtype=dtype, enable_layer_cuda_graph=graphs)
        with torch.no_grad():
            actual = model(input_features=features, feature_attention_mask=mask)["audio_embeds"]
        broken = None
        if dtype == "bf16" and not graphs:
            for layer in model.audio_tower.layers:
                layer.self_attn.packed_attention = UnboundedAttention()
            with torch.no_grad():
                broken = [model(input_features=features, feature_attention_mask=mask)["audio_embeds"]]
        else:
            pass
        report(f"audio {dtype} graphs={graphs}", [actual], [expected], broken)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
    for name, run in (("vision", vision), ("audio", audio)):
        (args.out / name).mkdir(exist_ok=True)
        run(args.out / name, monkeypatch)
    monkeypatch.undo()


if __name__ == "__main__":
    main()
