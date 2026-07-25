# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

from sglang_omni.models.higgs_tts import audio_codec


def test_higgs_codec_uses_upstream_transformers_architecture() -> None:
    assert audio_codec.HiggsAudioV2TokenizerConfig is HiggsAudioV2TokenizerConfig
    assert audio_codec.HiggsAudioV2TokenizerModel is HiggsAudioV2TokenizerModel

    config = HiggsAudioV2TokenizerConfig.from_json_file(
        audio_codec._BUNDLED_CODEC_CONFIG_PATH
    )
    with torch.device("meta"):
        model = audio_codec.HiggsAudioV2TokenizerModel(config)

    state = model.state_dict()
    assert len(state) == 527
    assert {
        key: tuple(state[key].shape)
        for key in (
            "acoustic_encoder.block.0.conv1.weight",
            "acoustic_decoder.block.0.conv_t1.weight",
            "quantizer.quantizers.0.codebook.embed",
            "semantic_model.encoder.layers.0.attention.q_proj.weight",
        )
    } == {
        "acoustic_encoder.block.0.conv1.weight": (128, 64, 16),
        "acoustic_decoder.block.0.conv_t1.weight": (1024, 512, 16),
        "quantizer.quantizers.0.codebook.embed": (1024, 64),
        "semantic_model.encoder.layers.0.attention.q_proj.weight": (768, 768),
    }
    assert config.frame_rate == 25
    assert config.num_quantizers == 8


class _FakeQuantizerLayer:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.to(torch.float32).unsqueeze(-1) + self.offset


def test_capture_safe_quantizer_decode_matches_additive_rvq() -> None:
    quantizer = SimpleNamespace(
        quantizers=[
            _FakeQuantizerLayer(0.25),
            _FakeQuantizerLayer(0.5),
            _FakeQuantizerLayer(0.75),
        ]
    )
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)

    actual = audio_codec._capture_safe_quantizer_decode(quantizer, codes)
    expected = sum(
        layer.decode(indices) for layer, indices in zip(quantizer.quantizers, codes)
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_codec_decode_replays_matching_shape_graph() -> None:
    graph_input = torch.zeros((1, 2, 3), dtype=torch.long)
    graph_output = torch.zeros((1, 1, 1), dtype=torch.float32)

    class _FakeGraph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1
            graph_output.fill_(float(graph_input.sum()))

    graph = _FakeGraph()
    codec = object.__new__(audio_codec.HiggsAudioCodec)
    codec._decode_cuda_graphs = {
        3: audio_codec._DecodeCudaGraph(
            graph=graph,
            input_codes=graph_input,
            output_audio=graph_output,
        )
    }
    codec._decode_cuda_graph_hits = 0
    codec._decode_cuda_graph_misses = 0
    codec._decode_cuda_graph_missed_shapes = set()

    output = codec.decode(torch.tensor([[1, 2], [3, 4], [5, 6]]))

    assert graph.replays == 1
    assert codec._decode_cuda_graph_hits == 1
    assert codec._decode_cuda_graph_misses == 0
    torch.testing.assert_close(output, torch.tensor([21.0]), rtol=0, atol=0)
