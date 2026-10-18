"""Verify equal-axis RoPE and live-position CUDA replay on the PR's real kernels."""

from __future__ import annotations

import copy
import json

import torch
from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.stages import register_qwen3_tts_hf_config
from sglang_omni.utils.hf import use_text_rope

apply_qwen_tts_transformers_compatibility_patches()
import qwen_tts.core.models.configuration_qwen3_tts as configuration


def reference_rotation(
    values: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    cosine, sine = cache[positions.long()].chunk(2, dim=-1)
    first, second = values.float().chunk(2, dim=-1)
    return torch.cat(
        (
            first * cosine[:, None] - second * sine[:, None],
            second * cosine[:, None] + first * sine[:, None],
        ),
        dim=-1,
    )


def assert_rotation(
    actual: torch.Tensor, expected: torch.Tensor, inputs: torch.Tensor
) -> None:
    first, second = inputs.float().chunk(2, dim=-1)
    magnitude = torch.cat((first.abs() + second.abs(),) * 2, dim=-1)
    # note (ratish): one bf16 output rounding plus fp32 fused arithmetic;
    # the bound scales with operands, including cancellation near zero.
    bound = magnitude * (2**-8 + 8 * 2**-24) + 1e-7
    assert torch.all((actual.float() - expected).abs() <= bound)


@torch.inference_mode()
def main() -> None:
    register_qwen3_tts_hf_config()

    parameters = {
        "rope_type": "default",
        "interleaved": True,
        "mrope_section": [24, 20, 20],
    }
    original = configuration.Qwen3TTSTalkerConfig(
        rope_theta=1_000_000, rope_scaling=copy.deepcopy(parameters)
    )
    normalized = copy.deepcopy(original)
    use_text_rope(normalized)
    assert original.rope_scaling == parameters
    root = configuration.Qwen3TTSConfig(
        talker_config={"rope_theta": 1_000_000, "rope_scaling": parameters}
    )
    assert root.text_config is root.talker_config
    assert "mrope_section" not in root.text_config.rope_scaling
    assert configuration.Qwen3TTSConfig().text_config.rope_parameters is None
    torch.manual_seed(1907)
    observations: list[dict[str, str | int | float]] = []
    for index_dtype in (torch.int32, torch.int64):
        with torch.device("cuda"):
            text_rope = get_rope(
                128,
                128,
                32768,
                1_000_000,
                rope_scaling=normalized.rope_scaling,
                dtype=torch.bfloat16,
            )
            multimodal_rope = get_rope(
                128,
                128,
                32768,
                1_000_000,
                rope_scaling=original.rope_scaling,
                dtype=torch.bfloat16,
            )
        assert not isinstance(text_rope, MRotaryEmbedding)
        assert isinstance(multimodal_rope, MRotaryEmbedding)
        assert text_rope.cos_sin_cache.dtype == torch.float32
        initial_positions = torch.tensor(
            [0, 1, 17, 1023, 8191, 32767], device="cuda", dtype=index_dtype
        )
        query_source = torch.randn(6, 16, 128, device="cuda", dtype=torch.bfloat16)
        key_source = torch.randn(6, 8, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key_source)
        cache = text_rope.cos_sin_cache
        # note (ratish): compare the equal-axis derivation at the same precision;
        # the serving MRoPE path subsequently narrows this cache to bf16.
        multimodal_rope.cos_sin_cache = cache
        axis_query, axis_key = multimodal_rope.forward_native(
            initial_positions.long()[None].expand(3, -1),
            query_source.float().flatten(1),
            key_source.float().flatten(1),
        )
        expected_query = reference_rotation(query_source, cache, initial_positions)
        expected_key = reference_rotation(key_source, cache, initial_positions)
        assert torch.equal(axis_query.view_as(expected_query), expected_query)
        assert torch.equal(axis_key.view_as(expected_key), expected_key)
        query = query_source.clone()
        key = key_source.clone()
        positions = initial_positions.clone()
        locations = torch.tensor(
            [13, 2, 23, 5, 31, 7], device="cuda", dtype=index_dtype
        )
        key_cache = torch.full((40, 8 * 128), 9.0, device="cuda", dtype=torch.bfloat16)
        value_cache = torch.full_like(key_cache, 9.0)
        store = FusedSetKVBufferArg(value.flatten(1), key_cache, value_cache, locations)

        def run() -> None:
            text_rope(
                positions,
                query.flatten(1),
                key.flatten(1),
                fused_set_kv_buffer_arg=store,
            )

        run()
        assert_rotation(query, expected_query, query_source)
        assert_rotation(key, expected_key, key_source)
        assert torch.equal(key_cache[locations.long()], key.flatten(1))
        assert torch.equal(value_cache[locations.long()], value.flatten(1))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        for iteration in range(3):
            positions.copy_((initial_positions.long() + iteration * 151) % 32768)
            locations.copy_(
                (torch.tensor([13, 2, 23, 5, 31, 7], device="cuda") + iteration) % 40
            )
            query.copy_(query_source)
            key.copy_(key_source)
            key_cache.fill_(9.0)
            value_cache.fill_(9.0)
            graph.replay()
            expected_query = reference_rotation(query_source, cache, positions)
            expected_key = reference_rotation(key_source, cache, positions)
            assert_rotation(query, expected_query, query_source)
            assert_rotation(key, expected_key, key_source)
            assert torch.equal(key_cache[locations.long()], key.flatten(1))
            assert torch.equal(value_cache[locations.long()], value.flatten(1))
            untouched = torch.ones(40, device="cuda", dtype=torch.bool)
            untouched[locations.long()] = False
            assert torch.all(key_cache[untouched] == 9)
            assert torch.all(value_cache[untouched] == 9)
        observations.append(
            {
                "index_dtype": str(index_dtype),
                "graph_replays": 3,
                "query_max_error": float((query.float() - expected_query).abs().max()),
                "key_max_error": float((key.float() - expected_key).abs().max()),
            }
        )
        graph.reset()
    print(json.dumps({"rope_checks": observations}, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
