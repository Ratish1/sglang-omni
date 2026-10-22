"""Check trained transformer numerics, retained state and graph performance."""

import argparse
import copy
import json
import statistics
from pathlib import Path

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.incremental_codec import (
    FusedIncrementalTransformer,
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
    incremental_transformer,
)
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)


def error(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


def state_for(transformer, dtype):
    state = Qwen3TTSIncrementalCodecState(
        frame_positions=torch.tensor([0, 23, 10000], device="cuda")
    )
    attention = transformer.layers[0].self_attn
    for index in range(len(transformer.layers)):
        shape = (
            3,
            transformer.config.num_key_value_heads,
            transformer.window_size - 1,
            attention.head_dim,
        )
        state.transformer_keys[index] = torch.randn(shape, device="cuda").to(dtype)
        state.transformer_values[index] = torch.randn(shape, device="cuda").to(dtype)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("mutation", "checkpoint"))
    mode = parser.parse_args().mode
    torch.set_num_threads(1)
    torch.manual_seed(3051)
    apply_qwen_tts_transformers_compatibility_patches()
    if mode == "mutation":
        from tests.unit_test.qwen3_tts.test_incremental_codec import (
            test_fused_transformer_tracks_fp32_as_closely_as_the_plain_step,
        )

        def ignore_weight(hidden_states, norm):
            normalized = hidden_states.float()
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(-1, keepdim=True) + norm.variance_epsilon
            )
            return normalized.to(hidden_states.dtype)

        FusedIncrementalTransformer.normalize = staticmethod(ignore_weight)
        test_fused_transformer_tracks_fp32_as_closely_as_the_plain_step()
        print(
            json.dumps({"ignoring_all_rmsnorm_weights_passes_new_test": True}),
            flush=True,
        )
        return

    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    reference = tokenizer.model.decoder.pre_transformer.eval()
    report = {"model": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "cases": [], "benchmarks": []}
    norms = [reference.norm] + [
        norm
        for layer in reference.layers
        for norm in (layer.input_layernorm, layer.post_attention_layernorm)
    ]
    report["non_unit_norm_elements"] = sum(
        int((norm.weight != 1).sum()) for norm in norms
    )
    for dtype in (torch.bfloat16, torch.float16):
        plain = copy.deepcopy(reference).to(dtype)
        fused = FusedIncrementalTransformer(copy.deepcopy(reference).to(dtype))
        truth = state_for(reference, torch.float32)
        states = {"reference": truth}
        for name in ("plain", "fused"):
            state = truth.clone()
            state.transformer_keys = {
                index: value.to(dtype)
                for index, value in state.transformer_keys.items()
            }
            state.transformer_values = {
                index: value.to(dtype)
                for index, value in state.transformer_values.items()
            }
            states[name] = state
        with torch.inference_mode():
            for frames, valid in ((1, None), (8, None), (64, None), (17, [17, 14, 9])):
                inputs = torch.randn(
                    3, frames, reference.config.latent_dim, device="cuda"
                )
                valid_frames = (
                    None if valid is None else torch.tensor(valid, device="cuda")
                )
                expected = incremental_transformer(
                    reference, inputs, states["reference"], valid_frames
                )
                ordinary = incremental_transformer(
                    plain, inputs.to(dtype), states["plain"], valid_frames
                )
                output = fused(inputs.to(dtype), states["fused"], valid_frames)
                assert torch.isfinite(output).all()
                key_ratios = []
                for field in ("transformer_keys", "transformer_values"):
                    for index, value in getattr(states["reference"], field).items():
                        p = error(getattr(states["plain"], field)[index], value)
                        f = error(getattr(states["fused"], field)[index], value)
                        key_ratios.append(f / p)
                report["cases"].append(
                    dict(
                        dtype=str(dtype),
                        frames=frames,
                        plain_error=error(ordinary, expected),
                        fused_error=error(output, expected),
                        max_state_error_ratio=max(key_ratios),
                    )
                )
                for state in states.values():
                    state.frame_positions += (
                        frames if valid_frames is None else valid_frames
                    )
            compiled = torch.compile(fused.__call__, dynamic=False, fullgraph=True)
            compiled_state = states["fused"].clone()
            eager_state = states["fused"].clone()
            compiled_reference_state = states["reference"].clone()
            compiled_plain_state = states["plain"].clone()
            for step in range(2):
                inputs = torch.randn(
                    3, 8, reference.config.latent_dim, device="cuda", dtype=dtype
                )
                expected = fused(inputs, eager_state)
                output = compiled(inputs, compiled_state)
                fp32_output = incremental_transformer(
                    reference, inputs.float(), compiled_reference_state
                )
                plain_output = incremental_transformer(
                    plain, inputs, compiled_plain_state
                )
                assert torch.isfinite(output).all()
                report["cases"].append(
                    dict(
                        dtype=str(dtype),
                        compiled_step=step,
                        compiled_eager_error=error(output, expected),
                        compiled_fp32_error=error(output, fp32_output),
                        plain_fp32_error=error(plain_output, fp32_output),
                    )
                )
                for state in (
                    compiled_state,
                    eager_state,
                    compiled_reference_state,
                    compiled_plain_state,
                ):
                    state.advance(8)
            for batch, frames in ((1, 1), (8, 4), (16, 64)):
                inputs = torch.randn(
                    batch,
                    frames,
                    reference.config.latent_dim,
                    device="cuda",
                    dtype=dtype,
                )
                for name, function in (
                    ("plain", lambda x, s: incremental_transformer(plain, x, s)),
                    ("fused", fused),
                ):
                    state = Qwen3TTSIncrementalCodecState()
                    function(inputs, state)
                    state.advance(frames)
                    for _ in range(3):
                        function(inputs, state)
                        state.advance(frames)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = function(inputs, state)
                    timings = []
                    for _ in range(5):
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        timings.append(start.elapsed_time(end) / 20)
                    report["benchmarks"].append(
                        dict(
                            dtype=str(dtype),
                            batch=batch,
                            frames=frames,
                            route=name,
                            median_ms=statistics.median(timings),
                        )
                    )
    decoder = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder.bfloat16())
    assert decoder.fused_transformer is not None
    arena = Qwen3TTSCodecStateArena(
        decoder, num_slots=3, device=torch.device("cuda", 0), dtype=torch.bfloat16
    )
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        decoder,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        num_quantizers=tokenizer.model.decoder.config.num_quantizers,
        mode="warm",
        fresh_frames=(4,),
        batch_sizes=(2,),
        min_free_gb=0,
        arena=arena,
        graph_pool=IncrementalCodecGraphPool(stream_priority=0),
    )
    runner.capture()
    assert runner.enabled, runner.disable_reason
    slots = [arena.acquire(), arena.acquire()]
    state = arena.gather(slots)
    waveform_errors = []
    with torch.inference_mode():
        for step in range(3):
            codes = torch.randint(
                tokenizer.model.decoder.config.codebook_size,
                (2, tokenizer.model.decoder.config.num_quantizers, 4),
                device="cuda",
            )
            expected = decoder.decode(codes, state)
            actual = runner.decode_slots(codes, slots)
            waveform_errors.append(error(actual, expected))
            torch.testing.assert_close(actual, expected)
            gathered = arena.gather(slots)
            for actual_kv, expected_kv in (
                (gathered.transformer_keys, state.transformer_keys),
                (gathered.transformer_values, state.transformer_values),
            ):
                for index in actual_kv:
                    assert torch.equal(actual_kv[index], expected_kv[index])
            assert torch.equal(gathered.frame_positions, state.frame_positions)
    report["full_trained_decoder_graph_waveform_errors"] = waveform_errors
    Path("/data/review3051/checkpoint.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
