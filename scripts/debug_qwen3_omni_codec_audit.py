# SPDX-License-Identifier: Apache-2.0
"""Compare full, graph and incremental vocoding of captured codec frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeCode2WavTransformerLayer,
)

from scripts.qwen3_omni_audit_utils import compare_tensors
from sglang_omni.models.qwen3_omni.components.code2wav import (
    FusedCode2WavTransformer,
    Qwen3OmniIncrementalCodec,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder


def capture_layer(
    module: Qwen3OmniMoeCode2WavTransformerLayer,
    inputs: tuple[torch.Tensor],
    output: torch.Tensor,
    *,
    snapshots: list[torch.Tensor],
) -> None:
    """The fixed hook signature includes the layer and its inputs."""
    snapshots.append(output.detach().clone())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--chunk-frames", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.chunk_frames <= 0:
        parser.error("Chunk frames must be positive")
    else:
        pass
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    trace_files = sorted(arguments.trace_directory.glob("*-post_*.pt"))
    assert trace_files
    codec_frames = []
    trace_hashes = {}
    for trace_file in trace_files:
        captured = torch.load(trace_file, weights_only=True)
        codes = captured["codec_ids"]
        assert codes.shape == (1, config.talker_config.num_code_groups)
        trace_hashes[trace_file.name] = hashlib.sha256(
            trace_file.read_bytes()
        ).hexdigest()
        if int(codes[0, 0]) == config.talker_config.codec_eos_token_id:
            assert trace_file == trace_files[-1]
        else:
            codec_frames.append(codes)
    codes = torch.cat(codec_frames, dim=0).T.unsqueeze(0).to(device)
    assert codes.min() >= 0 and codes.max() < config.code2wav_config.codebook_size
    torch.save(codes.cpu(), arguments.output_directory / "codes.pt")
    reference, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        config=config.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    native = load_code2wav_model(arguments.model_path, device=str(device), dtype="bf16")
    replaced_activations = fuse_vocoder_decoder(native.decoder)
    assert replaced_activations > 0
    assert isinstance(native.pre_transformer, FusedCode2WavTransformer)
    for name, parameter in reference.named_parameters():
        native_name = name.replace("pre_transformer.", "pre_transformer.transformer.")
        assert torch.equal(parameter, native.get_parameter(native_name)), name
    reference_layers: list[torch.Tensor] = []
    native_layers: list[torch.Tensor] = []
    hooks = [
        layer.register_forward_hook(partial(capture_layer, snapshots=reference_layers))
        for layer in reference.pre_transformer.layers
    ]
    layer_output = native.pre_transformer.layer_output

    def record_layer(
        layer: Qwen3OmniMoeCode2WavTransformerLayer,
        residual: torch.Tensor,
        attended: torch.Tensor,
    ) -> torch.Tensor:
        output = layer_output(layer, residual, attended)
        native_layers.append(output.detach().unsqueeze(0).clone())
        return output

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        expected = reference(codes)
        with patch.object(native.pre_transformer, "layer_output", record_layer):
            actual = native(codes)
    for hook in hooks:
        hook.remove()
    assert (
        len(reference_layers)
        == len(native_layers)
        == config.code2wav_config.num_hidden_layers
    )
    report = {
        "model_path": arguments.model_path,
        "gpu": torch.cuda.get_device_name(device),
        "transformers": version("transformers"),
        "trace_hashes": trace_hashes,
        "codec_frames": codes.shape[-1],
        "waveform_samples": actual.shape[-1],
        "reference_waveform_samples": expected.shape[-1],
        "transconv_trim_samples": int(
            codes.shape[-1] * native.total_upsample - actual.shape[-1]
        ),
        "fused_activations": replaced_activations,
        "waveform_error": asdict(compare_tensors(actual, expected)),
        "transformer_layers": [
            asdict(compare_tensors(actual_layer, expected_layer))
            for actual_layer, expected_layer in zip(
                native_layers, reference_layers, strict=True
            )
        ],
    }
    with torch.inference_mode():
        full_native = native(codes)
        report["math_vs_default_attention"] = asdict(
            compare_tensors(full_native, actual)
        )
        codec = Qwen3OmniIncrementalCodec(native)
        pool = CodecStatePool(
            codec.spec,
            num_slots=1,
            capture_slots=0,
            ring_frames=codec.spec.retained_context + arguments.chunk_frames,
            device=device,
            dtype=torch.bfloat16,
        )
        slot = pool.acquire()
        assert slot is not None
        slot_index = pool.stage_index([slot])
        chunks = []
        for offset in range(0, codes.shape[-1], arguments.chunk_frames):
            chunks.append(
                codec.decode_slots(
                    codes[..., offset : offset + arguments.chunk_frames],
                    pool,
                    slot_index,
                    first_decode=offset == 0,
                ).clone()
            )
        incremental = torch.cat(chunks, dim=-1)
        assert int(pool.frame_positions[slot]) == codes.shape[-1]
        pool.release(slot)
        report["incremental_vs_full_native"] = asdict(
            compare_tensors(incremental, full_native)
        )
        report["incremental_vs_math_native"] = asdict(
            compare_tensors(incremental, actual)
        )
        report["chunk_frames"] = arguments.chunk_frames
        report["ring_frames"] = pool.ring_frames
        report["ring_wraps"] = codes.shape[-1] // pool.ring_frames
        static_codes = codes.clone()
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            native(static_codes)
            native(static_codes)
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_output = native(static_codes)
        graph.replay()
        report["graph_vs_full_native"] = asdict(
            compare_tensors(graph_output, full_native)
        )
        graph.replay()
        report["graph_repeat_vs_full_native"] = asdict(
            compare_tensors(graph_output, full_native)
        )
    report["scope"] = (
        "Fixed real codec frames isolate vocoding from Talker sampling. Transformer layers, full waveform, repeated full-shape CUDA replay and incremental ring rollover are measured. This does not certify speech task accuracy or the scheduler's multi-request lifetime."
    )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {name: value for name, value in report.items() if name != "trace_hashes"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
