# SPDX-License-Identifier: Apache-2.0
"""Compare checkpoint audio and vision encoders at every layer and output."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder,
)

from scripts.qwen3_omni_audit_utils import compare_tensors
from sglang_omni.client.client import Client
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    GraphedLayerStack,
    Qwen3OmniAudioEncoder,
)
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto.request import StagePayload
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest


def capture_layer(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    *,
    snapshots: list[torch.Tensor],
    input_snapshots: list[torch.Tensor] | None = None,
) -> None:
    """The fixed hook signature includes the module and its positional inputs."""
    tensor = output[0] if isinstance(output, tuple) else output
    snapshots.append(tensor.detach().clone())
    if input_snapshots is not None:
        input_snapshots.append(inputs[0].detach().clone())
    else:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    processor = Qwen3OmniPreprocessor(
        arguments.model_path, max_seq_len=32768, video_decode_threads=1
    )
    request = ChatCompletionRequest.model_validate_json(
        arguments.request_file.read_text()
    )
    payload = StagePayload(
        request_id="encoder-audit",
        request=Client.build_omni_request(build_chat_generate_request(request)),
        data={},
    )
    payload = asyncio.run(processor(payload))
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    audio_inputs = state.encoder_inputs["audio_encoder"]
    features = audio_inputs["input_features"]
    mask = audio_inputs["feature_attention_mask"]
    assert isinstance(features, torch.Tensor) and isinstance(mask, torch.Tensor)
    packed = (
        features.transpose(1, 2)[mask.bool()].T.contiguous().to(device, torch.bfloat16)
    )
    lengths = mask.sum(-1).to(device)
    reference_audio, loading = Qwen3OmniMoeAudioEncoder.from_pretrained(
        arguments.model_path,
        config=config.thinker_config.audio_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        key_mapping={r"^thinker\.audio_tower\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference_audio.eval()
    native_audio = Qwen3OmniAudioEncoder(
        arguments.model_path,
        device=str(device),
        dtype=torch.bfloat16,
        enable_layer_cuda_graph=False,
    ).eval()
    for name, parameter in reference_audio.named_parameters():
        assert torch.equal(
            parameter, native_audio.audio_tower.get_parameter(name)
        ), name
    audio_reference_layers: list[torch.Tensor] = []
    audio_native_layers: list[torch.Tensor] = []
    hooks = [
        layer.register_forward_hook(
            partial(capture_layer, snapshots=audio_reference_layers)
        )
        for layer in reference_audio.layers
    ]
    hooks.extend(
        layer.register_forward_hook(
            partial(capture_layer, snapshots=audio_native_layers)
        )
        for layer in native_audio.audio_tower.layers
    )
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        expected_audio = reference_audio(packed, feature_lens=lengths).last_hidden_state
        actual_audio = native_audio(
            input_features=features, feature_attention_mask=mask
        )
    for hook in hooks:
        hook.remove()
    assert (
        len(audio_reference_layers)
        == len(audio_native_layers)
        == config.thinker_config.audio_config.encoder_layers
    )
    assert (
        int(actual_audio["audio_output_lengths"].sum())
        == actual_audio["audio_embeds"].shape[0]
    )
    audio_report = {
        "input_source": str(arguments.request_file),
        "reference_source_sha256": hashlib.sha256(
            Path(inspect.getfile(type(reference_audio))).read_bytes()
        ).hexdigest(),
        "output_error": asdict(
            compare_tensors(actual_audio["audio_embeds"], expected_audio)
        ),
        "layers": [
            asdict(compare_tensors(actual, expected))
            for actual, expected in zip(
                audio_native_layers, audio_reference_layers, strict=True
            )
        ],
    }
    native_audio.enable_layer_cuda_graph()
    runner = native_audio.layer_graph_runner
    assert runner is not None and runner.has_graphs
    replay_results: list[bool] = []
    replay_buckets: list[int] = []
    maybe_replay = runner.maybe_replay

    def record_replay(
        hidden_states: torch.Tensor,
        cumulative_sequence_lengths: torch.Tensor,
        segments: list[int],
    ) -> torch.Tensor | None:
        output = maybe_replay(hidden_states, cumulative_sequence_lengths, segments)
        replay_results.append(output is not None)
        bucket = runner.select(hidden_states.shape[0], segments)
        assert bucket is not None
        replay_buckets.append(bucket)
        return output

    with torch.inference_mode(), patch.object(runner, "maybe_replay", record_replay):
        graphed_audio = native_audio(
            input_features=features, feature_attention_mask=mask
        )
    assert replay_results == [True]
    audio_report["graph_replay_executed"] = True
    audio_report["graph_vs_eager_error"] = asdict(
        compare_tensors(graphed_audio["audio_embeds"], actual_audio["audio_embeds"])
    )
    captured_graph = runner.graphs[replay_buckets[0]]
    graph_stack = native_audio.audio_tower.layers
    assert isinstance(graph_stack, GraphedLayerStack)
    with torch.inference_mode():
        with patch.object(native_audio.audio_tower, "layers", graph_stack.layers):
            padded_eager = runner.run_layers(
                captured_graph.hidden_states.clone(),
                captured_graph.cu_seqlens.clone(),
                runner.window,
            )
        repeated_graph = native_audio(
            input_features=features, feature_attention_mask=mask
        )
    audio_report["graph_bucket_tokens"] = replay_buckets[0]
    audio_report["graph_stack_vs_same_padded_eager"] = asdict(
        compare_tensors(captured_graph.output, padded_eager)
    )
    audio_report["graph_repeat_error"] = asdict(
        compare_tensors(repeated_graph["audio_embeds"], graphed_audio["audio_embeds"])
    )
    (arguments.output_directory / "audio.json").write_text(
        json.dumps(audio_report, indent=2)
    )
    reference_vision, loading = Qwen3OmniMoeVisionEncoder.from_pretrained(
        arguments.model_path,
        config=config.thinker_config.vision_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        key_mapping={r"^thinker\.visual\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference_vision.eval()
    native_vision = Qwen3OmniImageEncoder(
        arguments.model_path, device=str(device), dtype=torch.bfloat16
    ).eval()
    for name, parameter in reference_vision.named_parameters():
        assert torch.equal(parameter, native_vision.visual.get_parameter(name)), name
    rows, columns = np.indices((56, 84))
    pixels = np.stack(
        ((rows * 3) % 256, (columns * 2) % 256, (rows + columns) % 256), axis=-1
    ).astype(np.uint8)
    image = Image.fromarray(pixels)
    image_inputs = processor.processor.image_processor(
        images=[image], return_tensors="pt"
    )
    video_inputs = processor.processor.video_processor(
        videos=[np.stack((pixels, pixels[:, ::-1].copy()))],
        fps=2,
        return_tensors="pt",
        device="cpu",
        do_sample_frames=False,
    )
    vision_reports = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for modality, model_inputs in (
            ("image", image_inputs),
            ("video", video_inputs),
        ):
            pixel_name = (
                "pixel_values" if modality == "image" else "pixel_values_videos"
            )
            grid_name = "image_grid_thw" if modality == "image" else "video_grid_thw"
            vision_reference_layers: list[torch.Tensor] = []
            vision_native_layers: list[torch.Tensor] = []
            reference_layer_inputs: list[torch.Tensor] = []
            native_layer_inputs: list[torch.Tensor] = []
            hooks = [
                block.register_forward_hook(
                    partial(
                        capture_layer,
                        snapshots=vision_reference_layers,
                        input_snapshots=reference_layer_inputs,
                    )
                )
                for block in reference_vision.blocks
            ]
            hooks.extend(
                block.register_forward_hook(
                    partial(
                        capture_layer,
                        snapshots=vision_native_layers,
                        input_snapshots=native_layer_inputs,
                    )
                )
                for block in native_vision.visual.blocks
            )
            expected = reference_vision(
                model_inputs[pixel_name].to(device, torch.bfloat16),
                grid_thw=model_inputs[grid_name].to(device),
                return_dict=True,
            )
            actual = native_vision(
                **{
                    pixel_name: model_inputs[pixel_name],
                    grid_name: model_inputs[grid_name],
                }
            )
            for hook in hooks:
                hook.remove()
            assert (
                len(vision_reference_layers)
                == len(vision_native_layers)
                == config.thinker_config.vision_config.depth
            )
            actual_features = actual[f"deepstack_visual_embeds_{modality}"]
            assert actual_features is not None
            vision_reports.append(
                {
                    "modality": modality,
                    "grid": model_inputs[grid_name].tolist(),
                    "patch_embedding_error": asdict(
                        compare_tensors(
                            native_vision.visual.patch_embed(
                                model_inputs[pixel_name].to(device, torch.bfloat16)
                            ),
                            reference_vision.patch_embed(
                                model_inputs[pixel_name].to(device, torch.bfloat16)
                            ),
                        )
                    ),
                    "layer_input_errors": [
                        asdict(compare_tensors(actual_input, reference_input))
                        for actual_input, reference_input in zip(
                            native_layer_inputs, reference_layer_inputs, strict=True
                        )
                    ],
                    "output_error": asdict(
                        compare_tensors(
                            actual[f"{modality}_embeds"], expected.pooler_output
                        )
                    ),
                    "layers": [
                        asdict(compare_tensors(actual_layer, reference_layer))
                        for actual_layer, reference_layer in zip(
                            vision_native_layers, vision_reference_layers, strict=True
                        )
                    ],
                    "deepstack_errors": [
                        asdict(compare_tensors(actual_feature, reference_feature))
                        for actual_feature, reference_feature in zip(
                            actual_features, expected.deepstack_features, strict=True
                        )
                    ],
                }
            )
    summary = {
        "model_path": arguments.model_path,
        "gpu": torch.cuda.get_device_name(device),
        "transformers": version("transformers"),
        "reference_attention": "sdpa_math",
        "reference_hub_kernels": False,
        "audio": audio_report,
        "vision": vision_reports,
        "scope": "Checkpoint layer and output numerics for one real audio request and synthetic image/video encoder fixtures. This does not certify task accuracy, mixed request routing, temporal media decode or full model quality.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        json.dumps(
            {
                "audio_layers": len(audio_report["layers"]),
                "vision_layers": [len(report["layers"]) for report in vision_reports],
                "output_directory": str(arguments.output_directory),
            }
        )
    )


if __name__ == "__main__":
    main()
