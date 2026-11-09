"""Replay actual encoder batches and compare frontend, eager and graph boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

from scripts.debug_qwen3_omni_feedback_audit import TensorDifference, compare_tensors
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    Qwen3OmniAudioEncoder,
    pack_padded_audio_features,
)
from sglang_omni.models.qwen3_omni.components.audio_layer_graph import (
    AudioLayerGraphRunner,
)
from sglang_omni.serve.protocol import ChatCompletionRequest


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@dataclass(kw_only=True)
class PaddingLayerDifference:
    layer_index: int
    padding_values_isolated: bool
    same_projections_attention_shape: TensorDifference
    padded_attention_to_float64: TensorDifference
    unpadded_attention_to_float64: TensorDifference
    local_layer_shape: TensorDifference
    cumulative_layer_shape: TensorDifference


def audit_padding(
    runner: AudioLayerGraphRunner,
    padded_hidden_states: torch.Tensor,
    padded_cumulative_sequence_lengths: torch.Tensor,
    cumulative_sequence_lengths: torch.Tensor,
    valid_token_count: int,
) -> list[PaddingLayerDifference]:
    padded_metadata = prepare_vision_attention_metadata(
        padded_cumulative_sequence_lengths,
        padded_cumulative_sequence_lengths.device,
        max_seqlen=runner.window,
    )
    metadata = prepare_vision_attention_metadata(
        cumulative_sequence_lengths,
        cumulative_sequence_lengths.device,
        max_seqlen=runner.window,
    )
    boundaries = cumulative_sequence_lengths.cpu().tolist()
    unpadded_hidden_states = padded_hidden_states[:valid_token_count].clone()
    layers = runner.tower.layers
    differences: list[PaddingLayerDifference] = []
    for layer_index, layer in enumerate(layers):
        attention = layer.self_attn
        projection = attention.qkv_proj(
            layer.self_attn_layer_norm(padded_hidden_states)
        ).reshape(len(padded_hidden_states), 3, attention.num_heads, -1)
        changed_padding = projection.clone()
        changed_padding[valid_token_count:] = 100
        outputs: dict[str, torch.Tensor] = {}
        for name, projected, attention_metadata in (
            ("padded", projection, padded_metadata),
            ("unpadded", projection[:valid_token_count], metadata),
            ("changed_padding", changed_padding, padded_metadata),
        ):
            query, key, value = projected.unbind(1)
            outputs[name] = runner.packed_attention(
                query,
                key,
                value,
                attention_metadata.cu_seqlens,
                bsz=1,
                seq_len=len(projected),
                softmax_scale=attention.scaling,
                forward_metadata=attention_metadata,
            )[:valid_token_count]
        expected: list[torch.Tensor] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            query, key, value = projection[start:end].unbind(1)
            query = query.transpose(0, 1).double()
            key = key.transpose(0, 1).double()
            value = value.transpose(0, 1).double()
            probabilities = torch.softmax(
                (query @ key.transpose(-1, -2)) * attention.scaling, dim=-1
            )
            expected.append((probabilities @ value).transpose(0, 1))
        reference = torch.cat(expected)
        runner.tower.layers = torch.nn.ModuleList([layer])
        same_input = runner.run_layers(
            padded_hidden_states[:valid_token_count],
            cumulative_sequence_lengths,
            runner.window,
        )
        next_padded = runner.run_layers(
            padded_hidden_states, padded_cumulative_sequence_lengths, runner.window
        )
        unpadded_hidden_states = runner.run_layers(
            unpadded_hidden_states, cumulative_sequence_lengths, runner.window
        )
        differences.append(
            PaddingLayerDifference(
                layer_index=layer_index,
                padding_values_isolated=torch.equal(
                    outputs["padded"], outputs["changed_padding"]
                ),
                same_projections_attention_shape=compare_tensors(
                    outputs["padded"], outputs["unpadded"]
                ),
                padded_attention_to_float64=compare_tensors(
                    outputs["padded"], reference
                ),
                unpadded_attention_to_float64=compare_tensors(
                    outputs["unpadded"], reference
                ),
                local_layer_shape=compare_tensors(
                    next_padded[:valid_token_count], same_input
                ),
                cumulative_layer_shape=compare_tensors(
                    next_padded[:valid_token_count], unpadded_hidden_states
                ),
            )
        )
        padded_hidden_states = next_padded
    runner.tower.layers = layers
    return differences


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--audit-padding", action="store_true")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    reference, loading = Qwen3OmniMoeAudioEncoder.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
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
    reference.eval()
    checkpoint_directory = Path(
        hf_hub_download(
            arguments.model_path, "config.json", revision=arguments.model_revision
        )
    ).parent
    native = Qwen3OmniAudioEncoder(
        str(checkpoint_directory),
        device=str(device),
        dtype=torch.bfloat16,
        enable_layer_cuda_graph=False,
    ).eval()
    native_parameters = dict(native.audio_tower.named_parameters())
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter, native_parameters[name]), name
    reports = {
        "model_revision": arguments.model_revision,
        "reference_parameters_exact": len(list(reference.parameters())),
        "positional_buffer": asdict(
            compare_tensors(
                native.audio_tower.positional_embedding.positional_embedding,
                reference.positional_embedding.positional_embedding.to(torch.bfloat16),
            )
        ),
        "frontend": [],
        "batches": [],
    }
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    batch_paths = sorted((arguments.capture_directory / "audio-forwards").glob("*.pt"))
    assert batch_paths
    sources: dict[str, list[tuple[torch.Tensor, str, int]]] = {}
    for batch_path in batch_paths:
        actual = torch.load(batch_path, weights_only=True)
        cursor = 0
        for row, output_rows in enumerate(actual["audio_output_lengths"].tolist()):
            signature = tensor_digest(
                actual["audio_embeds"][cursor : cursor + output_rows]
            )
            mask = actual["feature_attention_mask"][row].bool()
            features = actual["input_features"][row][:, mask]
            sources.setdefault(signature, []).append((features, batch_path.name, row))
            cursor += output_rows
        assert cursor == len(actual["audio_embeds"])
    for request_id in arguments.request_id:
        request = ChatCompletionRequest.model_validate_json(
            (arguments.request_directory / f"{request_id}.request.json").read_bytes()
        )
        assert request.audios and len(request.audios) == 1
        messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
        assert messages[-1]["role"] == "user"
        assert isinstance(messages[-1]["content"], str)
        messages[-1]["content"] = [
            {"type": "audio", "path": request.audios[0]},
            {"type": "text", "text": messages[-1]["content"]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        actual = torch.load(
            arguments.capture_directory / request_id / "audio-conditioning.pt",
            weights_only=True,
        )
        reference_mask = inputs["feature_attention_mask"].bool()
        matches = sources[tensor_digest(actual["audio_embeds"])]
        native_features, source_batch, source_row = matches[0]
        assert all(torch.equal(native_features, features) for features, _, _ in matches)
        reference_features = inputs["input_features"].permute(0, 2, 1)[reference_mask].T
        reports["frontend"].append(
            {
                "request_id": request_id,
                "source_batch": source_batch,
                "source_row": source_row,
                "features": asdict(
                    compare_tensors(native_features, reference_features)
                ),
                "encoder_input_dtype": asdict(
                    compare_tensors(
                        native_features.to(torch.bfloat16),
                        reference_features.to(torch.bfloat16),
                    )
                ),
            }
        )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(reports, indent=2)
    )
    for batch_path in batch_paths:
        actual = torch.load(batch_path, weights_only=True)
        features = actual["input_features"].to(device, dtype=torch.bfloat16)
        mask = actual["feature_attention_mask"].to(device)
        lengths = mask.sum(dim=1)
        packed = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
        assert torch.equal(packed, pack_padded_audio_features(features, mask, lengths))
        checkpoint_output = reference(packed, feature_lens=lengths).last_hidden_state
        eager_output = native(input_features=features, feature_attention_mask=mask)[
            "audio_embeds"
        ]
        torch.save(
            {"checkpoint": checkpoint_output.cpu(), "eager": eager_output.cpu()},
            arguments.output_directory / batch_path.name,
        )
        reports["batches"].append(
            {
                "filename": batch_path.name,
                "clips": len(lengths),
                "lengths": lengths.cpu().tolist(),
                "served_to_checkpoint": asdict(
                    compare_tensors(
                        actual["audio_embeds"].to(device), checkpoint_output
                    )
                ),
                "served_to_eager": asdict(
                    compare_tensors(actual["audio_embeds"].to(device), eager_output)
                ),
                "eager_to_checkpoint": asdict(
                    compare_tensors(eager_output, checkpoint_output)
                ),
            }
        )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(reports, indent=2)
    )
    native.enable_layer_cuda_graph()
    runner = native.layer_graph_runner
    assert runner is not None and runner.has_graphs
    for batch_path, report in zip(batch_paths, reports["batches"], strict=True):
        actual = torch.load(batch_path, weights_only=True)
        features = actual["input_features"].to(device, dtype=torch.bfloat16)
        mask = actual["feature_attention_mask"].to(device)
        graphed_output = native(input_features=features, feature_attention_mask=mask)[
            "audio_embeds"
        ]
        controls = torch.load(
            arguments.output_directory / batch_path.name, weights_only=True
        )
        segments = actual["cu_seqlens"].diff().tolist()
        bucket = runner.select(len(graphed_output), segments)
        report["bucket"] = bucket
        report["graph_to_eager"] = asdict(
            compare_tensors(graphed_output, controls["eager"].to(device))
        )
        report["served_to_graph"] = asdict(
            compare_tensors(actual["audio_embeds"].to(device), graphed_output)
        )
        if bucket is not None:
            captured = runner.graphs[bucket]
            wrapper = native.audio_tower.layers
            native.audio_tower.layers = wrapper.layers
            replayed = captured.output.clone()
            ordinary = runner.run_layers(
                captured.hidden_states.clone(),
                captured.cu_seqlens.clone(),
                runner.window,
            )
            native.audio_tower.layers = wrapper
            report["same_padded_operands_graph_to_eager"] = asdict(
                compare_tensors(replayed, ordinary)
            )
            if arguments.audit_padding:
                report["padding_audit"] = [
                    asdict(difference)
                    for difference in audit_padding(
                        runner,
                        captured.hidden_states.clone(),
                        captured.cu_seqlens.clone(),
                        actual["cu_seqlens"].to(device),
                        len(graphed_output),
                    )
                ]
            else:
                pass
        else:
            pass
        print(json.dumps(report), flush=True)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )


if __name__ == "__main__":
    main()
