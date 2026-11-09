"""Replay actual encoder batches and compare frontend, eager and graph boundaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    Qwen3OmniAudioEncoder,
    pack_padded_audio_features,
)
from sglang_omni.serve.protocol import ChatCompletionRequest


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
        native_mask = actual["feature_attention_mask"].bool()
        reference_mask = inputs["feature_attention_mask"].bool()
        native_features = actual["input_features"].permute(0, 2, 1)[native_mask].T
        reference_features = inputs["input_features"].permute(0, 2, 1)[reference_mask].T
        reports["frontend"].append(
            {
                "request_id": request_id,
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
    batch_paths = sorted((arguments.capture_directory / "audio-forwards").glob("*.pt"))
    assert batch_paths
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
        else:
            pass
        print(json.dumps(report), flush=True)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )


if __name__ == "__main__":
    main()
