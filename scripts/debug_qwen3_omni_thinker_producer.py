"""Compare actual embedded Thinker inputs with the checkpoint hidden producer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import DynamicCache, Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--compare-native-layers", action="store_true")
    parser.add_argument("--hold-native-prefix-cache", action="store_true")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    input_files = sorted(
        (arguments.delivery_directory / "inputs").glob(f"{arguments.request_id}-*.pt")
    )
    assert len(input_files) == 1, "this comparison requires one complete native prefill"
    captured_input = torch.load(input_files[0], weights_only=True)
    assert captured_input["request_id"] == arguments.request_id
    prefix_token_count = captured_input["prefix_length"]
    captured_writes = [
        torch.load(filename, weights_only=True)
        for filename in sorted((arguments.delivery_directory / "writes").glob("*.pt"))
    ]
    selected_writes = [
        written
        for written in captured_writes
        if written["request_id"] == arguments.request_id
        and written["forward_step"] == captured_input["forward_step"]
    ]
    assert len(selected_writes) == 1
    captured_output = selected_writes[0]
    assert captured_output["prefix_length"] == prefix_token_count
    assert torch.equal(
        captured_input["actual_input_ids"], captured_output["actual_input_ids"]
    )
    assert torch.equal(
        captured_input["causal_token_ids"], captured_output["causal_token_ids"]
    )
    assert captured_input["extend_length"] == len(captured_input["embedded_input"])
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    reference, loading = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=configuration.thinker_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    audit_checkpoint_weights(
        reference,
        Path(
            hf_hub_download(
                arguments.model_path, "config.json", revision=arguments.model_revision
            )
        ).parent,
        arguments.output_directory / "weight-audit.json",
    )
    expected_rotary = Qwen3OmniMoeThinkerTextRotaryEmbedding(reference.model.config)
    for buffer_name in ("inv_freq", "original_inv_freq"):
        actual = dict(reference.model.rotary_emb.named_buffers())[buffer_name].cpu()
        expected = dict(expected_rotary.named_buffers())[buffer_name]
        assert actual.dtype == expected.dtype == torch.float32
        assert torch.equal(actual, expected)
    embedded_input = captured_input["embedded_input"].to(reference.device)
    positions = captured_input["mrope_positions"].to(reference.device)
    assert positions.shape == (3, len(embedded_input))
    causal_token_ids = captured_input["causal_token_ids"].to(reference.device)
    assert torch.equal(
        causal_token_ids[prefix_token_count:].cpu(), captured_input["actual_input_ids"]
    )
    prefix_token_ids = causal_token_ids[:prefix_token_count]
    multimodal_token_ids = {
        configuration.thinker_config.audio_token_id,
        configuration.thinker_config.image_token_id,
        configuration.thinker_config.video_token_id,
    }
    assert not any(token in multimodal_token_ids for token in prefix_token_ids.tolist())
    # note (ratish): recomputing text-only prefix embeddings does not validate native cached KV.
    prefix_embeddings = reference.model.embed_tokens(prefix_token_ids)
    embedded_input = torch.cat((prefix_embeddings, embedded_input))
    prefix_positions = torch.arange(prefix_token_count, device=reference.device).expand(
        3, -1
    )
    positions = torch.cat((prefix_positions, positions), dim=1)
    outputs = reference.model(
        inputs_embeds=embedded_input.unsqueeze(0),
        position_ids=positions.unsqueeze(1),
        use_cache=False,
        output_hidden_states=True,
    )
    accepted_layer = configuration.talker_config.accept_hidden_layer
    assert 0 < accepted_layer < reference.model.config.num_hidden_layers
    assert torch.equal(outputs.hidden_states[0][0], embedded_input)
    accepted = outputs.hidden_states[accepted_layer][0]
    selected_positions = captured_output["prompt_positions"].long().to(reference.device)
    if arguments.compare_native_layers:
        native_layers = captured_output["layer_hidden"]
        assert set(native_layers) == {
            f"layer_{layer_index:02d}_hidden"
            for layer_index in range(accepted_layer + 1)
        }
        assert torch.equal(
            native_layers["layer_00_hidden"], captured_input["embedded_input"]
        )
        assert torch.equal(
            native_layers[f"layer_{accepted_layer:02d}_hidden"],
            captured_output["accepted_full_hidden"],
        )
        layer_comparisons = [
            {
                "layer_index": layer_index,
                "extended_rows": asdict(
                    compare_tensors(
                        native_layers[f"layer_{layer_index:02d}_hidden"].to(
                            reference.device
                        ),
                        outputs.hidden_states[layer_index][0, prefix_token_count:],
                    )
                ),
                "audio_rows": asdict(
                    compare_tensors(
                        native_layers[f"layer_{layer_index:02d}_hidden"].to(
                            reference.device
                        )[selected_positions - prefix_token_count],
                        outputs.hidden_states[layer_index][0, selected_positions],
                    )
                ),
            }
            for layer_index in range(accepted_layer + 1)
        ]
        (arguments.output_directory / "layer-comparisons.json").write_text(
            json.dumps(layer_comparisons, indent=2)
        )
    else:
        pass
    if arguments.hold_native_prefix_cache:
        assert arguments.compare_native_layers
        native_keys = captured_output["cache_keys"]
        native_values = captured_output["cache_values"]
        assert len(native_keys) == len(native_values) == accepted_layer + 1
        assert all(len(keys) == len(embedded_input) for keys in native_keys)
        cache = DynamicCache(config=reference.model.config)
        for layer_index in range(accepted_layer):
            cache.update(
                native_keys[layer_index][:prefix_token_count]
                .to(reference.device)
                .transpose(0, 1)
                .unsqueeze(0),
                native_values[layer_index][:prefix_token_count]
                .to(reference.device)
                .transpose(0, 1)
                .unsqueeze(0),
                layer_index,
            )
        query_token_count = len(captured_input["embedded_input"])
        query_positions = prefix_token_count + torch.arange(
            query_token_count, device=reference.device
        )
        key_positions = torch.arange(len(embedded_input), device=reference.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        attention_mask = torch.zeros(
            (1, 1, query_token_count, len(embedded_input)),
            dtype=reference.dtype,
            device=reference.device,
        ).masked_fill_(~allowed[None, None], float("-inf"))
        position_embeddings = reference.model.rotary_emb(
            embedded_input[None, prefix_token_count:],
            positions[:, None, prefix_token_count:],
        )
        local_comparisons = []
        local_outputs: dict[str, torch.Tensor] = {}
        for layer_index in range(accepted_layer):
            native_input = captured_output["layer_hidden"][
                f"layer_{layer_index:02d}_hidden"
            ].to(reference.device)
            output = reference.model.layers[layer_index](
                native_input.unsqueeze(0),
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )[0]
            key_states = cache.layers[layer_index].keys
            value_states = cache.layers[layer_index].values
            assert key_states is not None and value_states is not None
            key_states = key_states[0].transpose(0, 1)
            value_states = value_states[0].transpose(0, 1)
            assert torch.equal(
                key_states[:prefix_token_count].cpu(),
                native_keys[layer_index][:prefix_token_count],
            )
            assert torch.equal(
                value_states[:prefix_token_count].cpu(),
                native_values[layer_index][:prefix_token_count],
            )
            local_comparisons.append(
                {
                    "layer_index": layer_index,
                    "output": asdict(
                        compare_tensors(
                            captured_output["layer_hidden"][
                                f"layer_{layer_index + 1:02d}_hidden"
                            ].to(reference.device),
                            output,
                        )
                    ),
                    "keys": asdict(
                        compare_tensors(
                            native_keys[layer_index][prefix_token_count:].to(
                                reference.device
                            ),
                            key_states[prefix_token_count:],
                        )
                    ),
                    "values": asdict(
                        compare_tensors(
                            native_values[layer_index][prefix_token_count:].to(
                                reference.device
                            ),
                            value_states[prefix_token_count:],
                        )
                    ),
                }
            )
            local_outputs[f"layer_{layer_index:02d}_output"] = output.cpu()
        (
            arguments.output_directory / "native-prefix-local-comparisons.json"
        ).write_text(json.dumps(local_comparisons, indent=2))
        torch.save(local_outputs, arguments.output_directory / "local-layer-outputs.pt")
    else:
        pass
    report = {
        "request_id": arguments.request_id,
        "model_revision": arguments.model_revision,
        "native_prefix_length": captured_input["prefix_length"],
        "prompt_rows": len(embedded_input),
        "accepted_layer": accepted_layer,
        "multimodal_rows": len(selected_positions),
        "native_prefix_cache_held_for_local_controls": arguments.hold_native_prefix_cache,
        "native_extend": asdict(
            compare_tensors(
                captured_output["accepted_full_hidden"].to(reference.device),
                accepted[prefix_token_count:],
            )
        ),
        "multimodal_rows_comparison": asdict(
            compare_tensors(
                captured_output["producer_hidden"].to(reference.device),
                accepted[selected_positions],
            )
        ),
        "scope": "The full-prefill comparison uses actual native extended embeddings and positions, with a text-only prefix recomputed from checkpoint embeddings rather than held native prefix KV. Optional local controls hold each native layer input and its native prefix KV. Encoder output is held. Numerical differences do not by themselves identify an invalid native operator.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    torch.save(
        {
            "embedded_input": embedded_input.cpu(),
            "mrope_positions": positions.cpu(),
            "native_accepted_hidden": captured_output["accepted_full_hidden"],
            **{
                f"checkpoint_hidden_{layer}": hidden[0].cpu()
                for layer, hidden in enumerate(
                    outputs.hidden_states[: accepted_layer + 1]
                )
            },
        },
        arguments.output_directory / "producer-operands.pt",
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
