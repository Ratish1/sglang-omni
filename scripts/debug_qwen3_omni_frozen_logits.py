"""Teacher-force captured Talker inputs with reference or native KV history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import DynamicCache, Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


class ReferenceStateHook:
    def __init__(self, snapshots: dict[str, torch.Tensor], name: str) -> None:
        self.snapshots = snapshots
        self.name = name

    def tensor_output(
        self,
        module: torch.nn.Module,
        arguments: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        self.snapshots[self.name] = output[0, -1].clone()

    def attention_output(
        self,
        module: torch.nn.Module,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        self.snapshots[self.name] = output[0][0, -1].clone()

    def router_output(
        self,
        module: torch.nn.Module,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        for name, tensor in zip(("logits", "weights", "experts"), output, strict=True):
            self.snapshots[f"{self.name}_{name}"] = tensor[-1].clone()


def native_cache(
    snapshot: dict[str, torch.Tensor],
    config: Qwen3OmniMoeTalkerTextConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> DynamicCache:
    cache = DynamicCache(config=config)
    for layer_index in range(config.num_hidden_layers):
        keys = snapshot[f"layer_{layer_index}_keys"].transpose(0, 1).unsqueeze(0)
        values = snapshot[f"layer_{layer_index}_values"].transpose(0, 1).unsqueeze(0)
        cache.update(keys.to(device, dtype), values.to(device, dtype), layer_index)
    return cache


def probabilities(
    logits: torch.Tensor,
    repeated: torch.Tensor,
    allowed: torch.Tensor,
    temperature: float,
    repetition_penalty: float,
) -> torch.Tensor:
    penalized = torch.where(
        logits > 0, logits / repetition_penalty, logits * repetition_penalty
    )
    logits = torch.where(repeated, penalized, logits).masked_fill(~allowed, -torch.inf)
    return torch.softmax(logits / temperature, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    if arguments.dtype == "float32":
        reference.float()
    else:
        pass
    boundary_snapshots: dict[str, torch.Tensor] = {}
    for layer_index, layer in enumerate(reference.model.layers):
        for name, module in (
            ("attention_input", layer.input_layernorm),
            ("moe_input", layer.post_attention_layernorm),
            ("moe_output", layer.mlp),
            ("layer_output", layer),
        ):
            module.register_forward_hook(
                ReferenceStateHook(
                    boundary_snapshots, f"layer_{layer_index}_{name}"
                ).tensor_output
            )
        layer.self_attn.register_forward_hook(
            ReferenceStateHook(
                boundary_snapshots, f"layer_{layer_index}_attention_projection"
            ).attention_output
        )
        layer.mlp.gate.register_forward_hook(
            ReferenceStateHook(
                boundary_snapshots, f"layer_{layer_index}_router"
            ).router_output
        )
    reports = []
    with torch.inference_mode():
        for request_id in arguments.request_id:
            capture = None
            for filename in arguments.capture_directory.glob("request-*.pt"):
                candidate = torch.load(filename, map_location="cpu", weights_only=True)
                if candidate["request_id"] == request_id:
                    capture = candidate
                    break
                else:
                    pass
            assert capture is not None, request_id
            conditioning = torch.load(
                arguments.capture_directory
                / "talker_state"
                / f"{request_id}.conditioning.pt",
                map_location="cpu",
                weights_only=True,
            )
            codes = torch.load(
                arguments.capture_directory / "talker_state" / f"{request_id}.pt",
                map_location=reference.device,
                weights_only=True,
            )
            snapshots = capture["snapshots"]
            prefill = snapshots[0]["prefill_input_embeddings"].to(
                reference.device, reference.dtype
            )
            positions = snapshots[0]["prefill_positions"].to(reference.device)
            assert torch.equal(
                positions, torch.arange(len(prefill), device=reference.device)
            )
            native_hidden = conditioning["native_backbone_hidden_states"].to(
                reference.device
            )
            native_logits = (
                conditioning["native_codec_logits"].to(reference.device).float()
            )
            decode_inputs = conditioning["native_decode_input_embeddings"].to(
                reference.device, reference.dtype
            )
            frame_count = len(native_hidden)
            assert len(decode_inputs) == frame_count - 1
            top_k = torch.tensor([conditioning["top_k"]], device=reference.device)
            top_p = torch.tensor([conditioning["top_p"]], device=reference.device)
            minimum_probability = torch.zeros(1, device=reference.device)
            seed = torch.tensor(
                [conditioning["sampling_seed"]], device=reference.device
            )
            token_ids = torch.arange(native_logits.shape[-1], device=reference.device)
            allowed = (token_ids < config.code_predictor_config.vocab_size) | (
                token_ids == config.codec_eos_token_id
            )
            prefix_control = None
            for history in ("reference_past", "native_past"):
                cache = None
                repeated = torch.zeros(
                    (1, len(token_ids)), dtype=torch.bool, device=reference.device
                )
                for frame_index in range(frame_count):
                    if frame_index == 0:
                        embeddings = prefill[None]
                        frame_positions = positions
                    else:
                        embeddings = decode_inputs[frame_index - 1].view(1, 1, -1)
                        frame_positions = torch.tensor(
                            [len(prefill) + frame_index - 1], device=reference.device
                        )
                        if history == "native_past":
                            cache = native_cache(
                                snapshots[frame_index - 1],
                                config.text_config,
                                reference.dtype,
                                reference.device,
                            )
                        else:
                            pass
                    output = reference.model(
                        inputs_embeds=embeddings,
                        position_ids=frame_positions.view(1, 1, -1).expand(4, 1, -1),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    assert cache.get_seq_length() == len(prefill) + frame_index
                    if frame_index == 0:
                        if prefix_control is None:
                            prefix_control = output.last_hidden_state.clone()
                        else:
                            assert torch.equal(prefix_control, output.last_hidden_state)
                    else:
                        pass
                    expected_hidden = output.last_hidden_state[:, -1]
                    expected_logits = reference.codec_head(expected_hidden).float()
                    actual_hidden = native_hidden[frame_index : frame_index + 1]
                    actual_logits = native_logits[frame_index : frame_index + 1]
                    reprojected = reference.codec_head(
                        actual_hidden.to(reference.dtype)
                    ).float()
                    head_exact = torch.equal(reprojected, actual_logits)
                    if arguments.dtype == "bfloat16":
                        assert head_exact, (request_id, frame_index)
                    else:
                        pass
                    distributions = [
                        probabilities(
                            logits,
                            repeated,
                            allowed,
                            conditioning["temperature"],
                            conditioning["repetition_penalty"],
                        )
                        for logits in (actual_logits, expected_logits)
                    ]
                    choices = [
                        int(
                            top_k_top_p_min_p_sampling_from_probs_torch(
                                distribution,
                                top_k,
                                top_p,
                                minimum_probability,
                                False,
                                seed,
                                frame_positions[-1:],
                            )[0]
                        )
                        for distribution in distributions
                    ]
                    assert choices[0] == int(codes[frame_index, 0]), (
                        request_id,
                        frame_index,
                        choices[0],
                        int(codes[frame_index, 0]),
                    )
                    report = {
                        "request_id": request_id,
                        "history": history,
                        "reference_dtype": arguments.dtype,
                        "frame": frame_index,
                        "head_reprojection_exact": head_exact,
                        "hidden_relative_l2": float(
                            (actual_hidden.float() - expected_hidden.float()).norm()
                            / expected_hidden.float().norm()
                        ),
                        "logits_relative_l2": float(
                            (actual_logits - expected_logits).norm()
                            / expected_logits.norm()
                        ),
                        "probability_total_variation": float(
                            (distributions[0] - distributions[1]).abs().sum() / 2
                        ),
                        "native_token_id": choices[0],
                        "reference_token_id": choices[1],
                        "native_token_reference_probability": float(
                            distributions[1][0, choices[0]]
                        ),
                    }
                    reports.append(report)
                    assert (
                        len(boundary_snapshots)
                        == 8 * config.text_config.num_hidden_layers
                    )
                    torch.save(
                        {
                            "reference_boundaries": {
                                name: tensor.cpu()
                                for name, tensor in boundary_snapshots.items()
                            },
                            "native_logits": actual_logits.cpu(),
                            "reference_logits": expected_logits.cpu(),
                            "native_hidden_states": actual_hidden.cpu(),
                            "reference_hidden_states": expected_hidden.cpu(),
                            "repetition_mask": repeated.cpu(),
                            "seed": seed.cpu(),
                            "positions": frame_positions[-1:].cpu(),
                            "temperature": conditioning["temperature"],
                            "repetition_penalty": conditioning["repetition_penalty"],
                            "top_k": conditioning["top_k"],
                            "top_p": conditioning["top_p"],
                        },
                        arguments.output_directory
                        / f"{request_id}.{history}.{frame_index}.pt",
                    )
                    arguments.output_directory.joinpath("summary.json").write_text(
                        json.dumps(reports, indent=2)
                    )
                    print(json.dumps(report), flush=True)
                    repeated[:, codes[frame_index, 0]] = True


if __name__ == "__main__":
    main()
