"""Compare actual residual-code logits using frozen native backbone rows."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config
from sglang_omni.vendor.sglang.layers import ReplicatedLinear


@dataclass(kw_only=True)
class PredictorLogits:
    tensors: list[torch.Tensor] = field(default_factory=list)
    hidden_states: list[torch.Tensor] = field(default_factory=list)
    unrounded_logits: list[torch.Tensor] = field(default_factory=list)

    def record(
        self,
        module: ReplicatedLinear,
        inputs: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        assert module.bias is None
        self.tensors.append(output[0].clone())
        self.hidden_states.append(inputs[0].clone())
        self.unrounded_logits.append(
            torch.mm(
                inputs[0].reshape(-1, module.weight.shape[1]),
                module.weight.T,
                out_dtype=torch.float32,
            ).reshape_as(output[0])
        )


def teacher_forced_logits(
    native: Qwen3OmniTalker, codes: torch.Tensor, hidden_row: torch.Tensor
) -> torch.Tensor:
    embeddings = torch.cat(
        (hidden_row.view(1, 1, -1), native.model.codec_embedding(codes[:, :1])), dim=1
    )
    cache_length = 0
    logits = []
    for group in range(native.config.num_code_groups - 1):
        hidden_states = native.predictor_forward_tokens(
            token_embeds=embeddings, batch_size=1, cache_len=cache_length
        )
        logits.append(
            native.code_predictor.lm_head[group](hidden_states[:, -1:])[0].clone()
        )
        cache_length += embeddings.shape[1]
        if group < native.config.num_code_groups - 2:
            embeddings = native.code_predictor.model.codec_embedding[group](
                codes[:, group + 1 : group + 2]
            )
        else:
            pass
    return torch.cat(logits, dim=1).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.set_float32_matmul_precision("highest")
    scheduler = create_talker_ar_executor_from_config(
        arguments.model_path,
        gpu_id=0,
        max_running_requests=arguments.max_running_requests,
        total_gpu_memory_fraction=arguments.memory_fraction,
        code2wav_in_process=True,
    )
    native = scheduler.model_runner.model
    metadata = []
    for layer_index, layer in enumerate(native.model.layers):
        metadata.append(
            {
                "layer": layer_index,
                "input_norm_dtype": str(layer.input_layernorm.weight.dtype),
                "input_norm_cast_before_weight": layer.input_layernorm.cast_x_before_out_mul,
                "query_norm_dtype": str(layer.self_attn.q_norm.weight.dtype),
                "query_norm_cast_before_weight": layer.self_attn.q_norm.cast_x_before_out_mul,
                "key_norm_dtype": str(layer.self_attn.k_norm.weight.dtype),
                "key_norm_cast_before_weight": layer.self_attn.k_norm.cast_x_before_out_mul,
                "rotary_class": type(layer.self_attn.rotary_emb).__name__,
                "rotary_cache_dtype": str(
                    layer.self_attn.rotary_emb.cos_sin_cache.dtype
                ),
                "rotary_neox_style": layer.self_attn.rotary_emb.is_neox_style,
                "rotary_interleaved": layer.self_attn.rotary_emb.mrope_interleaved,
                "rotary_interleaved_glm": layer.self_attn.rotary_emb.mrope_interleaved_glm,
            }
        )
    (arguments.output_directory / "native-metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    rotary_cache = native.model.layers[0].self_attn.rotary_emb.cos_sin_cache
    assert all(
        torch.equal(rotary_cache, layer.self_attn.rotary_emb.cos_sin_cache)
        for layer in native.model.layers
    )
    torch.save(
        rotary_cache.cpu(), arguments.output_directory / "native-rotary-cache.pt"
    )
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
    float_predictor = copy.deepcopy(reference.code_predictor).float().eval()
    capture = PredictorLogits()
    handles = [
        head.register_forward_hook(capture.record)
        for head in native.code_predictor.lm_head
    ]
    reports = []
    with torch.inference_mode():
        for request_id in arguments.request_id:
            directory = arguments.capture_directory / "talker_state"
            conditioning = torch.load(
                directory / f"{request_id}.conditioning.pt",
                map_location=reference.device,
                weights_only=True,
            )
            saved_codes = torch.load(
                directory / f"{request_id}.pt",
                map_location=reference.device,
                weights_only=True,
            ).long()
            for frame, hidden_row in enumerate(
                conditioning["native_backbone_hidden_states"]
            ):
                codes = saved_codes[frame : frame + 1]
                assert int(codes[0, 0]) != config.codec_eos_token_id
                capture.tensors.clear()
                capture.hidden_states.clear()
                capture.unrounded_logits.clear()
                native_codes, _ = native.code_predictor_forward_incremental_eager(
                    codes[:, :1], hidden_row[None]
                )
                native_codes = native_codes.squeeze(-1).clone()
                assert torch.equal(native_codes, codes), (
                    request_id,
                    frame,
                    native_codes,
                    codes,
                )
                assert len(capture.tensors) == config.num_code_groups - 1
                native_logits = torch.cat(capture.tensors, dim=1).float()
                predictor_hidden_states = torch.cat(capture.hidden_states, dim=1)
                unrounded_logits = torch.cat(capture.unrounded_logits, dim=1)
                assert torch.equal(native_logits.argmax(-1), codes[:, 1:])
                assert torch.equal(
                    teacher_forced_logits(native, codes, hidden_row), native_logits
                )
                fused_layers = native.predictor_fused_layers
                assert fused_layers is not None
                native.predictor_fused_layers = None
                plain_logits = teacher_forced_logits(native, codes, hidden_row)
                native.predictor_fused_layers = fused_layers
                embeddings = [
                    hidden_row.view(1, 1, -1),
                    reference.model.codec_embedding(codes[:, :1]),
                ]
                embeddings.extend(
                    reference.code_predictor.model.codec_embedding[group](
                        codes[:, group + 1 : group + 2]
                    )
                    for group in range(config.num_code_groups - 2)
                )
                predictor_inputs = torch.cat(embeddings, dim=1)
                reference_logits = []
                for predictor in (reference.code_predictor, float_predictor):
                    hidden_states = predictor.model(
                        inputs_embeds=predictor_inputs.to(predictor.dtype),
                        use_cache=False,
                    ).last_hidden_state
                    reference_logits.append(
                        torch.stack(
                            [
                                head(hidden_states[:, group + 1])
                                for group, head in enumerate(predictor.lm_head)
                            ],
                            dim=1,
                        ).float()
                    )
                report = {
                    "request_id": request_id,
                    "frame": frame,
                    "native_replay_exact": True,
                    "native_argmax_exact": True,
                    "native_float32_relative_l2": float(
                        (native_logits - reference_logits[1]).norm()
                        / reference_logits[1].norm()
                    ),
                    "reference_float32_relative_l2": float(
                        (reference_logits[0] - reference_logits[1]).norm()
                        / reference_logits[1].norm()
                    ),
                    "plain_float32_relative_l2": float(
                        (plain_logits - reference_logits[1]).norm()
                        / reference_logits[1].norm()
                    ),
                    "groups": [
                        {
                            "group": group + 1,
                            "native_code": int(codes[0, group + 1]),
                            "unrounded_native_code": int(
                                unrounded_logits[0, group].argmax()
                            ),
                            "native_tied_maxima": int(
                                (
                                    native_logits[0, group]
                                    == native_logits[0, group].max()
                                ).sum()
                            ),
                            "unrounded_output_rounds_to_native": bool(
                                torch.equal(
                                    unrounded_logits[0, group]
                                    .to(native.model.codec_embedding.weight.dtype)
                                    .float(),
                                    native_logits[0, group],
                                )
                            ),
                            "plain_native_code": int(plain_logits[0, group].argmax()),
                            "reference_bfloat16_code": int(
                                reference_logits[0][0, group].argmax()
                            ),
                            "reference_float32_code": int(
                                reference_logits[1][0, group].argmax()
                            ),
                            "native_float32_logit_gap": float(
                                reference_logits[1][0, group].max()
                                - reference_logits[1][0, group, codes[0, group + 1]]
                            ),
                        }
                        for group in range(config.num_code_groups - 1)
                    ],
                }
                torch.save(
                    {
                        "native_logits": native_logits.cpu(),
                        "native_predictor_hidden_states": predictor_hidden_states.cpu(),
                        "native_unrounded_logits": unrounded_logits.cpu(),
                        "plain_logits": plain_logits.cpu(),
                        "reference_bfloat16_logits": reference_logits[0].cpu(),
                        "reference_float32_logits": reference_logits[1].cpu(),
                        "codes": codes.cpu(),
                        "hidden_states": hidden_row.cpu(),
                    },
                    arguments.output_directory / f"{request_id}.{frame}.pt",
                )
                reports.append(report)
                (arguments.output_directory / "summary.json").write_text(
                    json.dumps(reports, indent=2)
                )
                print(json.dumps(report), flush=True)
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
