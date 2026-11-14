# SPDX-License-Identifier: Apache-2.0
"""Generate checkpoint Talker codes with captured conditioning and sampling controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import xxhash
from huggingface_hub import hf_hub_download
from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from transformers import Qwen3OmniMoeConfig
from transformers.cache_utils import DynamicCache
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_normalization import NativeNormalization
from scripts.debug_qwen3_omni_predictor_cache import NativePredictorReplay
from scripts.debug_qwen3_omni_talker_experts import NativeTalkerExperts
from scripts.qwen3_omni_attention_inputs import PROJECTION_NAMES, NativeAttentionInputs
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from scripts.qwen3_omni_talker_attention import NativeBackboneAttention
from scripts.qwen3_omni_talker_expert_runtime import ObservedExpertProfileMap
from scripts.qwen3_omni_talker_rotary import NativeBackboneRotary
from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    load_thinker_embedding_rows,
)
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--attention-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument(
        "--sampling-order",
        choices=("rank", "token"),
        nargs="+",
        default=("rank",),
        help="Generation noise coordinates; native draw verification remains rank keyed.",
    )
    parser.add_argument(
        "--normalization-semantics",
        choices=("checkpoint", "native", "native-boundary", "native-query-key"),
        default="checkpoint",
    )
    parser.add_argument("--paired-normalization-controls", action="store_true")
    parser.add_argument(
        "--residual-policy",
        choices=("greedy", "sampled"),
        nargs="+",
        default=("greedy",),
        help="Free-running residual-code policies, in execution order, with repeatable request-derived Torch RNG.",
    )
    parser.add_argument("--audit-loaded-weights", action="store_true")
    parser.add_argument("--native-predictor-capture-directory", type=Path)
    parser.add_argument("--native-expert-capture-directory", type=Path)
    parser.add_argument("--native-expert-profile-map", type=Path)
    parser.add_argument("--native-shared-expert", action="store_true")
    parser.add_argument(
        "--expert-output-arithmetic",
        choices=(
            "native",
            "checkpoint",
            "rounded-contributions",
            "bf16-reduction",
            "checkpoint-reduction",
        ),
        default="native",
    )
    parser.add_argument("--native-shared-gate-batches", action="store_true")
    parser.add_argument("--shared-gate-decode-rows", type=int)
    parser.add_argument("--paired-native-expert-controls", action="store_true")
    parser.add_argument("--paired-native-attention-controls", action="store_true")
    parser.add_argument("--native-backbone-attention", action="store_true")
    parser.add_argument("--decode-page-table-tokens", type=int)
    parser.add_argument("--paired-native-projection-controls", action="store_true")
    parser.add_argument("--paired-native-rotary-controls", action="store_true")
    parser.add_argument(
        "--paired-native-routing-controls",
        action="store_true",
        help="Separate native indices, weight normalization, both checkpoint-rounded weight paths, and an exact baseline repeat.",
    )
    parser.add_argument(
        "--routing-controls",
        choices=(
            "reconstructed",
            "indices",
            "native",
            "rounded",
            "indices-rounded",
            "reconstructed-repeat",
        ),
        nargs="+",
    )
    parser.add_argument(
        "--native-expert-route-precision",
        choices=("checkpoint", "float32"),
        default="checkpoint",
    )
    parser.add_argument(
        "--native-feedback-reduction",
        choices=("sequential", "checkpoint"),
        default="sequential",
    )
    parser.add_argument(
        "--capture-layer-states",
        action="store_true",
        help="Retain layer outputs and new KV rows for every reference frame.",
    )
    parser.add_argument("--conditioning-directory", type=Path)
    parser.add_argument("--thinker-hidden-directory", type=Path)
    parser.add_argument(
        "--conditioning-variant",
        choices=(
            "checkpoint_audio_prefill",
            "native_audio_prefill",
            "checkpoint_native_features_prefill",
        ),
        default="checkpoint_audio_prefill",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.shared_gate_decode_rows is not None:
        assert arguments.native_shared_gate_batches
        assert arguments.shared_gate_decode_rows > 0
    else:
        pass
    if arguments.expert_output_arithmetic != "native":
        assert arguments.native_expert_capture_directory is not None
    else:
        pass
    if arguments.native_shared_gate_batches:
        assert arguments.native_expert_profile_map is not None
        assert arguments.native_shared_expert
        assert not arguments.paired_native_expert_controls
    else:
        pass
    if arguments.routing_controls is not None:
        assert arguments.paired_native_routing_controls
    else:
        pass
    if arguments.decode_page_table_tokens is not None:
        assert arguments.native_backbone_attention
        assert arguments.decode_page_table_tokens > 0
    else:
        pass
    if (
        arguments.paired_native_attention_controls
        or arguments.native_backbone_attention
    ):
        assert arguments.attention_implementation == "eager"
        assert arguments.dtype == "bfloat16"
        assert not (
            arguments.paired_native_attention_controls
            and arguments.native_backbone_attention
        )
    else:
        pass
    if arguments.native_expert_capture_directory is not None:
        assert arguments.dtype == "bfloat16"
    else:
        assert not arguments.paired_native_expert_controls
        assert not arguments.native_shared_expert
    if (
        arguments.native_predictor_capture_directory is not None
        and arguments.dtype != "bfloat16"
    ):
        raise ValueError("Native predictor recurrence requires BF16 weights and inputs")
    elif arguments.native_predictor_capture_directory is not None and any(
        policy != "greedy" for policy in arguments.residual_policy
    ):
        raise ValueError("Native predictor recurrence requires greedy residual policy")
    elif (
        arguments.native_predictor_capture_directory is None
        and arguments.native_feedback_reduction != "sequential"
    ):
        raise ValueError("Native feedback selection requires the native predictor")
    else:
        pass
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    model_config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    config = model_config.talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation=arguments.attention_implementation,
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    assert not reference.code_predictor.generation_config.do_sample
    reference.eval()
    assert all(
        layer.self_attn.config._attn_implementation
        == arguments.attention_implementation
        for layer in (*reference.model.layers, *reference.code_predictor.model.layers)
    )
    normalization = NativeNormalization(reference=reference)
    if (
        arguments.normalization_semantics.startswith("native")
        or arguments.paired_normalization_controls
    ):
        normalization.install()
    else:
        pass
    checkpoint_directory = Path(
        hf_hub_download(
            arguments.model_path, "config.json", revision=arguments.model_revision
        )
    ).parent
    if arguments.native_predictor_capture_directory is not None:
        predictor_audit_directory = (
            arguments.output_directory / "native-predictor-audit"
        )
        predictor_audit_directory.mkdir()
        native_predictor = NativePredictorReplay(
            reference=reference,
            capture_directory=arguments.native_predictor_capture_directory,
            output_directory=predictor_audit_directory,
            model_path=arguments.model_path,
            model_revision=arguments.model_revision,
            feedback_reduction=arguments.native_feedback_reduction,
        )
    else:
        native_predictor = None
    if arguments.native_expert_capture_directory is not None:
        expert_control = NativeTalkerExperts(
            reference=reference,
            capture_directory=arguments.native_expert_capture_directory,
            route_precision=arguments.native_expert_route_precision,
            output_arithmetic=arguments.expert_output_arithmetic,
            shared_enabled=arguments.native_shared_expert,
            is_shared_gate_batch_replay_enabled=arguments.native_shared_gate_batches,
            shared_gate_decode_rows=arguments.shared_gate_decode_rows,
        )
        expert_control.install()
    else:
        expert_control = None
    if arguments.native_expert_profile_map is not None:
        assert expert_control is not None
        observed_profiles = ObservedExpertProfileMap.model_validate_json(
            arguments.native_expert_profile_map.read_bytes()
        )
    else:
        observed_profiles = None
    if (
        arguments.paired_native_rotary_controls
        or arguments.paired_native_routing_controls
    ):
        assert arguments.native_predictor_capture_directory is not None
        assert not arguments.paired_native_projection_controls
        assert arguments.native_backbone_attention
        assert not (
            arguments.paired_native_rotary_controls
            and arguments.paired_native_routing_controls
        )
        if arguments.paired_native_routing_controls:
            assert (
                expert_control is not None
                and expert_control.route_precision == "float32"
            )
            assert not arguments.paired_native_expert_controls
        else:
            pass
        rotary_control = NativeBackboneRotary(
            reference, arguments.native_predictor_capture_directory
        )
        rotary_control.install()
    else:
        rotary_control = None
    if (
        arguments.paired_native_projection_controls
        or arguments.paired_native_rotary_controls
        or arguments.paired_native_routing_controls
    ):
        assert arguments.dtype == "bfloat16"
        projection_control = NativeAttentionInputs(reference=reference)
        projection_control.install()
    else:
        projection_control = None
    if (
        arguments.paired_native_attention_controls
        or arguments.native_backbone_attention
    ):
        attention_control = NativeBackboneAttention(
            reference=reference,
            decode_page_table_tokens=arguments.decode_page_table_tokens,
        )
        attention_control.install()
    else:
        attention_control = None
    special_embeddings = load_thinker_embedding_rows(
        str(checkpoint_directory),
        [
            model_config.tts_bos_token_id,
            model_config.tts_eos_token_id,
            model_config.tts_pad_token_id,
        ],
    ).to(device=reference.device, dtype=torch.bfloat16)
    reports = []
    with torch.inference_mode():
        projected_special_embeddings = reference.text_projection(special_embeddings)
        if arguments.dtype == "float32":
            reference.float()
            projected_special_embeddings = projected_special_embeddings.float()
        else:
            pass
        eos_embedding = projected_special_embeddings[1:2]
        pad_embedding = projected_special_embeddings[2:3]
        token_ids = torch.arange(config.text_config.vocab_size, device=reference.device)
        allowed_tokens = (token_ids < config.text_config.vocab_size - 1024) | (
            token_ids == config.codec_eos_token_id
        )
        top_k = torch.tensor([arguments.top_k], device=reference.device)
        top_p = torch.tensor([arguments.top_p], device=reference.device)
        minimum_probability = torch.zeros(1, device=reference.device)
        repetition_penalty = torch.tensor(
            [[arguments.repetition_penalty]],
            dtype=torch.float32,
            device=reference.device,
        )
        temperature = torch.tensor(
            [[arguments.temperature]], dtype=torch.float32, device=reference.device
        )
        intervention_mode = (
            "native"
            if arguments.normalization_semantics == "checkpoint"
            else arguments.normalization_semantics
        )
        normalization_modes = (
            ("checkpoint", intervention_mode, "checkpoint-repeat")
            if arguments.paired_normalization_controls
            else (arguments.normalization_semantics,)
        )
        expert_modes = (
            ("checkpoint", "fused", "checkpoint-repeat")
            if arguments.paired_native_expert_controls
            else ("fused" if expert_control is not None else "checkpoint",)
        )
        if arguments.paired_native_attention_controls:
            attention_modes = ("eager", "fa3", "eager-repeat")
        elif arguments.native_backbone_attention:
            attention_modes = ("fa3",)
        else:
            attention_modes = ("unchanged",)
        if arguments.paired_native_projection_controls:
            projection_modes = (
                "checkpoint",
                "merged",
                "paired",
                "merged-paired",
                "checkpoint-repeat",
            )
        elif (
            arguments.paired_native_rotary_controls
            or arguments.paired_native_routing_controls
        ):
            projection_modes = ("merged-paired",)
        else:
            projection_modes = ("checkpoint",)
        if arguments.paired_native_rotary_controls:
            rotary_modes = ("checkpoint", "native", "checkpoint-repeat")
        elif arguments.paired_native_routing_controls:
            rotary_modes = ("native",)
        else:
            rotary_modes = ("checkpoint",)
        routing_modes = (
            (
                "reconstructed",
                "indices",
                "native",
                "rounded",
                "indices-rounded",
                "reconstructed-repeat",
            )
            if arguments.paired_native_routing_controls
            else ("reconstructed",)
        )
        if arguments.routing_controls is not None:
            routing_modes = tuple(arguments.routing_controls)
        else:
            pass
        for (
            trace_directory,
            sampling_order,
            normalization_mode,
            expert_mode,
            projection_mode,
            rotary_mode,
            routing_mode,
            attention_mode,
            residual_policy_index,
            residual_policy,
        ) in (
            (
                directory,
                order,
                mode,
                expert_mode,
                projection_mode,
                rotary_mode,
                routing_mode,
                attention_mode,
                policy_index,
                policy,
            )
            for directory in arguments.trace_directory
            for order in arguments.sampling_order
            for mode in normalization_modes
            for expert_mode in expert_modes
            for projection_mode in projection_modes
            for rotary_mode in rotary_modes
            for routing_mode in routing_modes
            for attention_mode in attention_modes
            for policy_index, policy in enumerate(arguments.residual_policy)
        ):
            if rotary_control is not None:
                assert (
                    rotary_control.positions is None
                    and rotary_control.active_layer is None
                )
                rotary_control.enabled = rotary_mode == "native"
                rotary_control.calls = 0
                rotary_control.position_calls = 0
            else:
                pass
            if projection_control is not None:
                assert not projection_control.projections
                projection_control.projection_calls = 0
                projection_control.normalization_calls = 0
                if projection_mode in ("merged", "paired", "merged-paired"):
                    projection_control.enabled = True
                    projection_control.mode = projection_mode
                else:
                    projection_control.enabled = False
            else:
                pass
            if attention_control is not None:
                attention_control.set_enabled(attention_mode == "fa3")
            else:
                pass
            if expert_control is not None:
                expert_control.enabled = expert_mode == "fused"
                expert_control.calls = 0
                expert_control.kernel_calls = 0
                expert_control.router_calls = 0
                expert_control.routing_semantics = (
                    "reconstructed"
                    if routing_mode == "reconstructed-repeat"
                    else routing_mode
                )
                expert_control.shared_calls = 0
                expert_control.shared_gate_calls = 0
                expert_control.shared_gate_exact_input_calls = 0
                expert_control.shared_gate_snapshots.clear()
                expert_control.used_profiles.clear()
                expert_control.profiles = (
                    observed_profiles.profiles[str(trace_directory.resolve())]
                    if observed_profiles is not None
                    else None
                )
                if expert_control.profiles is not None:
                    assert (
                        len(expert_control.profiles.prefill)
                        == len(expert_control.profiles.decode)
                        == len(reference.model.layers)
                    )
                else:
                    pass
            else:
                pass
            reference.code_predictor.generation_config.do_sample = (
                residual_policy == "sampled"
            )
            normalization.boundary_calls = 0
            normalization.query_key_calls = 0
            normalization.boundary_enabled = normalization_mode in (
                "native",
                "native-boundary",
            )
            normalization.query_key_enabled = normalization_mode in (
                "native",
                "native-query-key",
            ) and not (
                projection_control is not None
                and projection_control.enabled
                and projection_control.mode in ("paired", "merged-paired")
            )
            prefill_file = next(trace_directory.glob("*-before_prefill.pt"))
            captured_sampling = json.loads(
                (trace_directory / "sampling.json").read_text()
            )
            assert captured_sampling["temperature"] == arguments.temperature
            assert captured_sampling["top_k"] == arguments.top_k
            assert captured_sampling["top_p"] == arguments.top_p
            assert (
                captured_sampling["repetition_penalty"] == arguments.repetition_penalty
            )
            assert captured_sampling["min_p"] == 0
            assert sorted(captured_sampling["suppress_tokens"]) == (
                (~allowed_tokens).nonzero(as_tuple=True)[0].tolist()
            )
            before = torch.load(prefill_file, map_location="cpu", weights_only=True)
            finished_file = trace_directory / "finished.pt"
            if finished_file.exists():
                finished = torch.load(finished_file, weights_only=True)
                before["trailing_text_embeds"] = finished["enqueued_text_rows"]
            else:
                finished = None
            prefill = before["prefill_embeds"].to(
                device=reference.device, dtype=reference.dtype
            )
            text_rows = before["trailing_text_embeds"].to(
                device=reference.device, dtype=reference.dtype
            )
            assert torch.equal(
                before["positions"], torch.arange(len(prefill))
            ), "this experiment requires captured linear audio-only positions"
            assert torch.equal(text_rows[-1:], eos_embedding), "incomplete text queue"
            complete_text_verified = False
            conditioning_file = trace_directory / "conditioning.pt"
            if arguments.dtype == "bfloat16" and (
                conditioning_file.exists() or finished is not None
            ):
                if finished is not None:
                    captured_conditioning = torch.load(
                        trace_directory / "prompt.pt", weights_only=True
                    )
                    captured_conditioning["assistant_token_ids"] = finished[
                        "assistant_token_ids"
                    ]
                else:
                    captured_conditioning = torch.load(
                        conditioning_file, map_location="cpu", weights_only=True
                    )
                assistant_ids = captured_conditioning["assistant_token_ids"]
                if int(assistant_ids[-1]) == model_config.im_end_token_id:
                    prompt_ids = captured_conditioning["prompt_input_ids"]
                    assistant_start = int(
                        (prompt_ids == model_config.im_start_token_id).nonzero(
                            as_tuple=True
                        )[0][-1]
                    )
                    assistant_prefix = prompt_ids[assistant_start:]
                    assert int(assistant_prefix[1]) == model_config.assistant_token_id
                    assistant_context_ids = torch.cat(
                        [assistant_prefix, assistant_ids[:-1]]
                    )
                    assistant_embeddings = load_thinker_embedding_rows(
                        str(checkpoint_directory), assistant_context_ids.tolist()
                    ).to(reference.device, dtype=reference.dtype)
                    projected_assistant = reference.text_projection(
                        assistant_embeddings
                    )[len(assistant_prefix) :]
                    expected_text_rows = torch.cat(
                        [projected_assistant[1:], eos_embedding], dim=0
                    )
                    assert torch.equal(expected_text_rows, text_rows)
                    codec_bos = reference.get_input_embeddings()(
                        torch.tensor([config.codec_bos_id], device=reference.device)
                    )
                    assert torch.equal(
                        projected_assistant[:1] + codec_bos, prefill[-1:]
                    )
                    complete_text_verified = True
                else:
                    pass
            else:
                pass
            native_files = sorted(
                trace_directory.glob(
                    "*-output.pt" if finished is not None else "*-post_*.pt"
                )
            )
            assert native_files
            if finished is not None:
                assert len(native_files) == len(finished["output_ids"])
                assert [filename.name for filename in native_files] == [
                    f"{frame:04d}-output.pt" for frame in range(len(native_files))
                ]
            else:
                pass
            native_frames = []
            for filename in native_files:
                captured = torch.load(filename, map_location="cpu", weights_only=True)
                native_frames.append(
                    {
                        name: (
                            captured[name].unsqueeze(0)
                            if finished is not None
                            else captured[name]
                        )
                        for name in ("codec_ids", "codec_feedback", "logits")
                    }
                )
            seed = (
                xxhash.xxh64_intdigest(trace_directory.name.encode())
                & SAMPLING_SEED_MASK
            )
            sampling_seed = torch.tensor([seed], device=reference.device)
            assert captured_sampling["sampling_seed"] == seed
            checked_feedback_rows = 0
            for frame_index, filename in enumerate(
                sorted(trace_directory.glob("*-before_decode.pt"))
            ):
                captured = torch.load(filename, map_location="cpu", weights_only=True)
                if finished is not None:
                    captured["feedback_input"] = finished["decode_input_history"][
                        frame_index
                    ].unsqueeze(0)
                else:
                    pass
                assert captured["sampling_seed"].tolist() == [seed]
                assert captured["positions"].tolist() == [len(prefill) + frame_index]
                next_text = (
                    text_rows[frame_index : frame_index + 1]
                    if frame_index < len(text_rows)
                    else pad_embedding
                )
                previous_feedback = native_frames[frame_index]["codec_feedback"].to(
                    reference.device
                )
                assert torch.equal(
                    previous_feedback + next_text.to(previous_feedback.dtype),
                    captured["feedback_input"].to(reference.device),
                ), f"uncaptured text handoff at {filename}"
                checked_feedback_rows += 1
            captured_prefill_probabilities = native_frames[0]["logits"].to(
                reference.device
            )
            reproduced_prefill_choice = top_k_top_p_min_p_sampling_from_probs_torch(
                captured_prefill_probabilities,
                top_k,
                top_p,
                minimum_probability,
                False,
                sampling_seed,
                before["positions"][-1:].to(reference.device),
            )
            assert int(reproduced_prefill_choice[0]) == int(
                native_frames[0]["codec_ids"][0, 0]
            )
            if arguments.conditioning_directory is not None:
                conditioning = torch.load(
                    arguments.conditioning_directory / f"{trace_directory.name}.pt",
                    map_location=reference.device,
                    weights_only=True,
                )
                assert torch.equal(
                    conditioning["native_prefill"].to(reference.dtype), prefill
                )
                prefill = conditioning[arguments.conditioning_variant].to(
                    reference.dtype
                )
                if arguments.thinker_hidden_directory is not None:
                    thinker_hidden = torch.load(
                        arguments.thinker_hidden_directory
                        / f"{trace_directory.name}.pt",
                        map_location=reference.device,
                        weights_only=True,
                    )
                    prompt_ids = conditioning["prompt_input_ids"][0]
                    audio_mask = (
                        prompt_ids == model_config.thinker_config.audio_token_id
                    )
                    prefill = conditioning["native_prefill"].to(reference.dtype).clone()
                    prefill[conditioning["audio_rows"]] = reference.hidden_projection(
                        thinker_hidden[0, audio_mask].to(reference.dtype)
                    )
                else:
                    pass
            else:
                assert arguments.thinker_hidden_directory is None
            sampling_directory = (
                arguments.output_directory / sampling_order
                if len(arguments.sampling_order) > 1
                else arguments.output_directory
            )
            mode_directory = (
                sampling_directory / normalization_mode
                if arguments.paired_normalization_controls
                else sampling_directory
            )
            if arguments.paired_native_expert_controls:
                mode_directory = mode_directory / expert_mode
            else:
                pass
            if attention_control is not None:
                mode_directory = mode_directory / attention_mode
            else:
                pass
            if projection_control is not None:
                mode_directory = mode_directory / projection_mode
            else:
                pass
            if rotary_control is not None:
                mode_directory = mode_directory / f"rotary-{rotary_mode}"
            else:
                pass
            if arguments.paired_native_routing_controls:
                mode_directory = mode_directory / f"routing-{routing_mode}"
            else:
                pass
            if len(arguments.residual_policy) > 1:
                mode_directory = (
                    mode_directory
                    / f"residual-{residual_policy_index}-{residual_policy}"
                )
            else:
                pass
            case_directory = (
                mode_directory / trace_directory.parent.name / trace_directory.name
            )
            case_directory.mkdir(parents=True)
            if residual_policy == "sampled" or len(arguments.residual_policy) > 1:
                torch.manual_seed(seed)
                torch.save(
                    {
                        "cpu": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(reference.device),
                    },
                    case_directory / "initial-rng-state.pt",
                )
            else:
                pass
            generated_ids = before["input_ids"][None].to(reference.device)
            prepared_inputs = {
                "inputs_embeds": prefill[None],
                "position_ids": before["positions"]
                .to(reference.device)
                .view(1, 1, -1)
                .expand(4, 1, -1),
                "use_cache": True,
                "output_hidden_states": True,
            }
            repeated_tokens = torch.zeros_like(token_ids, dtype=torch.bool)[None]
            first_codec_divergence = None
            first_main_code_divergence = None
            frame_reports = []
            termination = "frame_limit"
            frame_limit = min(arguments.max_frames, captured_sampling["max_new_tokens"])
            for frame_index in range(frame_limit):
                if arguments.capture_layer_states:
                    captured_inputs = prepared_inputs["inputs_embeds"].detach().cpu()
                    captured_positions = prepared_inputs["position_ids"].detach().cpu()
                else:
                    pass
                output = reference(**prepared_inputs)
                assert (
                    output.past_key_values.get_seq_length()
                    == len(prefill) + frame_index
                )
                assert output.generation_step == frame_index
                if arguments.capture_layer_states:
                    assert isinstance(output.past_key_values, DynamicCache)
                    tokens = captured_inputs.shape[1]
                    keys: list[torch.Tensor] = []
                    values: list[torch.Tensor] = []
                    for layer in output.past_key_values.layers:
                        assert layer.keys is not None and layer.values is not None
                        assert (
                            layer.keys.shape[-2]
                            == layer.values.shape[-2]
                            == len(prefill) + frame_index
                        )
                        keys.append(
                            layer.keys[..., -tokens:, :].detach().cpu().contiguous()
                        )
                        values.append(
                            layer.values[..., -tokens:, :].detach().cpu().contiguous()
                        )
                    torch.save(
                        {
                            "input_embeddings": captured_inputs,
                            "positions": captured_positions,
                            "hidden_states": [
                                hidden.detach().cpu().contiguous()
                                for hidden in output.hidden_states[0]
                            ],
                            "keys": keys,
                            "values": values,
                        },
                        case_directory / f"{frame_index:04d}-layers.pt",
                    )
                else:
                    pass
                logits = output.logits[:, -1].float()
                penalized_logits = torch.where(
                    logits > 0,
                    logits / repetition_penalty,
                    logits * repetition_penalty,
                )
                sampling_logits = torch.where(
                    repeated_tokens, penalized_logits, logits
                ).masked_fill(~allowed_tokens, float("-inf"))
                sampling_position = len(prefill) - 1 + frame_index
                probabilities = torch.softmax(sampling_logits / temperature, dim=-1)
                sampling_positions = torch.tensor(
                    [sampling_position], device=reference.device
                )
                if sampling_order == "rank":
                    sampled_code = top_k_top_p_min_p_sampling_from_probs_torch(
                        probabilities,
                        top_k,
                        top_p,
                        minimum_probability,
                        False,
                        sampling_seed,
                        sampling_positions,
                    ).long()
                else:
                    _, filtered_weights, sorted_token_ids, _ = (
                        top_k_top_p_min_p_sampling_from_probs_torch(
                            probabilities,
                            top_k,
                            top_p,
                            minimum_probability,
                            False,
                            sampling_seed,
                            sampling_positions,
                            return_filtered_probs=True,
                        )
                    )
                    weights_by_token = torch.zeros_like(filtered_weights).scatter_(
                        1, sorted_token_ids.long(), filtered_weights
                    )
                    sampled_code = (
                        multinomial_with_seed(
                            weights_by_token.double().log(),
                            sampling_seed,
                            sampling_positions,
                        )
                        .reshape(-1)
                        .long()
                    )
                repeated_tokens[:, sampled_code] = True
                is_eos = int(sampled_code[0]) == config.codec_eos_token_id
                generated_ids = torch.cat((generated_ids, sampled_code[:, None]), dim=1)
                if is_eos:
                    codes = torch.zeros(
                        (1, config.num_code_groups),
                        dtype=torch.long,
                        device=reference.device,
                    )
                    codes[:, 0] = sampled_code
                    termination = "codec_eos"
                else:
                    prepared_inputs = reference.prepare_inputs_for_generation(
                        generated_ids,
                        next_sequence_length=1,
                        past_key_values=output.past_key_values,
                        attention_mask=torch.ones_like(generated_ids),
                        is_first_iteration=False,
                        position_ids=torch.tensor(
                            [[[sampling_position + 1]]], device=reference.device
                        ).expand(4, 1, 1),
                        hidden_states=output.hidden_states,
                        generation_step=output.generation_step,
                        trailing_text_hidden=text_rows[None],
                        tts_pad_embed=pad_embedding[None],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    codes = prepared_inputs["residual_codes"]
                    if native_predictor is not None:
                        reference_codes = codes.clone()
                        predictor_query = output.hidden_states[0][-1][:, -1:]
                        codes, feedback = native_predictor.generate(
                            sampled_code, predictor_query
                        )
                        next_text = (
                            text_rows[frame_index : frame_index + 1]
                            if frame_index < len(text_rows)
                            else pad_embedding
                        )
                        prepared_inputs["residual_codes"] = codes
                        prepared_inputs["inputs_embeds"] = feedback + next_text[None]
                        torch.save(
                            {
                                "query": predictor_query.cpu(),
                                "reference_codes": reference_codes.cpu(),
                                "native_codes": codes.cpu(),
                                "feedback": feedback.cpu(),
                                "text_embedding": next_text.cpu(),
                                "next_inputs": prepared_inputs["inputs_embeds"].cpu(),
                                "keys": native_predictor.keys[
                                    :, :, : config.num_code_groups
                                ].cpu(),
                                "values": native_predictor.values[
                                    :, :, : config.num_code_groups
                                ].cpu(),
                            },
                            case_directory / f"{frame_index:04d}-predictor-handoff.pt",
                        )
                    else:
                        pass
                    assert int(codes[0, 0]) == int(sampled_code[0])
                    assert prepared_inputs["inputs_embeds"].shape == (
                        1,
                        1,
                        prefill.shape[-1],
                    )
                if frame_index < len(native_frames):
                    native_codes = native_frames[frame_index]["codec_ids"].to(
                        reference.device
                    )
                    same_main_code = int(native_codes[0, 0]) == int(sampled_code[0])
                    same_codec_frame = (
                        None if is_eos else torch.equal(native_codes, codes)
                    )
                    if not same_main_code and first_main_code_divergence is None:
                        first_main_code_divergence = frame_index
                    else:
                        pass
                    if same_codec_frame is False and first_codec_divergence is None:
                        first_codec_divergence = frame_index
                    else:
                        pass
                else:
                    same_main_code = None
                    same_codec_frame = None
                phase = "post_prefill" if frame_index == 0 else "post_decode"
                torch.save(
                    {
                        "codec_ids": codes.cpu(),
                        "hidden_states": output.hidden_states[0][-1][:, -1].cpu(),
                        "logits": logits.cpu(),
                    },
                    case_directory / f"{frame_index:04d}-{phase}.pt",
                )
                frame_reports.append(
                    {
                        "frame_index": frame_index,
                        "sampling_position": sampling_position,
                        "main_code": int(sampled_code[0]),
                        "same_native_main_code": same_main_code,
                        "same_native_codec_frame": same_codec_frame,
                    }
                )
                if is_eos:
                    break
                else:
                    pass
            if expert_control is not None:
                expected_calls = (
                    len(frame_reports) * len(reference.model.layers)
                    if expert_control.enabled
                    else 0
                )
                assert expert_control.calls == expected_calls
                if arguments.expert_output_arithmetic == "native":
                    expected_kernel_calls = expected_calls
                elif arguments.expert_output_arithmetic == "checkpoint":
                    expected_kernel_calls = 0
                else:
                    expected_kernel_calls = expected_calls * (
                        1 + config.text_config.num_experts_per_tok
                    )
                assert expert_control.kernel_calls == expected_kernel_calls
                assert expert_control.router_calls == (
                    expected_calls if expert_control.route_precision == "float32" else 0
                )
                assert expert_control.shared_calls == (
                    len(frame_reports) * len(reference.model.layers)
                    if expert_control.shared_enabled
                    else 0
                )
                assert expert_control.shared_gate_calls == (
                    expected_calls if arguments.native_shared_gate_batches else 0
                )
            else:
                pass
            if attention_control is not None:
                assert attention_control.calls == (
                    len(frame_reports) * len(reference.model.layers)
                    if attention_mode == "fa3"
                    else 0
                )
            else:
                pass
            if projection_control is not None:
                expected_projection_calls = (
                    len(frame_reports) * len(reference.model.layers)
                    if projection_control.enabled
                    else 0
                )
                assert not projection_control.projections
                assert projection_control.projection_calls == (
                    expected_projection_calls * len(PROJECTION_NAMES)
                )
                assert projection_control.normalization_calls == (
                    expected_projection_calls * 2
                    if projection_control.mode in ("paired", "merged-paired")
                    else 0
                )
            else:
                pass
            if rotary_control is not None:
                assert (
                    rotary_control.positions is None
                    and rotary_control.active_layer is None
                )
                assert rotary_control.calls == (
                    len(frame_reports) * len(reference.model.layers)
                    if rotary_control.enabled
                    else 0
                )
                assert rotary_control.position_calls == (
                    len(frame_reports) if rotary_control.enabled else 0
                )
            else:
                pass
            report = {
                "model_revision": arguments.model_revision,
                "frame_limit": frame_limit,
                "native_trace_directory": str(trace_directory),
                "reference_trace_directory": str(case_directory),
                "prefill_file_sha256": hashlib.sha256(
                    prefill_file.read_bytes()
                ).hexdigest(),
                "verified_sampling_seed": seed,
                "captured_feedback_rows_verified": checked_feedback_rows,
                "complete_text_queue_rows": len(text_rows),
                "complete_text_projection_verified": complete_text_verified,
                "first_codec_divergence": first_codec_divergence,
                "first_main_code_divergence": first_main_code_divergence,
                "native_frames_including_eos": len(native_frames),
                "reference_frames_including_eos": len(frame_reports),
                "termination": termination,
                "reference_dtype": arguments.dtype,
                "reference_attention_implementation": arguments.attention_implementation,
                "conditioning_variant": (
                    arguments.conditioning_variant
                    if arguments.conditioning_directory is not None
                    else "native"
                ),
                "thinker_hidden_directory": (
                    str(arguments.thinker_hidden_directory)
                    if arguments.thinker_hidden_directory is not None
                    else None
                ),
                "normalization_semantics": normalization_mode,
                "expert_mode": expert_mode,
                "attention_mode": attention_mode,
                "decode_page_table_tokens": arguments.decode_page_table_tokens,
                "attention_projection_mode": projection_mode,
                "rotary_mode": rotary_mode,
                "routing_mode": routing_mode,
                "native_rotary_calls": rotary_control.calls if rotary_control else 0,
                "native_projection_calls": (
                    projection_control.projection_calls if projection_control else 0
                ),
                "native_paired_normalization_calls": (
                    projection_control.normalization_calls if projection_control else 0
                ),
                "native_backbone_attention_calls": (
                    attention_control.calls if attention_control is not None else 0
                ),
                "native_expert_route_precision": arguments.native_expert_route_precision,
                "expert_output_arithmetic": arguments.expert_output_arithmetic,
                "native_expert_kernel_calls": (
                    expert_control.kernel_calls if expert_control else 0
                ),
                "native_expert_used_profiles": (
                    sorted(expert_control.used_profiles)
                    if expert_control is not None
                    else []
                ),
                "observed_expert_profiles_source": (
                    str(expert_control.profiles.source_capture)
                    if expert_control is not None
                    and expert_control.profiles is not None
                    else None
                ),
                "native_expert_calls": expert_control.calls if expert_control else 0,
                "native_shared_expert": arguments.native_shared_expert,
                "native_shared_gate_batches": arguments.native_shared_gate_batches,
                "shared_gate_decode_rows": arguments.shared_gate_decode_rows,
                "native_shared_gate_exact_input_calls": (
                    expert_control.shared_gate_exact_input_calls
                    if expert_control
                    else 0
                ),
                "native_shared_expert_calls": (
                    expert_control.shared_calls if expert_control else 0
                ),
                "native_expert_router_calls": (
                    expert_control.router_calls if expert_control else 0
                ),
                "sampling_order": sampling_order,
                "residual_policy": residual_policy,
                "native_predictor_capture_directory": (
                    str(arguments.native_predictor_capture_directory)
                    if arguments.native_predictor_capture_directory is not None
                    else None
                ),
                "native_feedback_reduction": (
                    arguments.native_feedback_reduction
                    if native_predictor is not None
                    else None
                ),
                "residual_torch_seed": (
                    seed
                    if residual_policy == "sampled"
                    or len(arguments.residual_policy) > 1
                    else None
                ),
                "normalization_boundary_calls": normalization.boundary_calls,
                "normalization_query_key_calls": normalization.query_key_calls,
                "frames": frame_reports,
            }
            reports.append(report)
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(
                json.dumps(
                    {name: value for name, value in report.items() if name != "frames"}
                ),
                flush=True,
            )
        if arguments.audit_loaded_weights and native_predictor is None:
            audit_checkpoint_weights(
                reference,
                checkpoint_directory,
                arguments.output_directory / "loaded-weight-audit.json",
            )
        else:
            pass


if __name__ == "__main__":
    main()
