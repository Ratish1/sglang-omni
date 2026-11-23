# SPDX-License-Identifier: Apache-2.0
"""Compare checkpoint continuation from validated native Talker conditioning."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from audit_qwen3_omni_native_predictor import load_native_talker
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_checkpoint_weights import load_talker
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.generation.utils import GenerationMixin
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

RESERVED_CODEC_TOKENS = 1024


@dataclass(kw_only=True)
class SamplingPolicy:
    seed: int
    temperature: float
    repetition_penalty: float
    top_k: int
    top_p: float
    vocabulary_tokens: int
    eos_token: int
    device: torch.device

    def probabilities(
        self, logits: torch.Tensor, repeated_tokens: torch.Tensor
    ) -> torch.Tensor:
        assert logits.dtype == torch.float32
        penalty = torch.tensor(
            [[self.repetition_penalty]], device=self.device, dtype=torch.float32
        )
        temperature = torch.tensor(
            [[self.temperature]], device=self.device, dtype=torch.float32
        )
        penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
        allowed = torch.arange(self.vocabulary_tokens, device=self.device)
        allowed = (allowed < self.vocabulary_tokens - RESERVED_CODEC_TOKENS) | (
            allowed == self.eos_token
        )
        masked = torch.where(repeated_tokens, penalized, logits).masked_fill(
            ~allowed, float("-inf")
        )
        return torch.softmax(masked / temperature, dim=-1)

    def sample(self, probabilities: torch.Tensor, position: int) -> torch.Tensor:
        return top_k_top_p_min_p_sampling_from_probs_torch(
            probabilities,
            torch.tensor([self.top_k], device=self.device),
            torch.tensor([self.top_p], device=self.device, dtype=torch.float32),
            torch.zeros(1, device=self.device),
            False,
            torch.tensor([self.seed], device=self.device, dtype=torch.int64),
            torch.tensor([position], device=self.device),
        ).long()


@dataclass(kw_only=True)
class NativeConditioning:
    prefill_embeddings: torch.Tensor
    input_ids: torch.Tensor
    text_embeddings: torch.Tensor
    padding_embedding: torch.Tensor
    codec_ids: torch.Tensor
    prefill_keys: torch.Tensor
    prefill_values: torch.Tensor
    first_feedback: torch.Tensor


def load_conditioning(
    directory: Path, policy: SamplingPolicy, destination: Path
) -> NativeConditioning:
    frames = validate(directory)
    metadata = json.loads((directory / "trajectory.json").read_text())
    assert metadata["frames"][0]["thinker_done"]
    assert metadata["frames"][0]["prefix_length"] == 0
    prefill_tokens = len(frames[0]["positions"])
    assert torch.equal(frames[0]["positions"], torch.arange(prefill_tokens))
    repeated = torch.zeros(
        (1, policy.vocabulary_tokens), dtype=torch.bool, device=policy.device
    )
    text_rows: list[torch.Tensor] = []
    padding: torch.Tensor | None = None
    for frame_index, (frame, state) in enumerate(
        zip(frames, metadata["frames"], strict=True)
    ):
        assert torch.equal(frame["rotary_positions"], frame["positions"].expand(3, -1))
        position = prefill_tokens - 1 + frame_index
        assert int(frame["positions"][-1]) == position
        logits = frame["logits"].to(policy.device).reshape(1, -1)
        if frame_index == 0:
            assert bool((logits >= 0).all())
            torch.testing.assert_close(logits.sum(), logits.new_tensor(1.0))
            probabilities = logits
        else:
            probabilities = policy.probabilities(logits, repeated)
            row = frame["text_embedding"].reshape(1, -1)
            if state["uses_text_padding"]:
                assert state["thinker_done"]
                if padding is None:
                    padding = row
                else:
                    assert torch.equal(row, padding)
            else:
                assert padding is None
                text_rows.append(row)
        sampled = policy.sample(probabilities, position)
        assert int(sampled[0]) == int(frame["codec_ids"][0]), (
            directory.name,
            frame_index,
            int(sampled[0]),
            int(frame["codec_ids"][0]),
        )
        repeated[:, sampled] = True
    assert padding is not None and text_rows
    codes = torch.stack([frame["codec_ids"] for frame in frames])
    assert int(codes[-1, 0]) == policy.eos_token
    assert not bool((codes[:-1, 0] == policy.eos_token).any())
    conditioning = NativeConditioning(
        prefill_embeddings=frames[0]["prefill_embeddings"],
        input_ids=frames[0]["input_ids"],
        text_embeddings=torch.cat(text_rows),
        padding_embedding=padding,
        codec_ids=codes,
        prefill_keys=frames[0]["stored_keys"],
        prefill_values=frames[0]["stored_values"],
        first_feedback=frames[0]["feedback"],
    )
    destination.mkdir()
    torch.save(asdict(conditioning), destination / "conditioning.pt")
    report = {
        "native_frames": len(frames),
        "verified_native_draws": len(frames),
        "prefill_tokens": prefill_tokens,
        "complete_consumed_text_rows": len(text_rows),
        "capture_sha256": sha256(
            (directory / "trajectory.pt").read_bytes()
        ).hexdigest(),
        "sampling_policy": {**asdict(policy), "device": str(policy.device)},
    }
    (destination / "conditioning.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"sample": directory.name, **report}), flush=True)
    return conditioning


def generate(
    model: Qwen3OmniMoeTalkerForConditionalGeneration,
    conditioning: NativeConditioning,
    policy: SamplingPolicy,
    max_frames: int,
    destination: Path,
    use_native_prefix: bool,
    use_sequential_feedback: bool,
    native_predictor: Qwen3OmniTalker | None,
    predictor_batch_rows: int | None,
) -> torch.Tensor:
    prefill = conditioning.prefill_embeddings.to(policy.device)[None]
    text_rows = conditioning.text_embeddings.to(policy.device)[None]
    padding = conditioning.padding_embedding.to(policy.device)[None]
    generated_ids = conditioning.input_ids.to(policy.device)[None]
    prefill_tokens = prefill.shape[1]
    prepared = {
        "inputs_embeds": prefill,
        "position_ids": torch.arange(prefill_tokens, device=policy.device)
        .view(1, 1, -1)
        .expand(4, 1, -1),
        "use_cache": True,
        "output_hidden_states": True,
    }
    repeated = torch.zeros(
        (1, policy.vocabulary_tokens), dtype=torch.bool, device=policy.device
    )
    codes: list[torch.Tensor] = []
    logits_history: list[torch.Tensor] = []
    hidden_history: list[torch.Tensor] = []
    for frame_index in range(max_frames):
        output = model(**prepared)
        assert output.generation_step == frame_index
        assert output.past_key_values.get_seq_length() == prefill_tokens + frame_index
        logits = output.logits[:, -1].float()
        assert torch.isfinite(logits).all()
        sampled = policy.sample(
            policy.probabilities(logits, repeated), prefill_tokens - 1 + frame_index
        )
        repeated[:, sampled] = True
        generated_ids = torch.cat((generated_ids, sampled[:, None]), dim=1)
        logits_history.append(logits[0].cpu())
        hidden_history.append(output.hidden_states[0][-1][0, -1].cpu())
        if int(sampled[0]) == policy.eos_token:
            eos = torch.zeros(model.config.num_code_groups, dtype=torch.long)
            eos[0] = policy.eos_token
            codes.append(eos)
            break
        else:
            preparation = dict(
                next_sequence_length=1,
                past_key_values=output.past_key_values,
                attention_mask=torch.ones_like(generated_ids),
                is_first_iteration=False,
                position_ids=torch.tensor(
                    [[[prefill_tokens + frame_index]]], device=policy.device
                ).expand(4, 1, 1),
                generation_step=output.generation_step,
                trailing_text_hidden=text_rows,
                tts_pad_embed=padding,
                output_hidden_states=True,
                use_cache=True,
            )
            if native_predictor is None:
                prepared = model.prepare_inputs_for_generation(
                    generated_ids, hidden_states=output.hidden_states, **preparation
                )
                native_feedback = None
            else:
                assert predictor_batch_rows is not None
                assert (
                    native_predictor.predictor_decode_graph_bucket_size(
                        predictor_batch_rows
                    )
                    == predictor_batch_rows
                )
                query = output.hidden_states[0][-1][:, -1:]
                result = native_predictor.code_predictor_forward_single_token_graph(
                    layer0_codes=sampled[:, None].repeat(predictor_batch_rows, 1),
                    talker_hidden=query.repeat(predictor_batch_rows, 1, 1),
                    batch_size=predictor_batch_rows,
                    code_dtype=sampled.dtype,
                )
                assert result is not None
                native_codes, native_feedback = result
                assert torch.equal(
                    native_codes, native_codes[:1].expand_as(native_codes)
                )
                assert torch.equal(
                    native_feedback, native_feedback[:1].expand_as(native_feedback)
                )
                prepared = GenerationMixin.prepare_inputs_for_generation(
                    model, generated_ids, **preparation
                )
                prepared["residual_codes"] = native_codes[:1, :, 0].clone()
            frame_codes = prepared["residual_codes"][0]
            assert int(frame_codes[0]) == int(sampled[0])
            embeddings = [model.get_input_embeddings()(frame_codes[:1])[None]]
            for codebook_index, embedding_table in enumerate(
                model.code_predictor.get_input_embeddings()
            ):
                embeddings.append(
                    embedding_table(
                        frame_codes[codebook_index + 1 : codebook_index + 2]
                    )[None]
                )
            sequential_feedback = torch.zeros_like(embeddings[0])
            for embedding in embeddings:
                assert embedding.dtype == torch.bfloat16
                sequential_feedback.add_(embedding)
            checkpoint_feedback = torch.cat(embeddings, dim=1).sum(1, keepdim=True)
            text_embedding = (
                text_rows[:, frame_index : frame_index + 1]
                if frame_index < text_rows.shape[1]
                else padding
            )
            if native_predictor is None:
                assert torch.equal(
                    prepared["inputs_embeds"], checkpoint_feedback + text_embedding
                ), frame_index
            else:
                assert native_feedback is not None
                assert torch.equal(
                    native_feedback[:1], sequential_feedback
                ), frame_index
                prepared["inputs_embeds"] = checkpoint_feedback + text_embedding
            if frame_index == 0:
                assert torch.equal(frame_codes.cpu(), conditioning.codec_ids[0])
                assert torch.equal(
                    sequential_feedback.cpu().reshape(-1),
                    conditioning.first_feedback.reshape(-1),
                )
                if use_native_prefix:
                    cache = output.past_key_values
                    assert isinstance(cache, DynamicCache)
                    assert prepared["past_key_values"] is cache
                    assert len(cache.layers) == len(conditioning.prefill_keys)
                    for layer, keys, values in zip(
                        cache.layers,
                        conditioning.prefill_keys,
                        conditioning.prefill_values,
                        strict=True,
                    ):
                        assert type(layer) is DynamicLayer
                        for actual, native in (
                            (layer.keys, keys),
                            (layer.values, values),
                        ):
                            expected = native.transpose(0, 1)[None].to(policy.device)
                            assert actual.shape == expected.shape
                            assert actual.dtype == expected.dtype == torch.bfloat16
                            actual.copy_(expected)
                            assert torch.equal(actual, expected)
                    assert cache.get_seq_length() == prefill_tokens
                else:
                    pass
            else:
                pass
            if use_sequential_feedback:
                prepared["inputs_embeds"] = sequential_feedback + text_embedding
            else:
                pass
            codes.append(frame_codes.cpu())
    else:
        raise RuntimeError(
            f"Reference did not reach codec EOS within {max_frames} frames"
        )
    generated_codes = torch.stack(codes)
    torch.save(
        {
            "codec_ids": generated_codes,
            "logits": torch.stack(logits_history),
            "predictor_queries": torch.stack(hidden_history),
        },
        destination,
    )
    return generated_codes


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.checkpoint).talker_config
    policy = SamplingPolicy(
        seed=arguments.seed,
        temperature=arguments.temperature,
        repetition_penalty=arguments.repetition_penalty,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        vocabulary_tokens=config.text_config.vocab_size,
        eos_token=config.codec_eos_token_id,
        device=device,
    )
    captures = {
        directory.name: load_conditioning(
            directory, policy, arguments.output / directory.name
        )
        for directory in arguments.capture
    }
    if arguments.audit_only:
        return
    else:
        pass
    model = load_talker(arguments.checkpoint, arguments.gpu, arguments.output)
    if arguments.native_predictor_config is None:
        assert arguments.predictor_batch_rows is None
        native_predictor = None
    else:
        assert arguments.predictor_batch_rows is not None
        native_scheduler = load_native_talker(
            arguments.checkpoint,
            arguments.native_predictor_config,
            arguments.gpu,
            arguments.output,
        )
        assert isinstance(native_scheduler.model_runner, QwenTalkerModelRunner)
        native_predictor = native_scheduler.model_runner.model
    reports = []
    for sample_id, conditioning in captures.items():
        previous_codes: torch.Tensor | None = None
        for repeat in range(2):
            generated_codes = generate(
                model,
                conditioning,
                policy,
                arguments.max_frames,
                arguments.output / sample_id / f"reference-{repeat}.pt",
                arguments.native_prefix,
                arguments.sequential_feedback,
                native_predictor,
                arguments.predictor_batch_rows,
            )
            report = {
                "sample_id": sample_id,
                "repeat": repeat,
                "native_prefix": arguments.native_prefix,
                "sequential_feedback": arguments.sequential_feedback,
                "native_predictor": native_predictor is not None,
                "predictor_batch_rows": arguments.predictor_batch_rows,
                "frames_including_eos": len(generated_codes),
                "same_codes_as_previous": (
                    torch.equal(previous_codes, generated_codes)
                    if previous_codes is not None
                    else None
                ),
            }
            reports.append(report)
            previous_codes = generated_codes
            (arguments.output / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)
    if native_predictor is not None:
        torch.distributed.destroy_process_group()
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--native-prefix", action="store_true")
    parser.add_argument("--sequential-feedback", action="store_true")
    parser.add_argument("--native-predictor-config", type=Path)
    parser.add_argument("--predictor-batch-rows", type=int)
    main(parser.parse_args())
else:
    pass
