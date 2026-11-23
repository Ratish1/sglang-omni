# SPDX-License-Identifier: Apache-2.0
"""Compare checkpoint continuation from validated native Talker conditioning."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate
from safetensors import safe_open
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

EXPERT_WEIGHT = re.compile(
    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
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


def audit_weights(
    model: Qwen3OmniMoeTalkerForConditionalGeneration,
    checkpoint: Path,
    destination: Path,
) -> None:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    tensors = model.state_dict(keep_vars=True)
    shards: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if name.startswith("talker."):
            shards.setdefault(shard, []).append(name)
        else:
            pass
    visited: set[int] = set()
    checked_tensors = 0
    for shard, names in sorted(shards.items()):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as weights:
            for checkpoint_name in names:
                expected = weights.get_tensor(checkpoint_name)
                name = checkpoint_name.removeprefix("talker.")
                expert = EXPERT_WEIGHT.fullmatch(name)
                if expert is None:
                    tensor = tensors[name]
                    actual = tensor
                else:
                    prefix, expert_index, projection = expert.groups()
                    if projection == "down_proj":
                        tensor = tensors[f"{prefix}.down_proj"]
                        actual = tensor[int(expert_index)]
                    else:
                        tensor = tensors[f"{prefix}.gate_up_proj"]
                        rows = expected.shape[0]
                        start = 0 if projection == "gate_proj" else rows
                        actual = tensor[int(expert_index), start : start + rows]
                assert actual.dtype == expected.dtype, checkpoint_name
                assert torch.equal(actual, expected.to(actual.device)), checkpoint_name
                visited.add(id(tensor))
                checked_tensors += 1
    assert all(id(tensor) in visited for tensor in tensors.values())
    (destination / "weights.json").write_text(
        json.dumps({"raw_checkpoint_tensors_verified": checked_tensors}, indent=2)
    )


def generate(
    model: Qwen3OmniMoeTalkerForConditionalGeneration,
    conditioning: NativeConditioning,
    policy: SamplingPolicy,
    max_frames: int,
    destination: Path,
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
            prepared = model.prepare_inputs_for_generation(
                generated_ids,
                next_sequence_length=1,
                past_key_values=output.past_key_values,
                attention_mask=torch.ones_like(generated_ids),
                is_first_iteration=False,
                position_ids=torch.tensor(
                    [[[prefill_tokens + frame_index]]], device=policy.device
                ).expand(4, 1, 1),
                hidden_states=output.hidden_states,
                generation_step=output.generation_step,
                trailing_text_hidden=text_rows,
                tts_pad_embed=padding,
                output_hidden_states=True,
                use_cache=True,
            )
            frame_codes = prepared["residual_codes"][0]
            assert int(frame_codes[0]) == int(sampled[0])
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
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    model, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.checkpoint,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    model.eval()
    model.code_predictor.generation_config.do_sample = False
    audit_weights(model, arguments.checkpoint, arguments.output)
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
            )
            report = {
                "sample_id": sample_id,
                "repeat": repeat,
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
    main(parser.parse_args())
else:
    pass
