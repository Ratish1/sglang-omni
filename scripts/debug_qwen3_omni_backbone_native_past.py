"""Compare complete captured Talker trajectories while holding native past KV fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from transformers import Qwen3OmniMoeConfig
from transformers.cache_utils import DynamicCache
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights


class SamplingParameters(BaseModel):
    temperature: float
    repetition_penalty: float
    suppress_tokens: list[int]


def log_probabilities(
    logits: torch.Tensor,
    history: torch.Tensor,
    allowed: torch.Tensor,
    parameters: SamplingParameters,
) -> torch.Tensor:
    repetition_penalty = torch.full(
        (1, 1), parameters.repetition_penalty, dtype=logits.dtype, device=logits.device
    )
    temperature = torch.full(
        (1, 1), parameters.temperature, dtype=logits.dtype, device=logits.device
    )
    repeated = torch.zeros_like(logits, dtype=torch.bool)
    repeated[:, history] = True
    penalties = torch.where(
        logits > 0,
        logits / repetition_penalty,
        logits * repetition_penalty,
    )
    processed = torch.where(repeated, penalties, logits)
    return torch.log_softmax(processed[:, allowed] / temperature, dim=-1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    register_checkpoint_conversion_mapping(
        configuration.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=configuration,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
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
    rotary = reference.model.rotary_emb
    expected_frequencies, _ = rotary.compute_default_rope_parameters(rotary.config)
    assert rotary.inv_freq.dtype == torch.float32
    assert torch.equal(rotary.inv_freq, expected_frequencies.to(rotary.inv_freq.device))
    reports = []
    for directory in sorted(arguments.capture_directory.iterdir()):
        initial = torch.load(directory / "0000-before_prefill.pt", weights_only=True)
        finished = torch.load(directory / "finished.pt", weights_only=True)
        parameters = SamplingParameters.model_validate_json(
            (directory / "sampling.json").read_bytes()
        )
        assert parameters.temperature > 0
        history = finished["decode_input_history"]
        output_files = sorted(directory.glob("*-output.pt"))
        assert len(output_files) == len(finished["output_ids"]) == len(history) + 1
        assert [filename.name for filename in output_files] == [
            f"{frame:04d}-output.pt" for frame in range(len(output_files))
        ]
        allowed = torch.ones(
            configuration.text_config.vocab_size,
            device=reference.device,
            dtype=torch.bool,
        )
        allowed[parameters.suppress_tokens] = False
        allowed_ids = torch.arange(len(allowed), device=reference.device)[allowed]
        for cache_owner in ("checkpoint", "native"):
            destination = arguments.output_directory / directory.name / cache_owner
            destination.mkdir(parents=True)
            cache = DynamicCache(config=configuration.text_config)
            native_keys: list[torch.Tensor] = []
            native_values: list[torch.Tensor] = []
            frames = []
            for frame, output_file in enumerate(output_files):
                observed = torch.load(output_file, weights_only=True)
                assert int(observed["codec_ids"][0]) == int(
                    finished["output_ids"][frame]
                )
                if frame == 0:
                    embeddings = (
                        initial["prefill_embeds"].unsqueeze(0).to(reference.device)
                    )
                    actual = (
                        torch.load(
                            directory / "0000-prefill-logits.pt", weights_only=True
                        )
                        .unsqueeze(0)
                        .to(reference.device)
                    )
                else:
                    embeddings = history[frame - 1].view(1, 1, -1).to(reference.device)
                    actual = observed["logits"].unsqueeze(0).to(reference.device)
                previous_tokens = cache.get_seq_length()
                assert previous_tokens + embeddings.shape[1] == int(
                    observed["sequence_length"]
                )
                positions = (
                    torch.arange(
                        previous_tokens,
                        previous_tokens + embeddings.shape[1],
                        device=reference.device,
                    )
                    .view(1, 1, -1)
                    .expand(4, 1, -1)
                )
                phase = "before_prefill" if frame == 0 else "before_decode"
                before = torch.load(
                    directory / f"{frame:04d}-{phase}.pt", weights_only=True
                )
                assert torch.equal(positions[0, 0].cpu(), before["positions"])
                output = reference.model(
                    inputs_embeds=embeddings,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                expected = reference.codec_head(output.last_hidden_state[:, -1]).float()
                assert expected.isfinite().all() and actual[:, allowed].isfinite().all()
                assert output.hidden_states is not None
                torch.save(
                    {
                        "logits": expected.cpu(),
                        "hidden_states": [
                            tensor.cpu() for tensor in output.hidden_states
                        ],
                    },
                    destination / output_file.name,
                )
                if cache_owner == "native":
                    cache.crop(-embeddings.shape[1])
                    assert cache.get_seq_length() == previous_tokens
                    repeated = reference.model(
                        inputs_embeds=embeddings,
                        position_ids=positions,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    repeated_logits = reference.codec_head(
                        repeated.last_hidden_state[:, -1]
                    ).float()
                    repeat_exact = torch.equal(expected, repeated_logits)
                    repeat_maximum_error = float(
                        (expected - repeated_logits).abs().max()
                    )
                    cache.crop(-embeddings.shape[1])
                    assert (
                        len(observed["keys"])
                        == len(observed["values"])
                        == (configuration.text_config.num_hidden_layers)
                    )
                    for layer, (keys, values) in enumerate(
                        zip(observed["keys"], observed["values"], strict=True)
                    ):
                        assert (
                            keys.shape
                            == values.shape
                            == (
                                embeddings.shape[1],
                                configuration.text_config.num_key_value_heads,
                                configuration.text_config.head_dim,
                            )
                        )
                        assert keys.dtype == values.dtype == torch.bfloat16
                        if frame == 0:
                            native_keys.append(keys)
                            native_values.append(values)
                        else:
                            native_keys[layer] = torch.cat((native_keys[layer], keys))
                            native_values[layer] = torch.cat(
                                (native_values[layer], values)
                            )
                        cache.update(
                            keys.permute(1, 0, 2).unsqueeze(0).to(reference.device),
                            values.permute(1, 0, 2).unsqueeze(0).to(reference.device),
                            layer,
                        )
                        assert torch.equal(
                            cache.layers[layer].keys.cpu(),
                            native_keys[layer].permute(1, 0, 2).unsqueeze(0),
                        )
                        assert torch.equal(
                            cache.layers[layer].values.cpu(),
                            native_values[layer].permute(1, 0, 2).unsqueeze(0),
                        )
                else:
                    repeat_exact = None
                    repeat_maximum_error = None
                assert cache.get_seq_length() == int(observed["sequence_length"])
                committed = finished["output_ids"][:frame].to(reference.device)
                native_distribution = log_probabilities(
                    actual, committed, allowed, parameters
                )
                expected_distribution = log_probabilities(
                    expected, committed, allowed, parameters
                )
                chosen = int(observed["codec_ids"][0])
                chosen_column = (allowed_ids == chosen).nonzero(as_tuple=True)[0]
                assert len(chosen_column) == 1
                difference = actual[:, allowed] - expected[:, allowed]
                row = {
                    "frame": frame,
                    "position": int(positions[0, 0, -1]),
                    "main_code": chosen,
                    "maximum_absolute_logit_error": float(difference.abs().max()),
                    "relative_logit_error": float(
                        difference.norm() / expected[:, allowed].norm()
                    ),
                    "kl_native_to_reference_pretruncation": float(
                        (
                            native_distribution.exp()
                            * (native_distribution - expected_distribution)
                        ).sum()
                    ),
                    "chosen_native_pretruncation_probability": float(
                        native_distribution[0, chosen_column].exp()
                    ),
                    "chosen_reference_pretruncation_probability": float(
                        expected_distribution[0, chosen_column].exp()
                    ),
                    "chosen_reference_rank": int(
                        (
                            expected_distribution
                            > expected_distribution[0, chosen_column]
                        ).sum()
                    )
                    + 1,
                    "reference_repeat_exact": repeat_exact,
                    "reference_repeat_maximum_error": repeat_maximum_error,
                }
                frames.append(row)
                with (destination / "frames.jsonl").open("a") as trace:
                    trace.write(json.dumps(row) + "\n")
            report = {
                "request_id": directory.name,
                "cache_owner": cache_owner,
                "frames": len(frames),
                "maximum_pretruncation_kl": max(
                    row["kl_native_to_reference_pretruncation"] for row in frames
                ),
                "maximum_relative_logit_error": max(
                    row["relative_logit_error"] for row in frames
                ),
                "nonrepeatable_reference_frames": (
                    sum(row["reference_repeat_exact"] is False for row in frames)
                    if cache_owner == "native"
                    else None
                ),
                "reference_repeat_frames": (
                    len(frames) if cache_owner == "native" else 0
                ),
                "scope": "Identical captured native inputs at every step. Native-cache mode replaces only reference past KV after each step, eliminating cross-frame reference drift. Reference comparison, not a gold-standard correctness or performance claim.",
            }
            reports.append(report)
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
