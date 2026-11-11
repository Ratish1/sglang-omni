"""Compare residual codes using actual native queries and every committed prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
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
    reports = []
    assert {
        directory.name for directory in arguments.capture_directory.iterdir()
    } == set(arguments.request_id)
    for request_id in arguments.request_id:
        directory = arguments.capture_directory / request_id
        destination = arguments.output_directory / directory.name
        destination.mkdir()
        finished = torch.load(directory / "finished.pt", weights_only=True)
        assert int(finished["output_ids"][-1]) == configuration.codec_eos_token_id
        output_files = sorted(directory.glob("*-output.pt"))
        assert [filename.name for filename in output_files] == [
            f"{frame:04d}-output.pt" for frame in range(len(finished["output_ids"]))
        ]
        rows = []
        for frame, filename in enumerate(output_files):
            observed = torch.load(filename, weights_only=True)
            codes = observed["codec_ids"].unsqueeze(0).to(reference.device).long()
            assert codes.shape == (1, configuration.num_code_groups)
            hidden_states = observed["hidden_states"].unsqueeze(0).to(reference.device)
            assert hidden_states.shape == (1, configuration.text_config.hidden_size)
            assert int(codes[0, 0]) == int(finished["output_ids"][frame])
            projected = reference.codec_head(hidden_states).float()
            actual_logits = (
                (
                    torch.load(directory / "0000-prefill-logits.pt", weights_only=True)
                    if frame == 0
                    else observed["logits"]
                )
                .unsqueeze(0)
                .to(reference.device)
            )
            finite_columns = actual_logits.isfinite()
            head_difference = (projected - actual_logits)[finite_columns]
            assert projected.isfinite().all()
            is_eos = int(codes[0, 0]) == configuration.codec_eos_token_id
            if is_eos:
                assert frame == len(output_files) - 1
                groups = []
            else:
                assert bool(
                    (
                        (codes >= 0)
                        & (codes < configuration.code_predictor_config.vocab_size)
                    ).all()
                )
                embeddings = [
                    hidden_states.unsqueeze(1),
                    reference.model.codec_embedding(codes[:, :1]),
                ]
                for group in range(configuration.num_code_groups - 2):
                    embeddings.append(
                        reference.code_predictor.model.codec_embedding[group](
                            codes[:, group + 1 : group + 2]
                        )
                    )
                predictor_inputs = torch.cat(embeddings, dim=1)
                complete = reference.code_predictor.model(
                    inputs_embeds=predictor_inputs, use_cache=False
                ).last_hidden_state
                repeated = reference.code_predictor.model(
                    inputs_embeds=predictor_inputs, use_cache=False
                ).last_hidden_state
                cache = DynamicCache(config=configuration.code_predictor_config)
                incremental = reference.code_predictor.model(
                    inputs_embeds=predictor_inputs[:, :2],
                    past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state
                groups = []
                for group, head in enumerate(reference.code_predictor.lm_head):
                    prefill_logits = head(complete[:, group + 1]).float()
                    repeated_logits = head(repeated[:, group + 1]).float()
                    incremental_logits = head(incremental[:, -1]).float()
                    assert prefill_logits.isfinite().all()
                    assert incremental_logits.isfinite().all()
                    chosen = int(codes[0, group + 1])
                    groups.append(
                        {
                            "group": group + 1,
                            "native_code": chosen,
                            "prefill_argmax": int(prefill_logits[0].argmax()),
                            "prefill_native_rank": int(
                                (prefill_logits[0] > prefill_logits[0, chosen]).sum()
                            )
                            + 1,
                            "prefill_native_logit_gap": float(
                                prefill_logits[0].max() - prefill_logits[0, chosen]
                            ),
                            "incremental_argmax": int(incremental_logits[0].argmax()),
                            "incremental_native_rank": int(
                                (
                                    incremental_logits[0]
                                    > incremental_logits[0, chosen]
                                ).sum()
                            )
                            + 1,
                            "incremental_native_logit_gap": float(
                                incremental_logits[0].max()
                                - incremental_logits[0, chosen]
                            ),
                            "prefill_repeat_exact": torch.equal(
                                prefill_logits, repeated_logits
                            ),
                            "prefill_repeat_maximum_error": float(
                                (prefill_logits - repeated_logits).abs().max()
                            ),
                            "prefill_incremental_maximum_error": float(
                                (prefill_logits - incremental_logits).abs().max()
                            ),
                        }
                    )
                    if group + 1 < len(reference.code_predictor.lm_head):
                        incremental = reference.code_predictor.model(
                            inputs_embeds=predictor_inputs[:, group + 2 : group + 3],
                            past_key_values=cache,
                            use_cache=True,
                        ).last_hidden_state
                    else:
                        pass
                assert cache.get_seq_length() == configuration.num_code_groups
            row = {
                "frame": frame,
                "is_eos": is_eos,
                "head_reprojection_exact": bool((head_difference == 0).all()),
                "head_reprojection_maximum_error": float(head_difference.abs().max()),
                "groups": groups,
            }
            rows.append(row)
            with (destination / "frames.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")
        report = {
            "request_id": directory.name,
            "frames": len(rows),
            "head_reprojection_exact": all(
                row["head_reprojection_exact"] for row in rows
            ),
            "residual_choices": sum(len(row["groups"]) for row in rows),
            "scope": "Actual native backbone query and complete native residual-code prefix, independently replayed as full prefill and incremental checkpoint execution. Full-prefill repeats measure reference nonrepeatability. Reference comparison, not assumed numerical gold or new free-running generation. EOS residuals are excluded because they are not spoken.",
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
