# SPDX-License-Identifier: Apache-2.0
"""Teacher-force captured residual-code prefixes through every predictor layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

from scripts.qwen3_omni_audit_utils import compare_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-file", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    reference, loading = (
        Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration.from_pretrained(
            arguments.model_path,
            config=config.code_predictor_config,
            dtype=torch.bfloat16,
            device_map={"": arguments.gpu},
            key_mapping={r"^talker\.code_predictor\.": ""},
            attn_implementation="sdpa",
            use_kernels=False,
            output_loading_info=True,
        )
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    layer_inputs: list[torch.Tensor] = []
    layer_outputs: list[torch.Tensor] = []

    def capture_layer(
        module: Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """The fixed hook signature includes the module and its inputs."""
        layer_inputs.append(inputs[0].detach().clone())
        layer_outputs.append(output.detach().clone())

    hooks = [
        layer.register_forward_hook(capture_layer) for layer in reference.model.layers
    ]
    reports = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for trace_file in arguments.trace_file:
            captured = torch.load(trace_file, weights_only=True)
            codes = captured["codec_ids"].to(device)
            hidden_states = captured["hidden_states"]
            candidates = [
                int(name.rsplit("_", 1)[-1])
                for name, tensor in captured.items()
                if name.startswith("predictor_input_0_")
                and torch.equal(tensor[:1], hidden_states.reshape(1, -1))
            ]
            assert candidates, trace_file
            pair_rows = min(candidates)
            assert pair_rows % 2 == 0
            batch_rows = pair_rows // 2
            prefix = (
                captured[f"predictor_input_0_{pair_rows}"][:2].to(device).unsqueeze(0)
            )
            embedding_rows = [prefix[:, 1]]
            groups = []
            for group in range(config.num_code_groups - 1):
                layer_inputs.clear()
                layer_outputs.clear()
                output = reference.model(inputs_embeds=prefix, use_cache=False)
                assert (
                    len(layer_outputs) == config.code_predictor_config.num_hidden_layers
                )
                logits = reference.lm_head[group](output.last_hidden_state[:, -1:])
                actual_logits = captured[f"predictor_logits_{group}_{batch_rows}"][
                    0, -1
                ].to(device)
                native_code = int(codes[0, group + 1])
                assert int(actual_logits.argmax()) == native_code
                expected_code = int(logits[0, -1].argmax())
                cache_tokens = 0 if group == 0 else group + 1
                new_tokens = 2 if group == 0 else 1
                active_rows = pair_rows if group == 0 else batch_rows
                groups.append(
                    {
                        "group": group + 1,
                        "native_code": native_code,
                        "reference_code": expected_code,
                        "native_code_reference_rank": int(
                            (logits[0, -1] > logits[0, -1, native_code]).sum()
                        ),
                        "reference_margin_against_native_code": float(
                            logits[0, -1, expected_code] - logits[0, -1, native_code]
                        ),
                        "logits_error": asdict(
                            compare_tensors(actual_logits, logits[0, -1])
                        ),
                        "layers": [
                            {
                                "layer": layer_index,
                                "input_error": asdict(
                                    compare_tensors(
                                        captured[
                                            f"predictor_layer_input_{layer_index}_{cache_tokens}_{active_rows}"
                                        ][:new_tokens].to(device),
                                        expected_input[0, -new_tokens:],
                                    )
                                ),
                                "output_error": asdict(
                                    compare_tensors(
                                        captured[
                                            f"predictor_layer_output_{layer_index}_{cache_tokens}_{active_rows}"
                                        ][:new_tokens].to(device),
                                        expected_output[0, -new_tokens:],
                                    )
                                ),
                            }
                            for layer_index, (
                                expected_input,
                                expected_output,
                            ) in enumerate(
                                zip(layer_inputs, layer_outputs, strict=True)
                            )
                        ],
                        "final_hidden_error": asdict(
                            compare_tensors(
                                captured[
                                    f"predictor_final_{cache_tokens}_{active_rows}"
                                ][:new_tokens].to(device),
                                output.last_hidden_state[0, -new_tokens:],
                            )
                        ),
                    }
                )
                next_embedding = reference.model.codec_embedding[group](
                    codes[:, group + 1 : group + 2]
                )
                embedding_rows.append(next_embedding[:, 0])
                prefix = torch.cat((prefix, next_embedding), dim=1)
            incremental_sum = torch.zeros_like(embedding_rows[0])
            for embedding in embedding_rows:
                incremental_sum.add_(embedding)
            feedback = captured["codec_feedback"].to(device)
            report = {
                "trace_file": str(trace_file),
                "trace_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
                "captured_batch_rows": batch_rows,
                "groups": groups,
                "feedback_incremental_error": asdict(
                    compare_tensors(feedback, incremental_sum)
                ),
                "feedback_tree_reduction_error": asdict(
                    compare_tensors(feedback, torch.stack(embedding_rows, dim=1).sum(1))
                ),
            }
            reports.append(report)
            (arguments.output_directory / f"{trace_file.stem}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(
                json.dumps(
                    {
                        "trace_file": str(trace_file),
                        "groups": len(groups),
                        "matching_codes": sum(
                            group["native_code"] == group["reference_code"]
                            for group in groups
                        ),
                        "feedback_incremental_error": report[
                            "feedback_incremental_error"
                        ],
                    }
                ),
                flush=True,
            )
    for hook in hooks:
        hook.remove()
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "model_path": arguments.model_path,
                "requests": reports,
                "scope": "Every residual group is conditioned on the same captured native codec prefix. Layer input/output, logits, deterministic argmax and both feedback reductions are measured. This is not a free-running sampling or speech-quality verdict.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
