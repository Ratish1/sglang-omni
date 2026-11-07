# SPDX-License-Identifier: Apache-2.0
"""Check actual residual-predictor logits and codebook choices on saved frames."""

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

from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config
from sglang_omni.vendor.sglang.layers import ReplicatedLinear


@dataclass(kw_only=True)
class PredictorHeadCapture:
    logits: list[torch.Tensor] = field(default_factory=list)

    def record(
        self,
        module: ReplicatedLinear,
        inputs: tuple[torch.Tensor],
        output: tuple[torch.Tensor, None],
    ) -> None:
        self.logits.append(output[0].detach().clone())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    scheduler = create_talker_ar_executor_from_config(
        arguments.model_path,
        gpu_id=arguments.gpu,
        max_running_requests=arguments.max_running_requests,
        total_gpu_memory_fraction=arguments.memory_fraction,
        code2wav_in_process=True,
    )
    native = scheduler.model_runner.model
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    float32_predictor = copy.deepcopy(reference.code_predictor).float().eval()
    capture = PredictorHeadCapture()
    handles = [
        head.register_forward_hook(capture.record)
        for head in native.code_predictor.lm_head
    ]
    reports = []
    with torch.inference_mode():
        for trace_directory in arguments.trace_directory:
            case_directory = arguments.output_directory / trace_directory.name
            case_directory.mkdir()
            for filename in sorted(trace_directory.glob("*-post_decode.pt")):
                saved = torch.load(filename, map_location="cpu", weights_only=True)
                codes = saved["codec_ids"].to(reference.device).long()
                hidden_states = saved["hidden_states"].to(reference.device)
                capture.logits.clear()
                native_codes, _ = native.code_predictor_forward_incremental_eager(
                    codes[:, :1], hidden_states
                )
                native_codes = native_codes.squeeze(-1).clone()
                assert len(capture.logits) == config.num_code_groups - 1
                native_logits = torch.cat(capture.logits, dim=1).float()
                assert torch.isfinite(native_logits).all()
                assert torch.equal(
                    native_logits.argmax(-1), native_codes[:, 1:]
                ), "codebook selection does not match the actual head logits"
                embeddings = [
                    hidden_states[:, None],
                    reference.model.codec_embedding(native_codes[:, :1]),
                ]
                for group in range(config.num_code_groups - 2):
                    embeddings.append(
                        reference.code_predictor.model.codec_embedding[group](
                            native_codes[:, group + 1 : group + 2]
                        )
                    )
                predictor_inputs = torch.cat(embeddings, dim=1)
                reference_logits = []
                for predictor in (reference.code_predictor, float32_predictor):
                    output = predictor.model(
                        inputs_embeds=predictor_inputs.to(predictor.dtype),
                        use_cache=False,
                    ).last_hidden_state
                    reference_logits.append(
                        torch.stack(
                            [
                                head(output[:, group + 1])
                                for group, head in enumerate(predictor.lm_head)
                            ],
                            dim=1,
                        ).float()
                    )
                report = {
                    "trace_file": str(filename),
                    "native_replay_matches_capture": torch.equal(native_codes, codes),
                    "codebook_argmax_exact": True,
                    "native_to_float32_relative_l2_error": float(
                        (native_logits - reference_logits[1]).norm()
                        / reference_logits[1].norm()
                    ),
                    "checkpoint_bfloat16_to_float32_relative_l2_error": float(
                        (reference_logits[0] - reference_logits[1]).norm()
                        / reference_logits[1].norm()
                    ),
                    "groups": [],
                }
                for group in range(config.num_code_groups - 1):
                    chosen = int(native_codes[0, group + 1])
                    report["groups"].append(
                        {
                            "group": group + 1,
                            "native_code": chosen,
                            "captured_code": int(codes[0, group + 1]),
                            "checkpoint_bfloat16_argmax": int(
                                reference_logits[0][0, group].argmax()
                            ),
                            "checkpoint_float32_argmax": int(
                                reference_logits[1][0, group].argmax()
                            ),
                            "checkpoint_float32_chosen_logit_gap": float(
                                reference_logits[1][0, group].max()
                                - reference_logits[1][0, group, chosen]
                            ),
                        }
                    )
                torch.save(
                    {
                        "native_logits": native_logits.cpu(),
                        "checkpoint_bfloat16_logits": reference_logits[0].cpu(),
                        "checkpoint_float32_logits": reference_logits[1].cpu(),
                        "native_codes": native_codes.cpu(),
                        "captured_codes": codes.cpu(),
                    },
                    case_directory / filename.name,
                )
                reports.append(report)
                (arguments.output_directory / "results.json").write_text(
                    json.dumps(reports, indent=2)
                )
                print(json.dumps(report), flush=True)
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
