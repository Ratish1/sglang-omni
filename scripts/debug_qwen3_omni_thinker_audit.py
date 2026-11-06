# SPDX-License-Identifier: Apache-2.0
"""Replay captured Thinker prefills through the checkpoint text backbone."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextDecoderLayer,
)

from scripts.qwen3_omni_audit_utils import compare_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--request-id", action="append")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).thinker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        key_mapping={r"^thinker\.": ""},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    reference_layers: list[torch.Tensor] = []

    def capture_layer(
        module: Qwen3OmniMoeThinkerTextDecoderLayer,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """The fixed hook signature includes the module and its inputs."""
        reference_layers.append(output.detach().squeeze(0).clone())

    hooks = [
        layer.register_forward_hook(capture_layer) for layer in reference.model.layers
    ]
    trace_files = sorted(arguments.trace_root.glob("*/thinker-prefill.pt"))
    if arguments.request_id is not None:
        trace_files = [
            trace_file
            for trace_file in trace_files
            if trace_file.parent.name in arguments.request_id
        ]
        assert len(trace_files) == len(set(arguments.request_id))
    else:
        pass
    assert trace_files, arguments.trace_root
    reports = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for trace_file in trace_files:
            captured = torch.load(trace_file, weights_only=True)
            token_count = int(captured["live_tokens"])
            embeddings = captured["input_embeds"][:token_count].to(device)
            positions = captured["positions"][..., :token_count].to(device)
            if positions.ndim == 1:
                positions = positions.reshape(1, 1, -1).expand(3, 1, -1)
            else:
                assert positions.shape[0] == 3 and positions.ndim == 2
                positions = positions.unsqueeze(1)
            reference_layers.clear()
            visual_arguments = {}
            if "deepstack" in captured:
                visual_masks = captured["visual_masks"][:token_count].to(device).bool()
                dense_features = captured["deepstack"][:token_count].to(device)
                visual_arguments = {
                    "visual_pos_masks": visual_masks.unsqueeze(0),
                    "deepstack_visual_embeds": [
                        features[visual_masks]
                        for features in dense_features.split(
                            config.text_config.hidden_size, dim=-1
                        )
                    ],
                }
            else:
                pass
            output = reference.model(
                inputs_embeds=embeddings.unsqueeze(0),
                position_ids=positions,
                use_cache=True,
                **visual_arguments,
            )
            assert len(reference_layers) == config.text_config.num_hidden_layers
            reference_logits = reference.lm_head(
                output.last_hidden_state[:, -1]
            ).float()
            native_logits = captured["logits"].to(device).float()
            probabilities = reference_logits.softmax(-1)
            native_probabilities = native_logits.softmax(-1)
            report = {
                "request_id": trace_file.parent.name,
                "trace_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
                "token_count": token_count,
                "has_visual_deepstack": "deepstack" in captured,
                "layer_errors": [
                    asdict(
                        compare_tensors(
                            captured[f"layer_{index}"][:token_count].to(device),
                            expected,
                        )
                    )
                    for index, expected in enumerate(reference_layers)
                ],
                "final_hidden_error": asdict(
                    compare_tensors(
                        captured["final"][:token_count].to(device),
                        output.last_hidden_state[0],
                    )
                ),
                "logits_error": asdict(
                    compare_tensors(native_logits, reference_logits)
                ),
                "raw_probability_l1_difference": float(
                    (probabilities - native_probabilities).abs().sum()
                ),
                "reference_top_tokens": reference_logits[0].topk(10).indices.tolist(),
                "native_top_tokens": native_logits[0].topk(10).indices.tolist(),
            }
            reports.append(report)
            (arguments.output_directory / f"{trace_file.parent.name}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(
                json.dumps(
                    {
                        "request_id": trace_file.parent.name,
                        "layers": len(reference_layers),
                        "final_hidden_error": report["final_hidden_error"],
                        "probability_l1": report["raw_probability_l1_difference"],
                    }
                ),
                flush=True,
            )
    for hook in hooks:
        hook.remove()
    source = Path(inspect.getfile(type(reference)))
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "model_path": arguments.model_path,
                "transformers": version("transformers"),
                "reference_source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "reference_attention": "sdpa_math",
                "reference_experts": "eager",
                "requests": reports,
                "scope": "Whole uncached single-request prefills with captured media-scattered embeddings, M-RoPE and visual deepstack. This does not validate the media encoders or free-generation quality, and does not claim a numerical tolerance passed.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
