# SPDX-License-Identifier: Apache-2.0
"""Replay recorded layer-zero expert batches without synthesizing peer rows."""

import argparse
import json
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import torch
import torch.nn.functional as functional
from compare_qwen3_omni_native_trajectories import difference, validate
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import cutlass_fused_moe
from safetensors import safe_open
from sglang.srt.environ import envs
from sglang.srt.layers.activation import silu_and_mul


@torch.inference_mode()
def main(
    root: Path, checkpoint: Path, autotune_cache: Path, tune_max_tokens: int
) -> None:
    torch.set_num_threads(1)
    assert not envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()
    assert AutoTuner.get().load_configs(str(autotune_cache))
    configuration = json.loads((checkpoint / "config.json").read_text())[
        "talker_config"
    ]["text_config"]
    assert configuration["hidden_act"] == "silu"
    expert_count = configuration["num_experts"]
    prefix = "talker.model.layers.0.mlp.experts."
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = [
        f"{prefix}{expert}.{projection}_proj.weight"
        for expert in range(expert_count)
        for projection in ("up", "gate", "down")
    ]
    shared_prefix = "talker.model.layers.0.mlp."
    names.extend(
        f"{shared_prefix}{name}.weight"
        for name in (
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert.down_proj",
            "shared_expert_gate",
            "gate",
        )
    )
    with ExitStack() as stack:
        shards = {
            shard: stack.enter_context(
                safe_open(checkpoint / shard, framework="pt", device="cpu")
            )
            for shard in sorted({weight_map[name] for name in names})
        }
        weights = {
            name: shards[weight_map[name]].get_tensor(name).cuda() for name in names
        }
    assert all(weight.dtype == torch.bfloat16 for weight in weights.values())
    # note (ratish): the unquantized CUTLASS loader stores up before gate.
    up_gate_weights = torch.stack(
        [
            torch.cat(
                (
                    weights[f"{prefix}{expert}.up_proj.weight"],
                    weights[f"{prefix}{expert}.gate_proj.weight"],
                )
            )
            for expert in range(expert_count)
        ]
    )
    down_weights = torch.stack(
        [
            weights[f"{prefix}{expert}.down_proj.weight"]
            for expert in range(expert_count)
        ]
    )
    shared_gate_up_weights = torch.cat(
        (
            weights[f"{shared_prefix}shared_expert.gate_proj.weight"],
            weights[f"{shared_prefix}shared_expert.up_proj.weight"],
        )
    )
    report = {
        "checkpoint": str(checkpoint),
        "autotune_cache_sha256": sha256(autotune_cache.read_bytes()).hexdigest(),
        "tune_max_tokens": tune_max_tokens,
        "scope": "Whole recorded padded input batch, original IDs and scales, selected-row routed/shared outputs, gates and combine. Layer zero only; no generated speech or performance claim.",
        "captures": {},
    }
    total_frames = 0
    total_mismatches = 0
    for capture in sorted(root.glob("*/captures/*/trajectory.pt")):
        frames = validate(capture.parent)
        metadata = json.loads(capture.with_suffix(".json").read_text())["frames"]
        mismatches = []
        checked_frames = 0
        for frame_index, (frame, step) in enumerate(zip(frames, metadata, strict=True)):
            if step["phase"] == "prefill":
                continue
            else:
                assert step["phase"] == "decode"
            inputs = frame["layer0_peer_mlp_normalized"].cuda()
            expert_ids = frame["layer0_peer_expert_ids"].cuda().int()
            expert_weights = frame["layer0_peer_expert_weights"].cuda().float()
            assert inputs.shape == (
                step["graph_batch_rows"],
                configuration["hidden_size"],
            )
            assert (
                expert_ids.shape
                == expert_weights.shape
                == (
                    inputs.shape[0],
                    configuration["num_experts_per_tok"],
                )
            )
            assert torch.equal(
                frame["layer0_peer_mlp_normalized"][step["request_row"]],
                frame["layer0_mlp_normalized"],
            )
            shared_gate = functional.linear(
                inputs, weights[f"{shared_prefix}shared_expert_gate.weight"]
            )
            shared_output = functional.linear(
                silu_and_mul(functional.linear(inputs, shared_gate_up_weights)),
                weights[f"{shared_prefix}shared_expert.down_proj.weight"],
            )
            router_logits = functional.linear(
                inputs, weights[f"{shared_prefix}gate.weight"]
            )
            output = cutlass_fused_moe(
                input=inputs,
                token_selected_experts=expert_ids,
                token_final_scales=expert_weights,
                fc1_expert_weights=up_gate_weights,
                fc2_expert_weights=down_weights,
                output_dtype=inputs.dtype,
                quant_scales=None,
                use_fused_finalize=False,
                tune_max_num_tokens=tune_max_tokens,
            )[0]
            combined_output = output + shared_output * torch.sigmoid(shared_gate)
            for name, values in (
                ("routed_experts", output),
                ("shared_gate", shared_gate),
                ("shared_expert", shared_output),
                ("router_logits", router_logits),
                ("layer_output", combined_output),
            ):
                selected = values[step["request_row"]].cpu()
                actual = frame[f"layer0_{name}"]
                assert torch.isfinite(selected).all()
                if not torch.equal(selected, actual):
                    mismatches.append(
                        {
                            "frame": frame_index,
                            "field": name,
                            **difference(actual, selected),
                        }
                    )
                else:
                    pass
            checked_frames += 1
        report["captures"][str(capture.relative_to(root))] = {
            "checked_frames": checked_frames,
            "mismatches": mismatches,
        }
        total_frames += checked_frames
        total_mismatches += len(mismatches)
        print(
            f"{capture.parent.name}: frames={checked_frames} mismatches={len(mismatches)}",
            flush=True,
        )
    assert total_frames > 0
    report["total_frames"] = total_frames
    report["total_mismatches"] = total_mismatches
    with (root / "moe-replay.json").open("x") as receipt:
        json.dump(report, receipt, indent=2)
    assert total_mismatches == 0, "Native expert replay did not reproduce its control"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("autotune_cache", type=Path)
    parser.add_argument("--tune-max-tokens", type=int, required=True)
    arguments = parser.parse_args()
    main(
        arguments.root,
        arguments.checkpoint,
        arguments.autotune_cache,
        arguments.tune_max_tokens,
    )
else:
    pass
