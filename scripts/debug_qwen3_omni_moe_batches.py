"""Replay original native MoE batches with observed tactics and verified weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights


class WeightFingerprint(BaseModel):
    shape: list[int]
    dtype: str
    sha256: str


class LayerRuntime(BaseModel):
    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    weights: dict[str, WeightFingerprint]


class TacticChoice(BaseModel):
    operation: str
    tensor_shapes: list[list[int]]
    tactic: int
    capturing: bool
    tuning: bool


class NativeRuntime(BaseModel):
    tune_max_num_tokens: int
    use_fused_finalize: bool
    layers: list[LayerRuntime]
    tactic_choices: list[TacticChoice]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--previous-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    runtime = NativeRuntime.model_validate_json(
        (arguments.capture_directory / "moe-runtime.json").read_bytes()
    )
    choices = [
        TacticChoice.model_validate(choice)
        for choice in json.loads(
            (arguments.capture_directory / "moe-choices-final.json").read_text()
        )
    ]
    assert choices[: len(runtime.tactic_choices)] == runtime.tactic_choices
    directory = arguments.capture_directory / arguments.request_id
    previous = arguments.previous_directory / arguments.request_id
    finished = torch.load(directory / "finished.pt", weights_only=True)
    earlier = torch.load(previous / "finished.pt", weights_only=True)
    for name in ("output_ids", "decode_input_history", "enqueued_text_rows"):
        assert torch.equal(finished[name], earlier[name]), name
    initial = torch.load(directory / "0000-before_prefill.pt", weights_only=True)
    prior_initial = torch.load(previous / "0000-before_prefill.pt", weights_only=True)
    assert initial.keys() == prior_initial.keys()
    assert all(
        torch.equal(tensor, prior_initial[name]) for name, tensor in initial.items()
    )
    output_files = sorted(directory.glob("*-output.pt"))
    assert [filename.name for filename in output_files] == [
        f"{frame:04d}-output.pt" for frame in range(len(finished["output_ids"]))
    ]
    for filename in output_files:
        observed = torch.load(filename, weights_only=True)
        old = torch.load(previous / filename.name, weights_only=True)
        assert bool(observed["is_extend"]) == (filename.name == "0000-output.pt")
        assert bool(observed["past_exact"]) and bool(old["past_exact"])
        for name in ("codec_ids", "hidden_states", "logits"):
            assert torch.equal(observed[name], old[name]), (filename.name, name)
        for name in ("keys", "values"):
            assert all(
                torch.equal(current, prior)
                for current, prior in zip(observed[name], old[name], strict=True)
            ), (filename.name, name)
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
        attn_implementation="eager",
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
    native_weights = []
    for layer, layer_runtime in zip(
        reference.model.layers, runtime.layers, strict=True
    ):
        assert layer_runtime.tp_size == layer_runtime.ep_size == 1
        assert layer_runtime.tp_rank == layer_runtime.ep_rank == 0
        gate, up = layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
        packed = torch.cat((up, gate), dim=1).contiguous()
        down = layer.mlp.experts.down_proj.contiguous()
        for name, tensor in (("w13_weight", packed), ("w2_weight", down)):
            fingerprint = layer_runtime.weights[name]
            assert list(tensor.shape) == fingerprint.shape
            assert str(tensor.dtype) == fingerprint.dtype == "torch.bfloat16"
            assert (
                hashlib.sha256(
                    tensor.detach().cpu().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                == fingerprint.sha256
            )
        native_weights.append((packed, down))
    filenames = sorted(directory.glob("*-operators.pt"))
    assert [filename.name for filename in filenames] == [
        f"{frame:04d}-operators.pt" for frame in range(len(finished["output_ids"]))
    ]
    exact_replays = 0
    comparisons = 0
    maximum_replay_error = 0.0
    with (arguments.output_directory / "layers.jsonl").open("w") as output:
        for filename in filenames:
            frame = int(filename.name.split("-")[0])
            phase = "before_prefill" if frame == 0 else "before_decode"
            before = torch.load(
                directory / f"{frame:04d}-{phase}.pt", weights_only=True
            )
            position = (
                before["mrope_positions"][:, -1]
                if frame == 0 and before["mrope_positions"].numel()
                else before["positions"][-1]
            )
            snapshots = torch.load(filename, weights_only=True)
            assert len(snapshots) == len(native_weights)
            for layer_index, (snapshot, (packed, down)) in enumerate(
                zip(snapshots, native_weights, strict=True)
            ):
                row = int(snapshot["batch_selected_row"])
                assert (snapshot["positions"] == position).all()
                capturing = bool(snapshot["batch_is_graph"])
                tensors = {}
                for name in (
                    "mlp_normalized",
                    "router_logits",
                    "expert_indices",
                    "expert_weights",
                    "routed_output",
                ):
                    batch = snapshot[f"batch_{name}"]
                    assert batch.ndim == 2 and torch.isfinite(batch).all()
                    assert torch.equal(batch[row], snapshot[name]), name
                    layout = snapshot[f"layout_{name}"].tolist()
                    assert layout[:2] == list(batch.shape)
                    assert layout[2:4] == list(batch.stride())
                    assert layout[4] == 0
                    tensors[name] = batch.to(reference.device)
                    assert tensors[name].data_ptr() % 16 == 0
                    assert tensors[name].stride() == batch.stride()
                hidden_states = tensors["mlp_normalized"]
                indices = tensors["expert_indices"]
                weights = tensors["expert_weights"]
                assert hidden_states.dtype == torch.bfloat16
                assert indices.dtype == torch.int32 and weights.dtype == torch.float32
                assert indices.shape == weights.shape
                assert indices.shape == (
                    len(hidden_states),
                    configuration.text_config.num_experts_per_tok,
                )
                assert indices.min() >= 0 and indices.max() < len(packed)
                assert weights.min() >= 0
                shapes = [
                    list(hidden_states.shape),
                    list(packed.shape),
                    [],
                    list(down.shape),
                    [],
                ]
                profiles = []
                for operation in (
                    "trtllm::fused_moe::gemm1",
                    "trtllm::fused_moe::gemm2",
                ):
                    tactics = {
                        choice.tactic
                        for choice in choices
                        if choice.operation == operation
                        and choice.tensor_shapes == shapes
                        and choice.capturing == capturing
                        and not choice.tuning
                    }
                    assert len(tactics) == 1, (shapes, capturing, operation, tactics)
                    profiles.append(tactics.pop())
                replayed = []
                for selected_profiles in (profiles, [-1, -1]):
                    replayed.append(
                        cutlass_fused_moe(
                            input=hidden_states,
                            token_selected_experts=indices,
                            token_final_scales=weights,
                            fc1_expert_weights=packed,
                            fc2_expert_weights=down,
                            output_dtype=torch.bfloat16,
                            quant_scales=None,
                            activation_type=ActivationType.Swiglu,
                            tune_max_num_tokens=runtime.tune_max_num_tokens,
                            use_fused_finalize=runtime.use_fused_finalize,
                            profile_ids=selected_profiles,
                        )[0]
                    )
                assert torch.equal(
                    hidden_states.cpu(), snapshot["batch_mlp_normalized"]
                )
                difference = compare_tensors(tensors["routed_output"], replayed[0])
                record = {
                    "frame": int(filename.name.split("-")[0]),
                    "layer": layer_index,
                    "selected_row": row,
                    "batch_tokens": len(hidden_states),
                    "captured_graph": capturing,
                    "profiles": profiles,
                    "native_replay": asdict(difference),
                    "selected_row_replay": asdict(
                        compare_tensors(tensors["routed_output"][row], replayed[0][row])
                    ),
                    "default_vs_observed": asdict(
                        compare_tensors(tensors["routed_output"], replayed[1])
                    ),
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                exact_replays += difference.exact
                comparisons += 1
                maximum_replay_error = max(
                    maximum_replay_error, difference.maximum_absolute_error
                )
            print(
                json.dumps(
                    {
                        "frame": filename.name,
                        "comparisons": comparisons,
                        "exact": exact_replays,
                    }
                ),
                flush=True,
            )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "request_id": arguments.request_id,
                "frames": len(filenames),
                "layer_batches": comparisons,
                "exact_native_replays": exact_replays,
                "maximum_native_replay_error": maximum_replay_error,
                "scope": "Complete original batches, native packed-weight hashes and observed per-geometry tactic choices. Fallback -1 identifies the same internal heuristic request, not a resolved tactic ID. Default replay is a numerical control, not a quality claim or backend recommendation.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
