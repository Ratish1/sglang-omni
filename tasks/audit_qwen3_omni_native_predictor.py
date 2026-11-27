# SPDX-License-Identifier: Apache-2.0
"""Check native predictor replay against captured codes and recurrent feedback."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from os import environ
from pathlib import Path
from typing import Literal

import torch
from compare_qwen3_omni_native_trajectories import validate

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from sglang_omni.utils.connection import find_available_port


@dataclass(kw_only=True)
class FrameReplay:
    frame_index: int
    batch_rows: int
    request_row: int
    duplicated_peers: bool
    changed_codebooks: list[int]
    changed_feedback_values: int
    maximum_feedback_difference: float
    codes_sha256: str
    feedback_sha256: str


@dataclass(kw_only=True)
class BufferReuseCheck:
    sample_id: str
    frame_index: int
    live_rows: int
    execution: Literal["eager", "graph"]
    changed_codes_after_previous_frame: int
    changed_feedback_after_previous_frame: int
    changed_cache_after_previous_frame: int
    changed_codes_after_nan_fill: int
    changed_feedback_after_nan_fill: int
    changed_cache_after_nan_fill: int


def predictor_snapshot(
    model: Qwen3OmniTalker,
    codes: torch.Tensor,
    queries: torch.Tensor,
    execution: Literal["eager", "graph"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if execution == "graph":
        result = model.code_predictor_forward_single_token_graph(
            layer0_codes=codes,
            talker_hidden=queries,
            batch_size=codes.shape[0],
            code_dtype=codes.dtype,
        )
        assert result is not None
    else:
        result = model.code_predictor_forward_incremental_eager(codes, queries)
    written_slots = model.config.num_code_groups
    return (
        result[0].clone(),
        result[1].clone(),
        model.predictor_k_cache[:, : codes.shape[0], :written_slots].clone(),
        model.predictor_v_cache[:, : codes.shape[0], :written_slots].clone(),
    )


def check_buffer_reuse(
    model: Qwen3OmniTalker,
    directories: list[Path],
    device: torch.device,
    output: Path,
) -> None:
    fused = model.predictor_fused_layers
    assert fused is not None
    scratch = (
        model.predictor_input_buffer,
        model.predictor_k_cache,
        model.predictor_v_cache,
        model.output_embeds,
        fused.q,
        fused.attention_output,
        fused.residual,
        fused.activated,
        fused.partials,
        fused.sum_sq_partials,
    )
    checks: list[BufferReuseCheck] = []
    for directory in directories:
        frames = validate(directory)[:-1]
        queries = torch.stack(
            [frame["predictor_query"].reshape(1, -1) for frame in frames]
        ).to(device)
        codes = torch.stack([frame["codec_ids"][:1] for frame in frames]).to(device)
        for live_rows in range(1, model.predictor_input_buffer.shape[0] + 1):
            for frame_index in range(len(frames)):
                indices = (torch.arange(live_rows, device=device) + frame_index) % len(
                    frames
                )
                current_queries = queries[indices]
                current_codes = codes[indices]
                previous_indices = (indices + 1) % len(frames)
                for execution in ("eager", "graph"):
                    assert not fused.counters.count_nonzero()
                    for buffer in scratch:
                        buffer.zero_()
                    (
                        expected_codes,
                        expected_feedback,
                        expected_keys,
                        expected_values,
                    ) = predictor_snapshot(
                        model, current_codes, current_queries, execution
                    )
                    predictor_snapshot(
                        model,
                        codes[previous_indices],
                        queries[previous_indices],
                        execution,
                    )
                    after_codes, after_feedback, after_keys, after_values = (
                        predictor_snapshot(
                            model, current_codes, current_queries, execution
                        )
                    )
                    assert not fused.counters.count_nonzero()
                    # note (ratish): only overwritten scratch is poisoned, never the live synchronization counters.
                    for buffer in scratch:
                        buffer.fill_(float("nan"))
                    (
                        poisoned_codes,
                        poisoned_feedback,
                        poisoned_keys,
                        poisoned_values,
                    ) = predictor_snapshot(
                        model, current_codes, current_queries, execution
                    )
                    assert torch.isfinite(poisoned_feedback).all()
                    assert torch.isfinite(poisoned_keys).all()
                    assert torch.isfinite(poisoned_values).all()
                    assert not fused.counters.count_nonzero()
                    assert torch.equal(current_queries, queries[indices])
                    assert torch.equal(current_codes, codes[indices])
                    check = BufferReuseCheck(
                        sample_id=directory.name,
                        frame_index=frame_index,
                        live_rows=live_rows,
                        execution=execution,
                        changed_codes_after_previous_frame=int(
                            after_codes.ne(expected_codes).sum()
                        ),
                        changed_feedback_after_previous_frame=int(
                            after_feedback.ne(expected_feedback).sum()
                        ),
                        changed_cache_after_previous_frame=int(
                            after_keys.ne(expected_keys).sum()
                            + after_values.ne(expected_values).sum()
                        ),
                        changed_codes_after_nan_fill=int(
                            poisoned_codes.ne(expected_codes).sum()
                        ),
                        changed_feedback_after_nan_fill=int(
                            poisoned_feedback.ne(expected_feedback).sum()
                        ),
                        changed_cache_after_nan_fill=int(
                            poisoned_keys.ne(expected_keys).sum()
                            + poisoned_values.ne(expected_values).sum()
                        ),
                    )
                    checks.append(check)
                    assert not any(
                        (
                            check.changed_codes_after_previous_frame,
                            check.changed_feedback_after_previous_frame,
                            check.changed_cache_after_previous_frame,
                            check.changed_codes_after_nan_fill,
                            check.changed_feedback_after_nan_fill,
                            check.changed_cache_after_nan_fill,
                        )
                    ), asdict(check)
            with (output / "buffer-reuse.json").open("w") as report:
                json.dump({"checks": [asdict(check) for check in checks]}, report)
            print(
                f"Buffer reuse passed: {directory.name}, live rows {live_rows}",
                flush=True,
            )


def load_native_talker(
    checkpoint: Path, config_path: Path, gpu_index: int, destination: Path
) -> QwenTalkerScheduler:
    config = ConfigManager.from_file(
        str(config_path), model_path_override=str(checkpoint)
    ).config
    assert config is not None
    factory_arguments = resolve_stage_factory_args(
        config.stage_named("talker_ar"), config, gpu_id=gpu_index
    )
    factory_arguments["nccl_port"] = find_available_port()
    scheduler = create_talker_ar_executor_from_config(**factory_arguments)
    runner = scheduler.model_runner
    assert isinstance(runner, QwenTalkerModelRunner)
    model = runner.model
    (destination / "native-runtime.json").write_text(
        json.dumps(
            {
                "factory_arguments": factory_arguments,
                "predictor_graph_batch_rows": model.predictor_decode_graph_batch_sizes,
                "fused_predictor_layers": model.predictor_fused_layers is not None,
                "exact_add_norm": model.predictor_exact_add_norm,
                "triton_attention": model.predictor_triton_attention,
                "rope_stores_kv": model.predictor_rope_stores_kv,
            },
            indent=2,
        )
    )
    return scheduler


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    environ["OMNI_NATIVE_TRAJECTORY_DIRECTORY"] = str(arguments.output)
    environ["OMNI_NATIVE_TRAJECTORY_SAMPLES"] = ",".join(
        directory.name for directory in arguments.capture
    )
    device = torch.device("cuda", arguments.gpu)
    torch.cuda.set_device(device)
    scheduler = load_native_talker(
        arguments.checkpoint, arguments.config, arguments.gpu, arguments.output
    )
    runner = scheduler.model_runner
    assert isinstance(runner, QwenTalkerModelRunner)
    model = runner.model
    comparisons: dict[str, dict[tuple[int, bool], FrameReplay]] = {}
    for repeat in range(2):
        directories = arguments.capture if repeat == 0 else arguments.capture[::-1]
        for directory in directories:
            frames = validate(directory)
            metadata = json.loads((directory / "trajectory.json").read_text())
            assert int(frames[-1]["codec_ids"][0]) == model.config.codec_eos_token_id
            frame_count = len(frames) - 1
            if repeat == 0:
                comparisons[directory.name] = {}
                frame_indices = range(frame_count)
            else:
                frame_indices = range(frame_count - 1, -1, -1)
            for frame_index in frame_indices:
                frame = frames[frame_index]
                state = metadata["frames"][frame_index]
                request_row = state["request_row"]
                if arguments.fixed_batch_rows is not None:
                    batch_rows = live_rows = arguments.fixed_batch_rows
                    use_graph = True
                elif state["phase"] == "prefill":
                    live_rows = len(state["request_ids"])
                    batch_rows = model.predictor_decode_graph_bucket_size(live_rows)
                    assert batch_rows is not None
                    use_graph = True
                else:
                    batch_rows = state["graph_batch_rows"]
                    live_rows = batch_rows
                    use_graph = state["cuda_graph"]
                assert 0 <= request_row < live_rows <= batch_rows
                query = frame["predictor_query"].to(device).reshape(1, 1, -1)
                main_code = frame["codec_ids"][:1].to(device).reshape(1, 1)
                assert query.dtype == torch.bfloat16
                assert main_code.dtype == torch.int64
                for duplicate_peers in (False, True):
                    if duplicate_peers:
                        queries = query.repeat(live_rows, 1, 1)
                        main_codes = main_code.repeat(live_rows, 1)
                    else:
                        queries = query.new_zeros(live_rows, 1, query.shape[-1])
                        main_codes = main_code.new_zeros(live_rows, 1)
                        queries[request_row].copy_(query[0])
                        main_codes[request_row].copy_(main_code[0])
                    if use_graph:
                        assert (
                            model.predictor_decode_graph_bucket_size(live_rows)
                            == batch_rows
                        )
                        result = model.code_predictor_forward_single_token_graph(
                            layer0_codes=main_codes,
                            talker_hidden=queries,
                            batch_size=live_rows,
                            code_dtype=main_codes.dtype,
                        )
                        assert result is not None
                        codes, feedback = result
                    else:
                        codes, feedback = (
                            model.code_predictor_forward_incremental_eager(
                                main_codes, queries
                            )
                        )
                    actual_codes = codes[request_row].flatten().cpu()
                    actual_feedback = feedback[request_row].flatten().cpu()
                    if duplicate_peers:
                        assert torch.equal(
                            codes, codes[request_row : request_row + 1].expand_as(codes)
                        )
                        assert torch.equal(
                            feedback,
                            feedback[request_row : request_row + 1].expand_as(feedback),
                        )
                    else:
                        pass
                    expected_codes = frame["codec_ids"].flatten()
                    expected_feedback = frame["feedback"].flatten()
                    assert torch.isfinite(actual_feedback).all()
                    assert actual_feedback.dtype == expected_feedback.dtype
                    comparison = FrameReplay(
                        frame_index=frame_index,
                        batch_rows=batch_rows,
                        request_row=request_row,
                        duplicated_peers=duplicate_peers,
                        changed_codebooks=(actual_codes != expected_codes)
                        .nonzero()
                        .flatten()
                        .tolist(),
                        changed_feedback_values=int(
                            actual_feedback.ne(expected_feedback).sum()
                        ),
                        maximum_feedback_difference=float(
                            (actual_feedback.float() - expected_feedback.float())
                            .abs()
                            .max()
                        ),
                        codes_sha256=sha256(actual_codes.numpy().tobytes()).hexdigest(),
                        feedback_sha256=sha256(
                            actual_feedback.view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    )
                    if repeat == 0:
                        comparisons[directory.name][
                            frame_index, duplicate_peers
                        ] = comparison
                    else:
                        assert (
                            comparison
                            == comparisons[directory.name][frame_index, duplicate_peers]
                        ), (directory.name, frame_index, duplicate_peers)
            ordered = [
                comparisons[directory.name][frame_index, duplicate_peers]
                for frame_index in range(frame_count)
                for duplicate_peers in (False, True)
            ]
            exact_frames = sum(
                not row.changed_codebooks and row.changed_feedback_values == 0
                for row in ordered
            )
            report = {
                "sample_id": directory.name,
                "repeat": repeat,
                "fixed_batch_rows": arguments.fixed_batch_rows,
                "peer_inputs": "zero or duplicated selected query, not captured peers",
                "main_code_dtype": "int64; original sampler index dtype not captured",
                "capture_sha256": sha256(
                    (directory / "trajectory.pt").read_bytes()
                ).hexdigest(),
                "compared_frames_per_peer_fill": frame_count,
                "exact_codes_and_feedback": exact_frames,
                "all_recorded_outputs_reproduced": exact_frames == len(ordered),
                "reverse_traversal_reproduced": True if repeat == 1 else None,
                "frames": [asdict(row) for row in ordered],
            }
            (arguments.output / f"{directory.name}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(
                json.dumps(
                    {name: value for name, value in report.items() if name != "frames"}
                ),
                flush=True,
            )
    if arguments.check_buffer_reuse:
        assert all(
            not row.changed_codebooks and row.changed_feedback_values == 0
            for sample in comparisons.values()
            for row in sample.values()
        ), "Native capture replay must match before buffer reuse checks"
        check_buffer_reuse(model, arguments.capture, device, arguments.output)
    else:
        pass
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fixed-batch-rows", type=int)
    parser.add_argument("--check-buffer-reuse", action="store_true")
    main(parser.parse_args())
else:
    pass
