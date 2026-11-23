# SPDX-License-Identifier: Apache-2.0
"""Check native predictor replay against captured codes and recurrent feedback."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.utils import find_available_port


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


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    device = torch.device("cuda", arguments.gpu)
    torch.cuda.set_device(device)
    config = ConfigManager.from_file(
        str(arguments.config), model_path_override=str(arguments.checkpoint)
    ).config
    assert config is not None
    factory_arguments = resolve_stage_factory_args(
        config.stage_named("talker_ar"), config, gpu_id=arguments.gpu
    )
    factory_arguments["nccl_port"] = find_available_port()
    scheduler = create_talker_ar_executor_from_config(**factory_arguments)
    runner = scheduler.model_runner
    assert isinstance(runner, QwenTalkerModelRunner)
    model = runner.model
    (arguments.output / "runtime.json").write_text(
        json.dumps(
            {
                "factory_arguments": factory_arguments,
                "predictor_graph_batch_rows": model.predictor_decode_graph_batch_sizes,
                "fused_predictor_layers": model.predictor_fused_layers is not None,
                "exact_add_norm": model.predictor_exact_add_norm,
                "triton_attention": model.predictor_triton_attention,
                "rope_stores_kv": model.predictor_rope_stores_kv,
                "peer_inputs": "zero or duplicated selected query, not captured peers",
                "main_code_dtype": "int64; original sampler index dtype not captured",
            },
            indent=2,
        )
    )
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
                if state["phase"] == "prefill":
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
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    main(parser.parse_args())
else:
    pass
