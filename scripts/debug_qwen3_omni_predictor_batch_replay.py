"""Replay actual padded predictor batches through fused and plain arithmetic."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata
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


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    checkpoint_directory = snapshot_download(
        arguments.model_path,
        revision=arguments.model_revision,
        local_files_only=True,
    )
    scheduler = create_talker_ar_executor_from_config(
        checkpoint_directory,
        gpu_id=arguments.gpu,
        max_running_requests=arguments.max_running_requests,
        total_gpu_memory_fraction=arguments.memory_fraction,
        code2wav_in_process=True,
    )
    native = scheduler.model_runner.model
    fused = native.predictor_fused_layers
    assert fused is not None and native.corpus_recorder is None
    capture = PredictorHeadCapture()
    handles = [
        head.register_forward_hook(capture.record)
        for head in native.code_predictor.lm_head
    ]
    requested_ids = set(arguments.request_id)
    covered_steps = {request_id: [] for request_id in requested_ids}
    total_batches = 0
    exact_captured_batches = 0
    repeat_exact_batches = 0
    changed_selected_frames = {request_id: [] for request_id in requested_ids}
    with (arguments.output_directory / "batches.jsonl").open("w") as reports:
        for line in (
            (arguments.capture_directory / "steps.jsonl").read_text().splitlines()
        ):
            metadata = StepMetadata.model_validate_json(line)
            if metadata.predictor_batch == 0:
                continue
            else:
                pass
            saved = torch.load(
                arguments.capture_directory / f"step-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            hidden_states = saved["predictor_hidden"].cuda()
            main_codes = (
                saved["layer0_codes"]
                .cuda()
                .to(torch.int32 if bool(saved["code_int32"]) else torch.int64)
            )
            assert (
                hidden_states.shape[0]
                == main_codes.shape[0]
                == metadata.predictor_batch
            )
            assert fused.covers(metadata.predictor_batch * 2)
            captured_logits = torch.stack(
                [
                    saved["snapshots"][f"predictor.{group}.output"]
                    for group in range(native.config.num_code_groups - 1)
                ],
                dim=1,
            )
            captured_codes = saved["codes"]
            runs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for use_fused in (True, False, True):
                native.predictor_fused_layers = fused if use_fused else None
                capture.logits.clear()
                codes, feedback = native.code_predictor_forward_incremental_eager(
                    main_codes, hidden_states
                )
                codes = codes.squeeze(-1).clone().cpu()
                feedback = feedback[:, 0].clone().cpu()
                logits = torch.stack(
                    [
                        values.reshape(metadata.predictor_batch, -1)
                        for values in capture.logits
                    ],
                    dim=1,
                ).cpu()
                assert len(capture.logits) == native.config.num_code_groups - 1
                assert logits.isfinite().all() and feedback.isfinite().all()
                assert torch.equal(logits.argmax(-1), codes[:, 1:])
                runs.append((codes, feedback, logits))
            native.predictor_fused_layers = fused
            fused_codes, fused_feedback, fused_logits = runs[0]
            plain_codes, plain_feedback, plain_logits = runs[1]
            captured_exact = (
                torch.equal(fused_logits, captured_logits)
                and torch.equal(fused_codes[: metadata.live_rows], captured_codes)
                and torch.equal(fused_feedback[: metadata.live_rows], saved["feedback"])
            )
            repeated_exact = all(
                torch.equal(first, repeated)
                for first, repeated in zip(runs[0], runs[2], strict=True)
            )
            exact_captured_batches += captured_exact
            repeat_exact_batches += repeated_exact
            total_batches += 1
            selected = []
            for row, request_id in enumerate(metadata.request_ids):
                if request_id in requested_ids:
                    covered_steps[request_id].append(metadata.step)
                    changed_groups = (
                        (fused_codes[row] != plain_codes[row])
                        .nonzero()
                        .flatten()
                        .tolist()
                    )
                    if changed_groups:
                        changed_selected_frames[request_id].append(
                            metadata.frame_indices[row]
                        )
                    else:
                        pass
                    selected.append(
                        {
                            "request_id": request_id,
                            "row": row,
                            "frame_index": metadata.frame_indices[row],
                            "changed_code_groups": changed_groups,
                            "fused_codes": fused_codes[row].tolist(),
                            "plain_codes": plain_codes[row].tolist(),
                            "feedback_maximum_absolute": float(
                                (
                                    fused_feedback[row].float()
                                    - plain_feedback[row].float()
                                )
                                .abs()
                                .max()
                            ),
                        }
                    )
                else:
                    pass
            reports.write(
                json.dumps(
                    {
                        "step": metadata.step,
                        "phase": metadata.phase,
                        "live_rows": metadata.live_rows,
                        "predictor_batch": metadata.predictor_batch,
                        "captured_logits_codes_feedback_exact": captured_exact,
                        "fused_repeat_exact_after_plain": repeated_exact,
                        "changed_real_code_coordinates": int(
                            (
                                fused_codes[: metadata.live_rows]
                                != plain_codes[: metadata.live_rows]
                            ).sum()
                        ),
                        "changed_padding_code_coordinates": int(
                            (
                                fused_codes[metadata.live_rows :]
                                != plain_codes[metadata.live_rows :]
                            ).sum()
                        ),
                        "logits_maximum_absolute": float(
                            (fused_logits.float() - plain_logits.float()).abs().max()
                        ),
                        "selected": selected,
                    }
                )
                + "\n"
            )
            reports.flush()
    for request_id in requested_ids:
        trace = json.loads(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_text()
        )
        assert covered_steps[request_id] == trace["steps"]
    summary = {
        "model_revision": arguments.model_revision,
        "batches": total_batches,
        "exact_captured_batches": exact_captured_batches,
        "fused_repeat_exact_batches": repeat_exact_batches,
        "selected_frames": {
            request_id: len(steps) for request_id, steps in covered_steps.items()
        },
        "changed_selected_frames": changed_selected_frames,
        "scope": "Original full padded predictor inputs, integer dtype, batch geometry and checkpoint weights. Fused/plain/fused controls isolate arithmetic from Thinker scheduling. Failure to reproduce captured logits, codes and feedback invalidates attribution to the original live frame.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    for handle in handles:
        handle.remove()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
