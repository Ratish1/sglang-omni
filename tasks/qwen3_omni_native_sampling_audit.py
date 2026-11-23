# SPDX-License-Identifier: Apache-2.0
"""Check live Talker sampling rows against independently recorded emitted codes."""

import json
from pathlib import Path

import torch

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


class NativeSamplingAudit:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.emitted_codes: dict[str, list[int]] = {}
        self.request_counts: dict[str, dict[str, int]] = {}
        self.previous_rows: dict[str, int] = {}

    def check(self, model: Qwen3OmniTalker, requests: list[SchedulerRequest]) -> None:
        batch_rows = len(requests)
        repetition_masks = model.repetition_mask[:batch_rows].cpu()
        suppression_masks = model.suppress_mask[:batch_rows].cpu()
        floating_parameters = torch.cat(
            (
                model.repetition_penalties[:batch_rows],
                model.sampling_temperatures[:batch_rows],
                model.sampling_top_ps[:batch_rows, None],
                model.sampling_min_ps[:batch_rows, None],
            ),
            dim=1,
        ).cpu()
        top_counts = model.sampling_top_ks[:batch_rows].cpu().tolist()
        seeds = model.sampling_seeds[:batch_rows].cpu().tolist()
        for row_index, request in enumerate(requests):
            request_data = request.data
            assert isinstance(request_data, SGLangARRequestData)
            sampling = request_data.req.sampling_params
            history = self.emitted_codes[request.request_id]
            assert list(request_data.req.output_ids) == history, request.request_id
            expected_parameters = torch.tensor(
                [
                    sampling.repetition_penalty,
                    sampling.temperature,
                    sampling.top_p,
                    sampling.min_p,
                ],
                dtype=torch.float32,
                device="cpu",
            )
            expected_repeated = (
                {code for code in history if 0 <= code < repetition_masks.shape[1]}
                if sampling.repetition_penalty != 1.0
                else set()
            )
            suppressed = (
                request_data.suppress_tokens
                or request_data.req._codec_suppress_tokens  # noqa: leading-underscore  # existing request field
            )
            expected_suppressed = {
                code
                for code in (suppressed or [])
                if 0 <= code < suppression_masks.shape[1]
            }
            expected_seed = sampling.sampling_seed
            if expected_seed is None:
                expected_seed = derive_sampling_seed(
                    "sglang-omni-unseeded-row", request_data.req.rid
                )
            else:
                pass
            actual_repeated = set(
                repetition_masks[row_index].nonzero().flatten().tolist()
            )
            actual_suppressed = set(
                suppression_masks[row_index].nonzero().flatten().tolist()
            )
            checks = {
                "repetition_history": actual_repeated == expected_repeated,
                "suppression": actual_suppressed == expected_suppressed,
                "floating_parameters": torch.equal(
                    floating_parameters[row_index], expected_parameters
                ),
                "top_k": top_counts[row_index] == sampling.top_k,
                "seed": seeds[row_index] == expected_seed,
            }
            if not all(checks.values()):
                failure = {
                    "request_id": request.request_id,
                    "row_index": row_index,
                    "emitted_codes": history,
                    "checks": checks,
                    "actual_repeated": sorted(actual_repeated),
                    "expected_repeated": sorted(expected_repeated),
                    "actual_suppressed": sorted(actual_suppressed),
                    "expected_suppressed": sorted(expected_suppressed),
                    "actual_parameters": floating_parameters[row_index].tolist(),
                    "expected_parameters": expected_parameters.tolist(),
                    "actual_seed": seeds[row_index],
                    "expected_seed": expected_seed,
                }
                (self.directory / "sampling-failure.json").write_text(
                    json.dumps(failure, indent=2) + "\n"
                )
                raise AssertionError(f"Sampling row mismatch: {request.request_id}")
            else:
                pass
            counts = self.request_counts[request.request_id]
            counts["checked_decode_frames"] += 1
            previous_row = self.previous_rows.get(request.request_id)
            if previous_row is None:
                counts["batch_entries"] += 1
            elif previous_row != row_index:
                counts["row_moves"] += 1
            else:
                counts["same_row_steps"] += 1
        self.previous_rows = {
            request.request_id: row_index for row_index, request in enumerate(requests)
        }

    def record_samples(
        self, sampled_codes: torch.Tensor, requests: list[SchedulerRequest]
    ) -> None:
        codes = sampled_codes.detach().reshape(-1).cpu().tolist()
        assert len(codes) == len(requests)
        for request, code in zip(requests, codes, strict=True):
            request_data = request.data
            assert isinstance(request_data, SGLangARRequestData)
            history = self.emitted_codes.setdefault(request.request_id, [])
            assert list(request_data.req.output_ids) == history, request.request_id
            history.append(code)
            self.request_counts.setdefault(
                request.request_id,
                {
                    "checked_decode_frames": 0,
                    "batch_entries": 0,
                    "row_moves": 0,
                    "same_row_steps": 0,
                },
            )

    def finish(self, request_id: str, request: SGLangARRequestData) -> None:
        history = self.emitted_codes.pop(request_id)
        assert list(request.req.output_ids) == history, request_id
        counts = self.request_counts.pop(request_id)
        with (self.directory / "sampling-audit.jsonl").open("a") as report:
            report.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "emitted_frames": len(history),
                        "finish_reason": request.finish_reason,
                        **counts,
                    }
                )
                + "\n"
            )
