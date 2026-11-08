# SPDX-License-Identifier: Apache-2.0
"""Request-owned, graph-visible sampling for residual speech codes."""

from __future__ import annotations

import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from sglang_omni.sampling.sampler_compile import (
    SamplerCompileForms,
    compile_sampler_functions,
)
from sglang_omni.sampling.seed import derive_sampling_seed, resolve_row_seed
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest

RESIDUAL_TOP_K = 50
RESIDUAL_TOP_P = 0.8


class ResidualCodeSampler:
    """Keep each request's residual draws independent of batching and replay."""

    def __init__(
        self,
        max_batch_size: int,
        num_code_groups: int,
        device: torch.device,
        vocab_size: int,
    ) -> None:
        self.residual_group_count = num_code_groups - 1
        self.sampling_state = torch.zeros(
            3, max_batch_size, dtype=torch.int64, device=device
        )
        self.request_seeds, self.frame_indices, self.greedy_rows = (
            self.sampling_state.unbind(0)
        )
        self.greedy_rows.fill_(1)
        self.top_k = TopKLogitsWarper(RESIDUAL_TOP_K)
        self.top_p = TopPLogitsWarper(RESIDUAL_TOP_P)
        compile_sampler_functions(
            SamplerCompileForms(
                device=device,
                vocab_size=vocab_size,
                seeded_log_probability_dtype=torch.float64,
            )
        )

    def prepare(self, requests: list[SchedulerRequest]) -> None:
        request_seeds: list[int] = []
        frame_indices: list[int] = []
        greedy_rows: list[int] = []
        for request in requests:
            assert isinstance(request.data, SGLangARRequestData)
            backend_request = request.data.req
            assert backend_request is not None
            sampling_parameters = backend_request.sampling_params
            if sampling_parameters.sampling_seed is None:
                seed_parent = request.request_id
            else:
                seed_parent = resolve_row_seed(sampling_parameters.sampling_seed)
            request_seeds.append(
                derive_sampling_seed("qwen3-omni-residual", seed_parent)
            )
            frame_indices.append(len(backend_request.output_ids))
            greedy_rows.append(int(sampling_parameters.top_k == 1))
        self.sampling_state[:, : len(requests)].copy_(
            torch.tensor(
                [request_seeds, frame_indices, greedy_rows],
                dtype=torch.int64,
                device="cpu",
            )
        )

    def sample(
        self,
        logits: torch.Tensor,
        residual_prefix: torch.Tensor,
        group_index: int,
        frame_offset: int,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        filtered_logits = self.top_p(
            residual_prefix, self.top_k(residual_prefix, logits.float())
        )
        draw_positions = (
            self.frame_indices[:batch_size] + frame_offset
        ) * self.residual_group_count + group_index
        sampled_codes = multinomial_with_seed(
            filtered_logits.double(),
            self.request_seeds[:batch_size],
            draw_positions,
        ).squeeze(1)
        return torch.where(
            self.greedy_rows[:batch_size] != 0, logits.argmax(-1), sampled_codes
        )
