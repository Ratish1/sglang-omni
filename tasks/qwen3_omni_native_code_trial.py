# SPDX-License-Identifier: Apache-2.0
"""Follow one residual-code intervention from a verified native speech prefix."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.utils.predictor_layers import codebook_step


class NativeCodeTrial:
    def __init__(
        self,
        reference: Path,
        sequence_tokens: int,
        residual_group: int,
        expected_code: int,
        replacement_code: int,
        destination: Path,
    ) -> None:
        self.reference: list[dict[str, torch.Tensor]] = torch.load(
            reference / "trajectory.pt", map_location="cpu", weights_only=True
        )
        self.request_id: str = json.loads((reference / "trajectory.json").read_text())[
            "request_id"
        ]
        self.sequence_tokens = sequence_tokens
        self.residual_group = residual_group
        self.expected_code = expected_code
        self.replacement_code = replacement_code
        self.destination = destination
        self.applied = False

    @torch.inference_mode()
    def apply(
        self,
        worker: ModelWorker,
        forward_batch: ForwardBatch,
        result: GenerationBatchResult,
        selected_row: int,
        frames: list[dict[str, torch.Tensor]],
    ) -> None:
        assert not self.applied
        assert forward_batch.spec_info is None
        runner = worker.model_runner
        model = runner.model
        pool = runner.token_to_kv_pool
        assert isinstance(pool, MHATokenToKVPool)
        batch_rows = forward_batch.batch_size
        assert result.can_run_cuda_graph
        assert runner.decode_cuda_graph_runner.bs == batch_rows
        assert model.predictor_fused_layers is not None
        assert model.predictor_fused_layers.covers(2 * batch_rows)
        assert 0 < self.residual_group < model.config.num_code_groups
        assert self.expected_code != self.replacement_code
        fields = (
            "prefill_embeddings",
            "input_ids",
            "positions",
            "rotary_positions",
            "predictor_query",
            "logits",
            "codec_ids",
            "feedback",
            "queued_feedback",
            "text_embedding",
            "consumed_input",
            "stored_keys",
            "stored_values",
        )
        assert len(frames) <= len(self.reference)
        for frame_index, (actual, expected) in enumerate(zip(frames, self.reference)):
            for name in fields:
                assert (name in actual) == (name in expected), (frame_index, name)
                if name in expected:
                    assert torch.equal(actual[name].cpu(), expected[name]), (
                        frame_index,
                        name,
                    )
                else:
                    pass
        current = frames[-1]
        assert int(current["positions"].flatten()[0]) + 1 == self.sequence_tokens
        original_codes = model.output_codes[:batch_rows].clone()
        original_feedback = model.output_embeds[:batch_rows].clone()
        main_codes = model.sampled_token_ids[:batch_rows].clone()
        predictor_query = result.logits_output.hidden_states[:batch_rows].clone()
        assert (
            int(original_codes[selected_row, self.residual_group]) == self.expected_code
        )
        slots = (
            torch.cat(
                [
                    runner.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices[row], : int(length)
                    ]
                    for row, length in enumerate(forward_batch.seq_lens_cpu)
                ]
            )
            .long()
            .unique()
        )
        buffers = (*pool.k_buffer, *pool.v_buffer)
        stored_cache = [buffer.index_select(0, slots) for buffer in buffers]
        model.code_predictor_forward_incremental_eager(main_codes, predictor_query)
        assert torch.equal(model.output_codes[:batch_rows], original_codes)
        assert torch.equal(model.output_embeds[:batch_rows], original_feedback)
        group = 0
        decision: dict[str, torch.Tensor] = {}

        def force_code(
            logits: torch.Tensor,
            embedding_weight: torch.Tensor,
            codes: torch.Tensor,
            summed: torch.Tensor,
        ) -> torch.Tensor:
            nonlocal group
            group += 1
            if group == self.residual_group:
                assert torch.isfinite(logits).all()
                before_sum = summed[selected_row].clone()
                assert int(logits[selected_row].argmax()) == self.expected_code
                assert (
                    logits[selected_row, self.expected_code]
                    == logits[selected_row, self.replacement_code]
                )
                next_embedding = codebook_step(logits, embedding_weight, codes, summed)
                assert int(codes[selected_row]) == self.expected_code
                replacement_embedding = embedding_weight[self.replacement_code]
                codes[selected_row] = self.replacement_code
                next_embedding[selected_row, 0].copy_(replacement_embedding)
                summed[selected_row].copy_(
                    (before_sum.float() + replacement_embedding.float()).to(
                        summed.dtype
                    )
                )
                decision.update(
                    {
                        "logits": logits[selected_row].clone(),
                        "feedback_before_group": before_sum,
                        "replacement_embedding": replacement_embedding.clone(),
                    }
                )
                return next_embedding
            else:
                return codebook_step(logits, embedding_weight, codes, summed)

        with patch(
            "sglang_omni.models.qwen3_omni.components.talker.codebook_step", force_code
        ):
            model.code_predictor_forward_incremental_eager(main_codes, predictor_query)
        assert group == model.config.num_code_groups - 1
        forced_codes = model.output_codes[:batch_rows].clone()
        forced_feedback = model.output_embeds[:batch_rows].clone()
        peers = torch.arange(batch_rows, device=main_codes.device) != selected_row
        assert torch.equal(forced_codes[peers], original_codes[peers])
        assert torch.equal(forced_feedback[peers], original_feedback[peers])
        assert torch.equal(
            forced_codes[selected_row, : self.residual_group],
            original_codes[selected_row, : self.residual_group],
        )
        assert (
            int(forced_codes[selected_row, self.residual_group])
            == self.replacement_code
        )
        assert torch.equal(main_codes, model.sampled_token_ids[:batch_rows])
        assert torch.equal(
            predictor_query, result.logits_output.hidden_states[:batch_rows]
        )
        assert torch.isfinite(forced_feedback).all()
        for buffer, expected in zip(buffers, stored_cache, strict=True):
            assert torch.equal(buffer.index_select(0, slots), expected)
        torch.save(
            {
                **{name: tensor.cpu() for name, tensor in decision.items()},
                "original_codes": original_codes[selected_row].cpu(),
                "forced_codes": forced_codes[selected_row].cpu(),
                "original_feedback": original_feedback[selected_row].cpu(),
                "forced_feedback": forced_feedback[selected_row].cpu(),
                "predictor_query": predictor_query[selected_row].cpu(),
            },
            self.destination.with_suffix(".pt"),
        )
        with self.destination.open("x") as receipt:
            json.dump(
                {
                    "request_id": self.request_id,
                    "sequence_tokens": self.sequence_tokens,
                    "frame_index": len(frames) - 1,
                    "residual_group": self.residual_group,
                    "expected_code": self.expected_code,
                    "replacement_code": self.replacement_code,
                    "batch_rows": batch_rows,
                    "prefix_exact": True,
                    "eager_control_exact": True,
                    "peers_unchanged": True,
                    "backbone_cache_unchanged": True,
                    "original_codes": original_codes[selected_row].tolist(),
                    "forced_codes": forced_codes[selected_row].tolist(),
                },
                receipt,
                indent=2,
            )
        self.applied = True
