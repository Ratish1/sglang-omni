# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_tts.repetition_ownership import repetition_penalty_owner
from sglang_omni.scheduling.types import RequestOutput


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._has_pending_code_step = False
        self._row_ids_cache: torch.Tensor | None = None
        self._repetition_mask: torch.Tensor | None = None
        self._repetition_penalty_column: torch.Tensor | None = None
        self._repetition_mask_last_sampled: torch.Tensor | None = None
        self._repetition_mask_prep_rids: list | None = None
        self._repetition_mask_active = False
        self._qwen_repetition_enabled = repetition_penalty_owner() in {
            "qwen",
            "double",
        }

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        self.model.prepare_decode_buffers(requests)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(
            forward_batch,
            input_embeds,
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._install_semantic_sampling_seeds(forward_batch, requests)
        next_token_ids = super()._sample_next_token_ids(
            logits_output,
            forward_batch,
            schedule_batch,
            requests,
        )
        if getattr(self, "_qwen_repetition_enabled", False):
            self._repetition_mask_last_sampled = (
                next_token_ids
                if self._repetition_mask_active
                and isinstance(next_token_ids, torch.Tensor)
                else None
            )
        return next_token_ids

    # ------------------------------------------------------------------
    # Qwen3-TTS logit shaping
    # ------------------------------------------------------------------

    def lookahead_eligible(self, batch: Any) -> bool:
        """Keep every history-owned diagnostic mode on the synchronous path."""

        for req in batch.reqs:
            data = getattr(req, "_omni_data", None)
            if data is None:
                return False
            if float(getattr(data, "qwen_repetition_penalty", 1.0)) != 1.0:
                return False
        return super().lookahead_eligible(batch)

    @staticmethod
    def _repetition_mask_fingerprint(requests: list) -> list | None:
        rids = []
        for sched_req in requests:
            rid = getattr(sched_req, "request_id", None)
            epoch = getattr(sched_req.data, "_qwen3_tts_prep_epoch", None)
            if rid is None or epoch is None:
                return None
            rids.append((rid, epoch))
        return rids

    @staticmethod
    def _every_penalized_row_grew_by_one(requests: list) -> bool:
        for sched_req in requests:
            data = sched_req.data
            if float(data.qwen_repetition_penalty) == 1.0:
                continue
            seen_len = getattr(data, "_rep_seen_len", None)
            output_ids = data.req.output_ids
            if seen_len is None or not output_ids:
                return False
            if len(output_ids) != seen_len + 1:
                return False
        return True

    def _ensure_repetition_mask(
        self,
        batch_size: int,
        vocab: int,
        device: Any,
    ) -> None:
        mask = getattr(self, "_repetition_mask", None)
        if (
            mask is not None
            and mask.shape[0] >= batch_size
            and mask.shape[1] == vocab
            and mask.device == device
        ):
            return
        rows = max(batch_size, 64)
        self._repetition_mask = torch.zeros(
            rows,
            vocab,
            dtype=torch.bool,
            device=device,
        )
        self._repetition_penalty_column = torch.ones(
            rows,
            1,
            dtype=torch.float32,
            device=device,
        )
        self._repetition_mask_prep_rids = None

    def _rebuild_repetition_mask(
        self,
        requests: list,
        vocab: int,
        device: Any,
    ) -> None:
        repetition_mask = self._repetition_mask
        penalty_column = self._repetition_penalty_column
        if repetition_mask is None or penalty_column is None:
            raise RuntimeError("Qwen3-TTS repetition mask buffers are not initialized")

        batch_size = len(requests)
        repetition_mask[:batch_size] = False
        repetition_rows: list[int] = []
        repetition_tokens: list[int] = []
        penalties = [1.0] * batch_size
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            penalty = float(data.qwen_repetition_penalty)
            penalties[row_idx] = penalty
            output_ids = data.req.output_ids
            if penalty != 1.0 and output_ids:
                seen = ModelRunner._rep_penalty_unique_tokens(
                    data,
                    output_ids,
                    vocab,
                )
                repetition_rows.extend([row_idx] * len(seen))
                repetition_tokens.extend(seen)

        if repetition_rows:
            pairs = torch.tensor(
                repetition_rows + repetition_tokens,
                dtype=torch.long,
                device=device,
            )
            pair_count = len(repetition_rows)
            repetition_mask[pairs[:pair_count], pairs[pair_count:]] = True
        penalty_column[:batch_size, 0] = torch.tensor(
            penalties,
            dtype=torch.float32,
            device=device,
        )
        self._repetition_mask_active = any(penalty != 1.0 for penalty in penalties)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        """Apply only the Qwen-owned diagnostic share of the public penalty."""

        if not getattr(self, "_qwen_repetition_enabled", False):
            return
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return
        if not any(
            float(sched_req.data.qwen_repetition_penalty) != 1.0
            for sched_req in requests
        ):
            self._repetition_mask_active = False
            return

        batch_size = len(requests)
        vocab = logits.shape[1]
        self._ensure_repetition_mask(batch_size, vocab, logits.device)
        repetition_mask = self._repetition_mask
        penalty_column = self._repetition_penalty_column
        if repetition_mask is None or penalty_column is None:
            raise RuntimeError("Qwen3-TTS repetition mask buffers are not initialized")

        fingerprint = self._repetition_mask_fingerprint(requests)
        last_sampled = getattr(self, "_repetition_mask_last_sampled", None)
        incremental_update = (
            fingerprint is not None
            and fingerprint == getattr(self, "_repetition_mask_prep_rids", None)
            and last_sampled is not None
            and last_sampled.shape[0] >= batch_size
            and self._every_penalized_row_grew_by_one(requests)
        )
        if incremental_update:
            rows = torch.arange(batch_size, device=logits.device)
            repetition_mask[
                rows,
                last_sampled[:batch_size].clamp(0, vocab - 1),
            ] = True
            for sched_req in requests:
                data = sched_req.data
                output_ids = data.req.output_ids
                if output_ids:
                    ModelRunner._rep_penalty_unique_tokens(data, output_ids, vocab)
        else:
            self._rebuild_repetition_mask(requests, vocab, logits.device)
        self._repetition_mask_prep_rids = fingerprint

        if self._repetition_mask_active:
            penalties = penalty_column[:batch_size]
            scores = logits.to(torch.float32)
            penalized = torch.where(
                scores > 0,
                scores / penalties,
                scores * penalties,
            )
            logits.copy_(
                torch.where(
                    repetition_mask[:batch_size],
                    penalized,
                    scores,
                ).to(logits.dtype)
            )

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return

        # Qwen3-TTS reserves the final 1024 configured token IDs for codec
        # control and suppresses that range except for codec EOS.
        configured_vocab = int(self.model.config.vocab_size)
        suppress_start = max(0, configured_vocab - 1024)
        suppress_stop = min(configured_vocab, logits.shape[1])
        if suppress_start >= suppress_stop:
            return

        active_logits = logits[: len(requests)]
        codec_eos = int(self.model.config.codec_eos_token_id)
        if suppress_start <= codec_eos < suppress_stop:
            active_logits[:, suppress_start:codec_eos] = float("-inf")
            active_logits[:, codec_eos + 1 : suppress_stop] = float("-inf")
        else:
            active_logits[:, suppress_start:suppress_stop] = float("-inf")

    def _install_semantic_sampling_seeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        forward_batch.sampling_info.sampling_seed = (
            self.model._semantic_sampling_seed_tensor[:batch_size]
        )

    def _collect_codes(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._has_pending_code_step = False
        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        semantic_positions = self._sample_positions(forward_batch, layer0_codes.device)
        self.model.code_predictor_forward(
            layer0_codes,
            hidden,
            semantic_positions=semantic_positions,
        )
        # Note: (Jiaxin Deng) stage the ids into pinned host memory now so the
        # output processor's .tolist() waits on an event instead of issuing a
        # blocking pageable copy inside the decode loop.
        self._stage_token_ids(result, result.next_token_ids)
        self._has_pending_code_step = True

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        if not self._has_pending_code_step:
            return
        self._has_pending_code_step = False
        eos_id = int(self.model.config.codec_eos_token_id)
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        batch_size = len(scheduler_output.requests)
        codes_snap = self.model._output_codes[:batch_size].detach().clone()
        embeds_snap = self.model._output_embeds[:batch_size].detach().clone()
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            code_chunk = codes_snap[row_idx]
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.latest_stream_code_chunk = code_chunk
            sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])

    def _sample_positions(
        self, forward_batch: Any, device: torch.device
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        if is_decode:
            positions = getattr(forward_batch, "positions", None)
            if positions is not None:
                return positions.to(device=device, dtype=torch.long)

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            return (seq_lens.to(device=device, dtype=torch.long) - 1).clamp_min(0)

        positions = getattr(forward_batch, "positions", None)
        if positions is not None:
            return positions.to(device=device, dtype=torch.long)

        raise RuntimeError("Qwen3-TTS subtalker sampling requires semantic positions")

    def _write_feedback_buffers(self, forward_batch: Any, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        decode_feedback_embedding = self.model._decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        row_ids = self._decode_row_ids(batch_size, input_ids)
        rows = []

        for row_idx, sched_req in enumerate(requests):
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=decode_feedback_embedding.weight.device,
                dtype=decode_feedback_embedding.weight.dtype,
            )
            if combined is None:
                token_id = input_ids[row_idx : row_idx + 1].to(
                    device=decode_feedback_embedding.weight.device
                )
                combined = self.model.get_input_embeddings()(token_id).reshape(-1)
            QwenTalkerModelRunner._append_decode_input_history(sched_req.data, combined)
            rows.append(combined)
        with torch.no_grad():
            torch.stack(rows, dim=0, out=decode_feedback_embedding.weight[:batch_size])
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)

    def _decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = getattr(self, "_row_ids_cache", None)
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self._row_ids_cache = cached
        return cached[:batch_size]

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            piece = QwenTalkerModelRunner._projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=req_len,
                device=forward_batch.input_ids.device,
            )
            if piece is None or int(piece.shape[0]) != req_len:
                have = 0 if piece is None else int(piece.shape[0])
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {have} rows, need {req_len}"
                )
            pieces.append(piece)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
