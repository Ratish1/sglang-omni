# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) model runner for OmniScheduler."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts_local.radix_hash import gpu_radix_row_hash
from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeJournal
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.scheduling.types import RequestOutput


class MossTTSLocalModelRunner(ModelRunner):
    """Drives the per-frame local-transformer decode and feedback embeddings.

    Per step: the backbone (radix-cached, CUDA-graphed) produces one hidden
    state per request; :meth:`_collect_frame` then runs the batched local
    micro-decode — a binary continue/stop decision and 12 sequentially
    sampled RVQ codes — and stages the next frame's summed embedding through
    ``model._decode_input_embedding`` so the next decode step stays
    CUDA-graph-replayable (decode input_ids are row indices).
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

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
        self._write_decode_input_embedding(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS Local prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            if data.output_rows:
                # KV-pressure retraction re-prefills with an extend region
                # spanning already-generated frames; their rows live in
                # output_rows, not prompt_rows. The resumed prefill samples
                # the next frame itself, superseding any feedback embedding
                # stranded by the retraction.
                generated = torch.stack(data.output_rows, dim=0)
                rows = torch.cat([rows.to(generated.device), generated], dim=0)
                self.model._state_pool.reset_for_refill(sched_req.request_id)
            current_rows = rows[prefix_len : prefix_len + req_len]
            if int(current_rows.shape[0]) != req_len:
                raise RuntimeError(
                    f"MOSS-TTS Local prefill row mismatch for {req.rid}: have "
                    f"{int(current_rows.shape[0])} rows, need {req_len} "
                    f"(prefix={prefix_len}, prompt={int(data.prompt_rows.shape[0])}, "
                    f"generated={len(data.output_rows)})"
                )
            embeds = self.model._prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.dtype,
        )

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        pool = self.model._state_pool
        weight = self.model._decode_input_embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-TTS Local decode input_ids must contain one row id per request"
            )
        if batch_size > pool.padding_row:
            raise RuntimeError(
                "MOSS-TTS Local decode batch exceeds the staged decode-embedding "
                f"rows ({batch_size} > {pool.padding_row})"
            )
        pool_rows = [pool.acquire_row(sched_req.request_id) for sched_req in requests]
        row_tensor = torch.tensor(pool_rows, dtype=torch.long, device=weight.device)
        with torch.no_grad():
            weight[:batch_size].copy_(pool.feedback_embeds[row_tensor])

        row_ids = torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def _collect_frame(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return
        recorder = self._active_event_recorder()
        with self._profile_scope(
            "moss_tts_local.collect_frame.setup",
            recorder=recorder,
            requests=requests,
        ):
            hidden_states = getattr(result.logits_output, "hidden_states", None)
            if not isinstance(hidden_states, torch.Tensor):
                raise RuntimeError(
                    "MOSS-TTS Local model output did not include hidden states"
                )
            hidden_ndim = int(hidden_states.ndim)
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[:, -1, :]

            cfg = self.model.config
            device = hidden_states.device
            pool = self.model._state_pool
            datas = [sched_req.data for sched_req in requests]
            batch_size = len(datas)
            num_channels = int(cfg.n_vq) + 1
            collect_metadata = self._collect_frame_profile_metadata(
                schedule_batch=schedule_batch,
                batch_size=batch_size,
                num_channels=num_channels,
                hidden_ndim=hidden_ndim,
                hidden_size=int(hidden_states.shape[-1]),
            )
            self._emit_profile_event(
                recorder,
                requests,
                "moss_tts_local_collect_frame_start",
                collect_metadata,
            )

        with self._profile_scope(
            "moss_tts_local.collect_frame.pool_rows",
            recorder=recorder,
            requests=requests,
            metadata=collect_metadata,
        ):
            pool_rows = []
            for sched_req in requests:
                rid = sched_req.request_id
                row = self.model.acquire_row(rid)
                pool_rows.append(row)
                pool.ensure_params(row, rid, sched_req.data)

        with self._profile_scope(
            "moss_tts_local.collect_frame.param_gather",
            recorder=recorder,
            requests=requests,
            metadata=collect_metadata,
        ):
            row_t = torch.tensor(
                pool_rows, dtype=torch.long, device=pool.feedback_embeds.device
            )
            params = {
                "text_temp": pool.text_temp[row_t],
                "text_top_p": pool.text_top_p[row_t],
                "text_top_k": pool.text_top_k[row_t],
                "audio_temp": pool.audio_temp[row_t],
                "audio_top_p": pool.audio_top_p[row_t],
                "audio_top_k": pool.audio_top_k[row_t],
                "seeds": pool.seeds[row_t],
            }
            text_temp = params["text_temp"]
            text_top_p = params["text_top_p"]
            text_top_k = params["text_top_k"]
            audio_temp = params["audio_temp"]
            audio_top_p = params["audio_top_p"]
            audio_top_k = params["audio_top_k"]
            sampling_seeds = params["seeds"]

        with self._profile_scope(
            "moss_tts_local.collect_frame.sampling_state",
            recorder=recorder,
            requests=requests,
            metadata=collect_metadata,
        ):
            gen_steps = torch.tensor(
                [int(d.generation_steps) for d in datas],
                dtype=torch.long,
                device=device,
            )
            rep_penalties = [float(d.audio_repetition_penalty) for d in datas]
            rep_histories = self._gather_rep_histories(datas, rep_penalties, device)

        def sample_text(logits: torch.Tensor) -> torch.Tensor:
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=text_temp,
                top_p=text_top_p,
                top_k=text_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels,
            )

        def sample_audio(logits: torch.Tensor, channel: int) -> torch.Tensor:
            if rep_histories is not None:
                self._apply_audio_repetition_penalty(
                    logits, rep_histories, rep_penalties, channel
                )
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels + channel + 1,
            )

        with self._profile_scope(
            "moss_tts_local.collect_frame.path_select",
            recorder=recorder,
            requests=requests,
            metadata=collect_metadata,
        ):
            frame_graph_max_bs = int(getattr(self.model, "frame_graph_max_bs", 0) or 0)
            fallback_reason = self._frame_decode_fallback_reason(
                batch_size=batch_size,
                frame_graph_max_bs=frame_graph_max_bs,
                has_repetition_penalty=rep_histories is not None,
            )
            use_graph = fallback_reason is None
            frame_metadata = dict(collect_metadata)
            frame_metadata.update(
                {
                    "used_frame_graph": use_graph,
                    "frame_decode_path": "cuda_graph" if use_graph else "eager",
                    "frame_graph_max_bs": frame_graph_max_bs,
                    "fallback_reason": fallback_reason,
                    "repetition_penalty_rows": sum(
                        1 for penalty in rep_penalties if penalty != 1.0
                    ),
                }
            )
            self._emit_profile_event(
                recorder,
                requests,
                "moss_tts_local_frame_decode_start",
                frame_metadata,
            )
        if use_graph:
            with self._profile_scope(
                "moss_tts_local.frame_decode.cuda_graph",
                recorder=recorder,
                requests=requests,
                metadata=frame_metadata,
            ):
                stop_choice, codes, feedback = self.model.decode_frame_graphed(
                    hidden_states,
                    text_temperature=text_temp,
                    text_top_p=text_top_p,
                    text_top_k=text_top_k,
                    audio_temperature=audio_temp,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    seeds=sampling_seeds,
                    base_positions=gen_steps * num_channels,
                )
            self._emit_profile_event(
                recorder,
                requests,
                "moss_tts_local_frame_decode_end",
                frame_metadata,
            )
            # The graph outputs are static buffers that the next replay (any
            # later prefill or decode step) overwrites; snapshot what we keep.
            with self._profile_scope(
                "moss_tts_local.collect_frame.graph_output_clone",
                recorder=recorder,
                requests=requests,
                metadata=frame_metadata,
            ):
                codes = codes.clone()
                embeds = feedback.clone()
        else:
            with self._profile_scope(
                "moss_tts_local.frame_decode.eager",
                recorder=recorder,
                requests=requests,
                metadata=frame_metadata,
            ):
                stop_choice, codes = self.model.decode_frame(
                    hidden_states,
                    sample_text=sample_text,
                    sample_audio=sample_audio,
                )
            self._emit_profile_event(
                recorder,
                requests,
                "moss_tts_local_frame_decode_end",
                frame_metadata,
            )
            embeds = None

        with self._profile_scope(
            "moss_tts_local.collect_frame.row_build",
            recorder=recorder,
            requests=requests,
            metadata=frame_metadata,
        ):
            slot_id = int(cfg.audio_assistant_slot_token_id)
            end_id = int(cfg.audio_end_token_id)
            next_text = torch.where(
                stop_choice == 0,
                torch.full((batch_size,), slot_id, dtype=torch.long, device=device),
                torch.full((batch_size,), end_id, dtype=torch.long, device=device),
            )

            rows = torch.empty(
                (batch_size, num_channels), dtype=torch.long, device=device
            )
            rows[:, 0] = next_text
            rows[:, 1:] = codes

        with self._profile_scope(
            "moss_tts_local.collect_frame.radix_hash",
            recorder=recorder,
            requests=requests,
            metadata=frame_metadata,
        ):
            next_token_ids = self._row_radix_token_ids(rows, next_text, end_id)
            result.next_token_ids = next_token_ids
            schedule_batch.output_ids = next_token_ids

        if embeds is None:
            with self._profile_scope(
                "moss_tts_local.collect_frame.eager_feedback_embed",
                recorder=recorder,
                requests=requests,
                metadata=frame_metadata,
            ):
                embeds = self.model._prepare_multi_modal_inputs(
                    rows.to(device=self.model.device)
                )

        with self._profile_scope(
            "moss_tts_local.collect_frame.emit_filter",
            recorder=recorder,
            requests=requests,
            metadata=frame_metadata,
        ):
            emit_indices = [
                i
                for i, sched_req in enumerate(requests)
                if not self._is_chunked_request(sched_req)
            ]
            collect_end_metadata = dict(frame_metadata)
            collect_end_metadata.update(
                {
                    "emitted_count": len(emit_indices),
                    "chunked_count": batch_size - len(emit_indices),
                    "has_journal": bool(emit_indices),
                }
            )
        if not emit_indices:
            self._emit_profile_event(
                recorder,
                requests,
                "moss_tts_local_collect_frame_end",
                collect_end_metadata,
            )
            return

        with self._profile_scope(
            "moss_tts_local.collect_frame.feedback_write",
            recorder=recorder,
            requests=requests,
            metadata=collect_end_metadata,
        ):
            emit_index_t = torch.tensor(
                emit_indices, dtype=torch.long, device=rows.device
            )
            emit_pool_rows = [pool_rows[i] for i in emit_indices]
            emit_row_t = row_t[emit_index_t.to(device=row_t.device)]
            emit_embeds = embeds.index_select(0, emit_index_t.to(device=embeds.device))
            pool.feedback_embeds[emit_row_t] = emit_embeds.detach().to(
                device=pool.feedback_embeds.device,
                dtype=pool.feedback_embeds.dtype,
            )

        with self._profile_scope(
            "moss_tts_local.collect_frame.journal",
            recorder=recorder,
            requests=[requests[i] for i in emit_indices],
            metadata=collect_end_metadata,
        ):
            result.moss_journal = MossTTSLocalDecodeJournal(
                rids=[requests[i].request_id for i in emit_indices],
                pool_rows=emit_pool_rows,
                rows=rows.index_select(0, emit_index_t),
            )
        self._emit_profile_event(
            recorder,
            requests,
            "moss_tts_local_collect_frame_end",
            collect_end_metadata,
        )

    @staticmethod
    @contextmanager
    def _profile_scope(
        name: str,
        *,
        recorder: Any | None = None,
        requests: list | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        emit_fine_events = (
            recorder is not None
            and bool(requests)
            and os.environ.get("SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS") == "1"
        )
        event_base = name.replace(".", "_")
        if emit_fine_events:
            event_metadata = dict(metadata or {})
            event_metadata.update(
                {
                    "scope": name,
                    "shared_batch_interval": True,
                    "fine_frame_scope": True,
                }
            )
            MossTTSLocalModelRunner._emit_profile_event(
                recorder,
                requests,
                f"{event_base}_start",
                event_metadata,
            )
        try:
            with torch.profiler.record_function(name):
                yield
        finally:
            if emit_fine_events:
                MossTTSLocalModelRunner._emit_profile_event(
                    recorder,
                    requests,
                    f"{event_base}_end",
                    event_metadata,
                )

    @staticmethod
    def _active_event_recorder() -> Any | None:
        recorder = _get_event_recorder()
        return recorder if recorder.is_active() else None

    @staticmethod
    def _emit_profile_event(
        recorder: Any | None,
        requests: list,
        event_name: str,
        metadata: dict[str, Any],
    ) -> None:
        if recorder is None:
            return
        for sched_req in requests:
            recorder.emit(
                request_id=sched_req.request_id,
                stage=None,
                event_name=event_name,
                metadata=metadata,
            )

    @staticmethod
    def _collect_frame_profile_metadata(
        *,
        schedule_batch: Any,
        batch_size: int,
        num_channels: int,
        hidden_ndim: int,
        hidden_size: int,
    ) -> dict[str, Any]:
        forward_mode = getattr(schedule_batch, "forward_mode", None)
        is_extend = (
            forward_mode is not None
            and hasattr(forward_mode, "is_extend")
            and bool(forward_mode.is_extend())
        )
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        return {
            "batch_size": int(batch_size),
            "num_channels": int(num_channels),
            "hidden_ndim": int(hidden_ndim),
            "hidden_size": int(hidden_size),
            "is_extend": is_extend,
            "is_decode": is_decode,
            "shared_batch_interval": True,
        }

    @staticmethod
    def _frame_decode_fallback_reason(
        *,
        batch_size: int,
        frame_graph_max_bs: int,
        has_repetition_penalty: bool,
    ) -> str | None:
        if has_repetition_penalty:
            return "repetition_penalty"
        if frame_graph_max_bs <= 0:
            return "frame_graph_unavailable"
        if batch_size > frame_graph_max_bs:
            return "batch_exceeds_frame_graph"
        return None

    @staticmethod
    def _row_radix_token_ids(
        rows: torch.Tensor,
        next_text: torch.Tensor,
        end_id: int,
    ) -> torch.Tensor:
        """Radix-cache token ids for generated frames.

        The scheduler appends one token id per frame to the request's KV
        chain, and the radix tree keys on those ids. The text channel alone is
        the same assistant-slot id for every continuing frame of every
        request, so a re-prefill after retraction could falsely prefix-match
        into another identical-prompt request's cached generated region. Hash
        the full multi-channel row — the same keying used for prompt rows —
        so a radix match implies identical audio content (a per-position id
        clash is ~1/151643 and only matters on top of an identical full
        prefix). The hash is folded below the special-token band because the
        scheduler finishes any request whose generated id crosses the vocab
        boundary (``Req._check_vocab_boundary_finish``); the stop decision
        keeps the raw audio_end id so eos detection still fires.

        Unlike the prompt path (``build_row_cache_key_ids``'s host-side
        blake2b), this runs every decode step on a device tensor, so it uses
        the capture-safe tensor-native polynomial hash in :mod:`radix_hash` —
        no GPU->CPU sync. See ``docs/design/gpu_radix_hash.md``.
        """
        return gpu_radix_row_hash(rows, next_text, end_id)

    @staticmethod
    def _gather_rep_histories(
        datas: list,
        rep_penalties: list[float],
        device: torch.device,
    ) -> list[torch.Tensor | None] | None:
        """Per-request generated-code history, only when a penalty is active.

        Upstream v1.5 applies the audio repetition penalty over each channel's
        previously *generated* frames only (the prompt's reference codes are
        excluded), so the history snapshot is taken from ``output_rows``.
        """
        if all(penalty == 1.0 for penalty in rep_penalties):
            return None
        histories: list[torch.Tensor | None] = []
        for data, penalty in zip(datas, rep_penalties):
            if penalty == 1.0 or not data.output_rows:
                histories.append(None)
                continue
            stacked = torch.stack(data.output_rows, dim=0)[:, 1:]
            histories.append(stacked.to(device=device, dtype=torch.long))
        return histories

    @staticmethod
    def _apply_audio_repetition_penalty(
        logits: torch.Tensor,
        histories: list[torch.Tensor | None],
        penalties: list[float],
        channel: int,
    ) -> None:
        """In-place penalty on fp32 logits, matching upstream order (before
        temperature scaling)."""
        vocab = logits.shape[-1]
        for row, (history, penalty) in enumerate(zip(histories, penalties)):
            if history is None or penalty == 1.0:
                continue
            tokens = torch.unique(history[:, channel])
            tokens = tokens[(tokens >= 0) & (tokens < vocab)]
            if tokens.numel() == 0:
                continue
            scores = logits[row, tokens]
            logits[row, tokens] = torch.where(
                scores < 0, scores * penalty, scores / penalty
            )

    @staticmethod
    def _is_chunked_request(sched_req: Any) -> bool:
        req = getattr(sched_req.data, "req", None)
        return req is not None and getattr(req, "is_chunked", 0) > 0

    def finalize_skip_rids(self, scheduler_output) -> set[str]:
        """Non-final chunked-prefill rows must not advance ``generation_steps``.

        Their micro-decode still runs (as today), but the spurious step would
        shift the final chunk's sampling position off the no-chunk path; the
        sampling is positional (``position = generation_steps * num_channels +
        channel``), so suppressing the advance keeps the chunked path
        bit-identical to the single-shot prefill path.
        """
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if self._is_chunked_request(sched_req)
        }

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        # The per-step journal is the single source of truth for output
        # collection. A missing journal means no frame was produced this step
        # (e.g. a prefill-only batch), which is the synchronous-baseline early
        # return.
        journal = getattr(result, "moss_journal", None)
        if journal is None:
            return

        end_id = int(self.model.config.audio_end_token_id)
        expected_reqs = [
            sched_req
            for sched_req in scheduler_output.requests
            if not self._is_chunked_request(sched_req)
        ]
        expected_rids = [sched_req.request_id for sched_req in expected_reqs]
        rows_len = int(journal.rows.shape[0])
        if len(journal.rids) != rows_len or len(journal.pool_rows) != rows_len:
            raise RuntimeError(
                "MOSS-TTS Local journal length mismatch: "
                f"rids={len(journal.rids)} pool_rows={len(journal.pool_rows)} "
                f"rows={rows_len}"
            )
        if journal.rids != expected_rids:
            raise RuntimeError(
                "MOSS-TTS Local journal/batch alignment broken: "
                f"{journal.rids} != {expected_rids}"
            )
        for i, sched_req in enumerate(expected_reqs):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == end_id:
                continue
            sched_req.data.output_rows.append(journal.rows[i])
