# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from queue import Queue

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import sampling_from_probs_torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import DynamicGradMode
from sglang.srt.utils.common import is_pin_memory_available

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
    accept_cosyvoice3_stream_token,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    TOTAL_VOCAB_SIZE,
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    TOKEN_HOP_LEN,
    first_ar_flush_tokens,
    prompt_token_len,
)
from sglang_omni.platforms import current_platform
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import ARRequestData, SchedulerRequest

COSYVOICE3_RAS_WINDOW_SIZE = 10
# note (Yucheng Hu): the redraw reusing the first draw's seeded noise would be
# conditioned on the rejected candidate having won that draw.
COSYVOICE3_RAS_SEED_SALT = 0x2545F491


@torch.compile(dynamic=True, disable=not current_platform.is_cuda())
def mask_repeated_speech_tokens(
    window: torch.Tensor,
    candidates: torch.Tensor,
    top_ks: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows whose sampled speech token is in their window, top-k above 1, and every
    row's float64 probabilities with its sampled token masked."""
    repeated = (
        (window == candidates.unsqueeze(1)).any(dim=1)
        & (candidates < VOCAB_SIZE)
        & (top_ks != 1)
    )
    # note (ratish): float64, as the first seeded draw feeds the compiled
    # multinomial_with_seed, so the redraw reuses that graph, not a new one.
    row_probs = probs.to(torch.float64)
    redraw_probs = row_probs.scatter(1, candidates.unsqueeze(1), 0.0)
    # note (Yucheng Hu): multinomial rejects an all-zero row, and a row whose
    # whole mass sat on the candidate has nothing else to draw.
    return repeated, torch.where(
        redraw_probs.sum(dim=1, keepdim=True) > 0, redraw_probs, row_probs
    )


class FunCosyVoice3ModelRunner(ModelRunner):
    """Runs Fun-CosyVoice3 AR steps and collects generated speech tokens."""

    tp_worker: ModelWorker
    model: FunCosyVoice3SGLangModel

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        token_hop_len: int = TOKEN_HOP_LEN,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        hop = int(token_hop_len)
        if hop <= 0:
            raise ValueError(f"token_hop_len must be positive, got {token_hop_len}")
        else:
            pass
        self.token_hop_len = hop
        self.ar_followup_flush_tokens = hop
        self.outbox: Queue[OutgoingMessage] | None = None
        self.vocoder_target = "vocoder"
        self.cosyvoice3_recent_tokens: dict[str, list[int]] = {}
        # note (ratish): each request slot's last speech tokens, kept on the device so a
        # step's repeat check needs no host read of the step before.
        self.recent_speech_tokens = torch.full(
            (
                tp_worker.model_runner.req_to_token_pool.req_to_token.shape[0],
                COSYVOICE3_RAS_WINDOW_SIZE,
            ),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.recent_speech_token_owners: dict[int, str] = {}
        if current_platform.is_cuda():
            # note (ratish): Dynamo specializes a one row batch; the entry built at two
            # rows serves every larger one, compiled here rather than on a request.
            with DynamicGradMode():
                for rows in (1, 2):
                    mask_repeated_speech_tokens(
                        torch.full(
                            (rows, COSYVOICE3_RAS_WINDOW_SIZE),
                            -1,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.zeros(rows, dtype=torch.long, device=self.device),
                        torch.ones(rows, dtype=torch.int32, device=self.device),
                        torch.ones(
                            (rows, TOTAL_VOCAB_SIZE),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    )
        else:
            pass

    def set_stream_outbox(self, outbox: Queue[OutgoingMessage]) -> None:
        self.outbox = outbox

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self.build_prefill_input_embeds(forward_batch, requests)
        return self.forward_with_input_embeds(forward_batch, input_embeds)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_tokens(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_tokens(result, forward_batch, schedule_batch, requests)

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """A decode step may launch before the one before it resolves: the repetition
        penalty takes that step's ids on the device, the repeat window lives on the
        device, and min_new_tokens counts one step late, as SGLang's overlap
        scheduler does."""
        return all(
            req.sampling_params.frequency_penalty == 0.0
            and req.sampling_params.presence_penalty == 0.0
            and req.custom_logit_processor is None
            for req in batch.reqs
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        scaling_penalties = (
            forward_batch.sampling_info.acc_scaling_penalties if is_lookahead else None
        )
        if scaling_penalties is None:
            return
        else:
            pass
        # note (ratish): the scheduler folded in tokens up to the step before the one
        # still in flight; that step's ids are this step's inputs, and the scatter
        # repeats harmlessly when the scheduler folds them in later.
        penalties = torch.tensor(
            [
                sched_req.data.req.sampling_params.repetition_penalty
                for sched_req in requests
            ],
            dtype=scaling_penalties.dtype,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)
        scaling_penalties.scatter_(
            1, forward_batch.input_ids.long().unsqueeze(1), penalties.unsqueeze(1)
        )

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        super().post_decode_resolve(
            launch_buf, result, forward_batch, schedule_batch, requests
        )
        if launch_buf is None:
            return
        else:
            pass
        # note (ratish): a lookahead step resolves after the step before it, which can
        # finish or retract its request.
        self.emit_speech_tokens(
            [
                (sched_req, token_id)
                for sched_req, token_id in zip(
                    requests, result.next_token_ids.tolist(), strict=True
                )
                if not (
                    sched_req.data.req.finished()
                    or self.req_is_retracted(sched_req.data.req)
                )
            ]
        )

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        """Sample the first speech token before collecting prefill output."""
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        """Sample each speech token before collecting decode output."""
        del forward_batch, schedule_batch, requests
        return True

    def apply_repetition_penalty(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        """Leave repetition-penalty ownership to SGLang's forward snapshot.

        SGLangExecutionBridge copies SamplingBatchInfo with the
        accumulated scaling penalties before this runner samples. Applying the
        host-side incremental helper as well would penalize each token twice.
        """
        del logits_output, requests

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        if not self.is_mps_ras_fallback(logits_output.next_token_logits):
            next_token_ids = super().sample_next_token_ids(
                logits_output,
                forward_batch,
                schedule_batch,
                requests,
            )
            if forward_batch.forward_mode.is_decode():
                return self.redraw_repeated_tokens(
                    logits_output, forward_batch, requests, next_token_ids
                )
            else:
                # note (ratish): rows sampled outside a decode step skip the window,
                # so their next decode step rebuilds it from the host output ids.
                for sched_req in requests:
                    self.recent_speech_token_owners.pop(
                        sched_req.data.req.kv.req_pool_idx, None
                    )
                return next_token_ids
        else:
            pass
        if len(requests) != 1:
            raise RuntimeError(
                "Fun-CosyVoice3 Torch MPS currently requires max_running_requests=1"
            )
        else:
            pass

        self.apply_repetition_penalty(logits_output, requests)
        self.apply_codec_suppress_tokens(logits_output, requests)
        self.install_sampling_seeds(forward_batch, requests)
        sampling_info = forward_batch.sampling_info
        installed_seeds = sampling_info.sampling_seed
        rng_context: AbstractContextManager[None] = nullcontext()
        if installed_seeds is not None:
            # Note (yexiaodong): MPS cannot represent the sampler's float64
            # probabilities, so preserve filtering while sampling from a
            # stable per-request/per-step RNG seed.
            sampling_params = requests[0].data.req.sampling_params
            row_seed = (
                int(sampling_params.sampling_seed)
                if sampling_params.sampling_seed is not None
                else 42
            )
            req = requests[0].data.req
            absolute_position = len(req.origin_input_ids) + len(req.output_ids) - 1
            step_seed = (row_seed + absolute_position * 0x9E3779B1) & SAMPLING_SEED_MASK
            device_index = logits_output.next_token_logits.device.index or 0
            rng_context = torch.random.fork_rng(
                devices=[device_index],
                device_type="mps",
            )
            sampling_info.sampling_seed = None
        else:
            pass

        wants_rollout_logprob = any(sr.data.return_logprob for sr in requests)
        if wants_rollout_logprob:
            self.enable_sampler_logprobs(forward_batch, len(requests))
        else:
            pass
        try:
            with rng_context:
                if installed_seeds is not None:
                    torch.manual_seed(step_seed)
                else:
                    pass
                # Note (yexiaodong): Scope eager mode to sampling because
                # SGLang compiles the repetition helper independently.
                with torch.compiler.set_stance("force_eager"):
                    next_token_ids = self.tp_worker.model_runner.sample(
                        logits_output,
                        forward_batch,
                    )
                next_token_ids = self.apply_ras_fallback(
                    logits_output,
                    next_token_ids,
                    sampling_info,
                    requests,
                )
        finally:
            sampling_info.sampling_seed = installed_seeds
        if wants_rollout_logprob:
            next_token_logprobs = logits_output.next_token_logprobs
            if next_token_logprobs is None:
                raise RuntimeError(
                    "Sampler did not populate next_token_logprobs when "
                    "return_logprob is enabled"
                )
            else:
                pass
            self.record_rollout_logprobs(
                next_token_logprobs,
                next_token_ids,
                requests,
            )
        else:
            pass
        return next_token_ids

    def is_mps_ras_fallback(self, logits: torch.Tensor) -> bool:
        """Torch MPS without float64 samples one request and redraws it itself."""
        return (
            logits.device.type == "mps" and not current_platform.is_float64_supported()
        )

    def apply_ras_fallback(
        self,
        logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        """Apply CosyVoice3's repetition-aware redraw on Torch/MPS.

        The upstream CosyVoice sampler draws from top-k/top-p first and, when
        that candidate appears in the recent ten speech tokens, redraws once
        from the full distribution with the candidate masked. SGLang's MPS
        sampler has already applied all request constraints and materialized
        probabilities by this point, so the second draw can stay on MPS
        without invoking the float64 seeded sampler.
        """
        request = requests[0]
        request_id = str(request.request_id)
        if (
            self.cosyvoice3_recent_tokens
            and request_id not in self.cosyvoice3_recent_tokens
        ):
            # Note (yexiaodong): MPS sampling is single-request; clear state
            # here because aborts can bypass the normal finish hook.
            self.cosyvoice3_recent_tokens.clear()
        else:
            pass
        recent = self.cosyvoice3_recent_tokens.setdefault(request_id, [])
        token_ids = next_token_ids.reshape(-1)
        token_id = int(token_ids[0].item())

        if not sampling_info.is_all_greedy and token_id < VOCAB_SIZE:
            if token_id in recent[-COSYVOICE3_RAS_WINDOW_SIZE:]:
                probs = logits_output.next_token_logits
                if probs is None or probs.ndim != 2:
                    raise RuntimeError(
                        "CosyVoice3 Torch/MPS RAS requires sampled probabilities"
                    )
                else:
                    pass
                fallback_probs = probs[0].to(dtype=torch.float32).clone()
                fallback_probs[token_id] = 0.0
                fallback_probs.clamp_(min=0.0)
                if float(fallback_probs.sum().item()) <= 0.0:
                    # Note (yexiaodong): A collapsed top-k/top-p row has no
                    # valid redraw distribution, so keep the sampled token.
                    fallback = None
                else:
                    fallback = torch.multinomial(
                        fallback_probs.unsqueeze(0), num_samples=1
                    ).reshape(-1)
                if fallback is not None:
                    next_token_ids = next_token_ids.clone()
                    next_token_ids[0] = fallback.to(dtype=next_token_ids.dtype)
                    token_id = int(fallback[0].item())
                else:
                    pass
                if (
                    fallback is not None
                    and logits_output.next_token_logprobs is not None
                ):
                    # Note (yexiaodong): Update rollout logprobs after redraw
                    # so they describe the emitted token, not the rejected one.
                    fallback_logprobs = torch.log(
                        probs.clamp_min(torch.finfo(probs.dtype).tiny)
                    )
                    logits_output.next_token_logprobs = fallback_logprobs.gather(
                        1, next_token_ids.long().view(-1, 1)
                    ).view(-1)
                else:
                    pass
            else:
                pass
        else:
            pass

        if 0 <= token_id < VOCAB_SIZE:
            recent.append(token_id)
            del recent[:-COSYVOICE3_RAS_WINDOW_SIZE]
        else:
            pass
        return next_token_ids

    def on_request_finished(
        self, request_id: str, req_data: ARRequestData | None
    ) -> None:
        if req_data is not None:
            self.flush_code_chunks(request_id, req_data, force=True)
        else:
            pass
        recent_tokens = getattr(self, "cosyvoice3_recent_tokens", None)
        if recent_tokens is not None:
            recent_tokens.pop(str(request_id), None)
        else:
            pass

    def collect_tokens(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        if result.next_token_ids is None:
            return
        else:
            pass
        token_ids = result.next_token_ids
        if token_ids.ndim != 1:
            token_ids = token_ids.reshape(-1)
        else:
            pass
        # note (guozhihao-224): one batched D2H instead of per-request .item() syncs.
        # note (Yucheng Hu): a pageable D2H on the default stream stalls other threads'
        # kernel launches while it waits, an event wait does not.
        self.stage_token_ids(result, token_ids)
        self.emit_speech_tokens(
            list(zip(requests, self.resolve_host_token_ids(result).tolist()))
        )

    def emit_speech_tokens(
        self, request_tokens: list[tuple[SchedulerRequest, int]]
    ) -> None:
        for sched_req, token_id in request_tokens:
            if token_id >= VOCAB_SIZE:
                continue
            else:
                pass
            token = torch.tensor([token_id], dtype=torch.long)
            sched_req.data.output_codes.append(token)
            self.queue_or_emit_code_chunk(sched_req, token)

    def redraw_repeated_tokens(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        sampled_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Redraw a speech token its row emitted in the last ten with it masked, so EOS
        stays reachable."""
        stale = [
            sched_req
            for sched_req in requests
            if self.recent_speech_token_owners.get(sched_req.data.req.kv.req_pool_idx)
            != sched_req.request_id
        ]
        if stale:
            # note (ratish): a request's first decode step runs after its prefill
            # step resolved, so its host output ids are complete here.
            pin_memory = is_pin_memory_available(self.device)
            stale_slots = torch.tensor(
                [sched_req.data.req.kv.req_pool_idx for sched_req in stale],
                pin_memory=pin_memory,
            )
            histories = torch.tensor(
                [
                    (
                        [-1] * COSYVOICE3_RAS_WINDOW_SIZE
                        + list(sched_req.data.req.output_ids)
                    )[-COSYVOICE3_RAS_WINDOW_SIZE:]
                    for sched_req in stale
                ],
                dtype=torch.long,
                pin_memory=pin_memory,
            )
            self.recent_speech_tokens[
                stale_slots.to(self.device, non_blocking=True)
            ] = histories.to(self.device, non_blocking=True)
            for sched_req in stale:
                self.recent_speech_token_owners[sched_req.data.req.kv.req_pool_idx] = (
                    sched_req.request_id
                )
        else:
            pass
        slots = forward_batch.req_pool_indices
        window = self.recent_speech_tokens[slots]
        candidates = sampled_ids.reshape(-1).long()
        sampling_info = forward_batch.sampling_info
        if sampling_info.is_all_greedy:
            token_ids = candidates
        else:
            # note (ratish): both samplers apply top-k and top-p to a copy,
            # leaving the tempered probabilities here, or the tempered logits on Ascend.
            if self.tp_worker.model_runner.sampler.use_ascend_backend:
                probs = torch.softmax(
                    logits_output.next_token_logits, dim=-1, dtype=torch.float64
                )
            else:
                probs = logits_output.next_token_logits
            repeated, redraw_probs = mask_repeated_speech_tokens(
                window, candidates, sampling_info.top_ks, probs
            )
            sampling_seeds = sampling_info.sampling_seed
            redraw_seeds = (
                None
                if sampling_seeds is None
                else sampling_seeds ^ COSYVOICE3_RAS_SEED_SALT
            )
            redrawn_ids = sampling_from_probs_torch(
                redraw_probs, redraw_seeds, forward_batch.positions
            )
            token_ids = torch.where(repeated, redrawn_ids.long(), candidates)
        self.recent_speech_tokens[slots] = torch.cat(
            (window[:, 1:], token_ids.unsqueeze(1)), dim=1
        )
        return token_ids.to(sampled_ids.dtype).view_as(sampled_ids)

    def queue_or_emit_code_chunk(
        self,
        sched_req: SchedulerRequest,
        token: torch.Tensor,
    ) -> None:
        data: CosyVoice3SGLangRequestData = sched_req.data
        if self.outbox is None or data.stream_metadata is None:
            return
        else:
            pass
        if not accept_cosyvoice3_stream_token(data, token):
            return
        else:
            pass
        data.stream_code_buffer.append(token)
        data.stream_code_seen += 1
        if int(data.stream_code_next_flush) <= 0:
            # note (guozhihao-224): first flush is hop+lookahead. Prompt hop
            # alignment is applied on Flow prompt tensors, not extra AR tokens.
            data.stream_code_next_flush = first_ar_flush_tokens(
                prompt_token_len(data.flow_prompt_speech_token),
                hop_len=self.token_hop_len,
            )
        else:
            pass
        if data.stream_code_seen >= data.stream_code_next_flush:
            self.flush_code_chunks(sched_req.request_id, data, force=False)
        else:
            pass

    def flush_code_chunks(
        self,
        request_id: str,
        data: CosyVoice3SGLangRequestData,
        *,
        force: bool,
    ) -> None:
        pending = data.stream_code_buffer
        if not pending:
            return
        else:
            pass
        payload = pending[0] if len(pending) == 1 else torch.cat(pending, dim=0)
        pending.clear()
        if not force:
            data.stream_code_next_flush = (
                int(data.stream_code_seen) + self.ar_followup_flush_tokens
            )
        else:
            pass
        self.emit_code_chunk(request_id, data, payload)

    def emit_code_chunk(
        self,
        request_id: str,
        data: CosyVoice3SGLangRequestData,
        codes: torch.Tensor,
    ) -> None:
        if self.outbox is None:
            return
        else:
            pass
        metadata = data.stream_metadata
        if metadata is None:
            return
        else:
            pass
        chunk_metadata = dict(metadata)
        # note (guozhihao-224): first chunk carries Flow prompt tensors so
        # vocoder can start before the AR payload arrives.
        if not data.stream_prompt_sent:
            chunk_metadata["flow_prompt_speech_token"] = data.flow_prompt_speech_token
            chunk_metadata["flow_prompt_speech_feat"] = data.flow_prompt_speech_feat
            chunk_metadata["flow_embedding"] = data.flow_embedding
            data.stream_prompt_sent = True
        else:
            pass
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=self.vocoder_target,
                data=codes,
                metadata=chunk_metadata,
            )
        )

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data: CosyVoice3SGLangRequestData = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prompt_input_embeds
            if prompt_embeds is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 prefill requires prompt_input_embeds"
                )
            else:
                pass
            pieces.append(prompt_embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def forward_with_input_embeds(
        self,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        else:
            pass
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
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
