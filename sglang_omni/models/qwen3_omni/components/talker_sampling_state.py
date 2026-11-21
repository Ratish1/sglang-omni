"""The talker's sampling parameters and output tokens kept per request slot."""

from __future__ import annotations

import dataclasses

import torch
from sglang.srt.utils.common import async_h2d

from sglang_omni.sampling.seed import (
    SAMPLING_SEED_MASK,
    derive_sampling_seed,
    resolve_row_seed,
)
from sglang_omni.scheduling.types import SchedulerRequest


@dataclasses.dataclass(frozen=True, kw_only=True)
class TalkerSamplingRows:
    """The talker decode sample's inputs, one row per request slot or per batch row."""

    output_token_mask: torch.Tensor
    suppress_mask: torch.Tensor
    repetition_penalties: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    seeds: torch.Tensor


@dataclasses.dataclass(frozen=True, kw_only=True)
class TalkerSamplingStatePool:
    """TalkerSamplingRows by request slot; slot 0 is the row every padded graph row reads and writes."""

    slot_rows: TalkerSamplingRows
    # note (ratish): an indexed write of a Python bool copies it to the device and waits for the stream; a device value is a kernel argument.
    mask_set_value: torch.Tensor
    suppress_token_cache: dict[tuple[int, ...], torch.Tensor | None] = (
        dataclasses.field(default_factory=dict)
    )

    @classmethod
    def allocate(
        cls, *, slot_count: int, vocab_size: int, device: torch.device
    ) -> TalkerSamplingStatePool:
        return cls(
            slot_rows=TalkerSamplingRows(
                output_token_mask=torch.zeros(
                    slot_count, vocab_size, dtype=torch.bool, device=device
                ),
                suppress_mask=torch.zeros(
                    slot_count, vocab_size, dtype=torch.bool, device=device
                ),
                repetition_penalties=torch.ones(
                    slot_count, 1, dtype=torch.float32, device=device
                ),
                temperatures=torch.ones(
                    slot_count, 1, dtype=torch.float32, device=device
                ),
                top_ps=torch.ones(slot_count, dtype=torch.float32, device=device),
                top_ks=torch.ones(slot_count, dtype=torch.int32, device=device),
                min_ps=torch.zeros(slot_count, dtype=torch.float32, device=device),
                seeds=torch.zeros(slot_count, dtype=torch.int64, device=device),
            ),
            mask_set_value=torch.ones((), dtype=torch.bool, device=device),
        )

    def write_rows(
        self,
        request_slots: torch.Tensor,
        requests: list[SchedulerRequest],
        first_token_ids: torch.Tensor,
    ) -> None:
        """Reset each prefilled request's slot to its sampling parameters, suppress set and output tokens.

        Args:
            request_slots: [requests], the slot of each request.
            first_token_ids: [requests], the token each request's prefill sampled.
        """
        slot_rows = self.slot_rows
        vocab_size = slot_rows.output_token_mask.shape[1]
        device = request_slots.device
        repetition_penalties: list[float] = []
        temperatures: list[float] = []
        top_ps: list[float] = []
        min_ps: list[float] = []
        top_ks: list[int] = []
        seeds: list[int] = []
        output_rows: list[int] = []
        output_token_ids: list[int] = []
        suppress_groups: dict[int, tuple[torch.Tensor, list[int]]] = {}
        for row_index, scheduler_request in enumerate(requests):
            request_data = scheduler_request.data
            request = request_data.req
            sampling_params = request.sampling_params
            repetition_penalties.append(float(sampling_params.repetition_penalty))
            temperatures.append(float(sampling_params.temperature))
            top_ps.append(float(sampling_params.top_p))
            min_ps.append(float(sampling_params.min_p))
            top_ks.append(int(sampling_params.top_k))
            # Unseeded: rank-shared seed from the request id (same on every TP
            # rank), not os.urandom, else the ranks desync.
            seed = sampling_params.sampling_seed
            if seed is None:
                seed = derive_sampling_seed("sglang-omni-unseeded-row", request.rid)
            elif not (0 <= seed <= SAMPLING_SEED_MASK):
                seed = resolve_row_seed(seed)
                sampling_params.sampling_seed = seed
            else:
                pass
            seeds.append(seed)

            row_output_token_ids = {
                token_id
                for token_id in (int(token) for token in request.output_ids)
                if 0 <= token_id < vocab_size
            }
            output_rows.extend([row_index] * len(row_output_token_ids))
            output_token_ids.extend(row_output_token_ids)

            suppress_tokens = (
                request_data.suppress_tokens or request._codec_suppress_tokens
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            if suppress_tokens:
                suppress_content = tuple(int(token_id) for token_id in suppress_tokens)
                if suppress_content not in self.suppress_token_cache:
                    valid_token_ids = [
                        token_id
                        for token_id in suppress_content
                        if 0 <= token_id < vocab_size
                    ]
                    self.suppress_token_cache[suppress_content] = (
                        async_h2d(valid_token_ids, dtype=torch.int64, device=device)
                        if valid_token_ids
                        else None
                    )
                else:
                    pass
                suppress_token_ids = self.suppress_token_cache[suppress_content]
            else:
                suppress_token_ids = None
            if suppress_token_ids is None:
                pass
            elif id(suppress_token_ids) in suppress_groups:
                suppress_groups[id(suppress_token_ids)][1].append(row_index)
            else:
                suppress_groups[id(suppress_token_ids)] = (
                    suppress_token_ids,
                    [row_index],
                )

        slot_rows.output_token_mask.index_fill_(0, request_slots, False)
        slot_rows.suppress_mask.index_fill_(0, request_slots, False)
        float_parameters = async_h2d(
            [repetition_penalties, temperatures, top_ps, min_ps],
            dtype=torch.float32,
            device=device,
        )
        slot_rows.repetition_penalties.index_copy_(
            0, request_slots, float_parameters[0].unsqueeze(1)
        )
        slot_rows.temperatures.index_copy_(
            0, request_slots, float_parameters[1].unsqueeze(1)
        )
        slot_rows.top_ps.index_copy_(0, request_slots, float_parameters[2])
        slot_rows.min_ps.index_copy_(0, request_slots, float_parameters[3])
        slot_rows.top_ks.index_copy_(
            0, request_slots, async_h2d(top_ks, dtype=torch.int32, device=device)
        )
        slot_rows.seeds.index_copy_(
            0, request_slots, async_h2d(seeds, dtype=torch.int64, device=device)
        )
        if output_rows:
            output_pairs = async_h2d(
                output_rows + output_token_ids, dtype=torch.int64, device=device
            )
            slot_rows.output_token_mask.index_put_(
                (
                    request_slots[output_pairs[: len(output_rows)]],
                    output_pairs[len(output_rows) :],
                ),
                self.mask_set_value,
            )
        else:
            pass
        self.mark_output_tokens(request_slots, first_token_ids.to(torch.int64))
        for suppress_token_ids, suppress_rows in suppress_groups.values():
            if len(suppress_rows) == len(requests):
                group_slots = request_slots
            else:
                group_slots = request_slots[
                    async_h2d(suppress_rows, dtype=torch.int64, device=device)
                ]
            slot_rows.suppress_mask.index_put_(
                (group_slots[:, None], suppress_token_ids[None, :]),
                self.mask_set_value,
            )

    def gather(self, request_slots: torch.Tensor) -> TalkerSamplingRows:
        """The rows of these slots, in their order."""
        slot_rows = self.slot_rows
        return TalkerSamplingRows(
            output_token_mask=slot_rows.output_token_mask.index_select(
                0, request_slots
            ),
            suppress_mask=slot_rows.suppress_mask.index_select(0, request_slots),
            repetition_penalties=slot_rows.repetition_penalties.index_select(
                0, request_slots
            ),
            temperatures=slot_rows.temperatures.index_select(0, request_slots),
            top_ps=slot_rows.top_ps.index_select(0, request_slots),
            top_ks=slot_rows.top_ks.index_select(0, request_slots),
            min_ps=slot_rows.min_ps.index_select(0, request_slots),
            seeds=slot_rows.seeds.index_select(0, request_slots),
        )

    def mark_output_tokens(
        self, request_slots: torch.Tensor, token_ids: torch.Tensor
    ) -> None:
        """Add each slot's sampled token to its output tokens."""
        self.slot_rows.output_token_mask.index_put_(
            (request_slots, token_ids), self.mask_set_value
        )
