"""Audit native Thinker cache reads against the actual prior slot writers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(kw_only=True)
class WrittenCacheRow:
    request_id: str
    prefix_sha256: bytes
    keys_and_values: torch.Tensor


@dataclass(kw_only=True)
class RequestCacheRows:
    request_id: str
    prefix_tokens: int
    slot_ids: torch.Tensor
    prefix_sha256: list[bytes]


@dataclass(kw_only=True)
class ThinkerCacheHistory:
    directory: Path
    pool: MHATokenToKVPool
    written_rows: dict[int, WrittenCacheRow] = field(default_factory=dict)
    forwards: int = 0
    written_tokens: int = 0
    checked_tokens: int = 0

    def read(self, slot_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack(
                    (
                        self.pool.get_key_buffer(layer_index)[slot_ids],
                        self.pool.get_value_buffer(layer_index)[slot_ids],
                    ),
                    dim=1,
                )
                for layer_index in self.pool.model_layer_ids
            ],
            dim=1,
        ).cpu()

    def prepare(
        self, forward_batch: ForwardBatch, schedule_batch: ScheduleBatch
    ) -> list[RequestCacheRows]:
        assert not self.pool.use_hnd
        assert self.pool.kv_cache_layout == "nhd"
        sequence_lengths = schedule_batch.seq_lens_cpu.tolist()
        prefix_lengths = (
            list(forward_batch.extend_prefix_lens_cpu)
            if forward_batch.forward_mode.is_extend()
            else [length - 1 for length in sequence_lengths]
        )
        request_rows = []
        input_offset = 0
        for request, sequence_tokens, prefix_tokens in zip(
            schedule_batch.reqs, sequence_lengths, prefix_lengths, strict=True
        ):
            token_ids = (list(request.origin_input_ids) + list(request.output_ids))[
                :sequence_tokens
            ]
            assert len(token_ids) == sequence_tokens
            new_tokens = sequence_tokens - prefix_tokens
            assert (
                token_ids[prefix_tokens:]
                == forward_batch.input_ids[
                    input_offset : input_offset + new_tokens
                ].tolist()
            )
            slot_ids = forward_batch.req_to_token_pool.req_to_token[
                request.req_pool_idx, :sequence_tokens
            ].long()
            assert torch.equal(
                slot_ids[prefix_tokens:],
                forward_batch.out_cache_loc[
                    input_offset : input_offset + new_tokens
                ].long(),
            )
            prefix_hash = hashlib.sha256()
            prefix_digests = []
            for token_id in token_ids:
                prefix_hash.update(token_id.to_bytes(8, "little", signed=True))
                prefix_digests.append(prefix_hash.digest())
            request_rows.append(
                RequestCacheRows(
                    request_id=request.rid,
                    prefix_tokens=prefix_tokens,
                    slot_ids=slot_ids,
                    prefix_sha256=prefix_digests,
                )
            )
            input_offset += new_tokens
        assert input_offset == forward_batch.input_ids.numel()
        return request_rows

    def verify(
        self, request_rows: list[RequestCacheRows], phase: Literal["before", "after"]
    ) -> None:
        for request in request_rows:
            if request.prefix_tokens == 0:
                continue
            else:
                pass
            slot_ids = request.slot_ids[: request.prefix_tokens]
            previous = [self.written_rows[slot_id] for slot_id in slot_ids.tolist()]
            expected = torch.stack([row.keys_and_values for row in previous])
            actual = self.read(slot_ids)
            keys_and_values_exact = (
                (actual == expected).flatten(start_dim=3).all(dim=-1).all(dim=0)
            )
            identities_exact = all(
                row.prefix_sha256 == digest
                for row, digest in zip(
                    previous,
                    request.prefix_sha256[: request.prefix_tokens],
                    strict=True,
                )
            )
            report = {
                "forward": self.forwards,
                "request_id": request.request_id,
                "phase": phase,
                "prefix_tokens": request.prefix_tokens,
                "previous_writer_ids": sorted({row.request_id for row in previous}),
                "token_prefixes_exact": identities_exact,
                "layer_keys_and_values_exact": keys_and_values_exact.tolist(),
            }
            with (self.directory / "checks.jsonl").open("a") as output:
                output.write(json.dumps(report) + "\n")
            self.checked_tokens += request.prefix_tokens
            if not (identities_exact and bool(keys_and_values_exact.all())):
                torch.save(
                    {
                        "slot_ids": slot_ids.cpu(),
                        "expected": expected,
                        "actual": actual,
                        "report": report,
                    },
                    self.directory / f"failure-{request.request_id}-{phase}.pt",
                )
                raise RuntimeError(f"Thinker cache history mismatch: {report}")
            else:
                pass

    def record(self, request_rows: list[RequestCacheRows]) -> None:
        slot_ids = torch.cat(
            [request.slot_ids[request.prefix_tokens :] for request in request_rows]
        )
        actual = self.read(slot_ids)
        row_index = 0
        for request in request_rows:
            for slot_id, prefix_digest in zip(
                request.slot_ids[request.prefix_tokens :].tolist(),
                request.prefix_sha256[request.prefix_tokens :],
                strict=True,
            ):
                self.written_rows[slot_id] = WrittenCacheRow(
                    request_id=request.request_id,
                    prefix_sha256=prefix_digest,
                    keys_and_values=actual[row_index],
                )
                row_index += 1
        assert row_index == len(actual)
        self.written_tokens += row_index
        self.forwards += 1
        (self.directory / "summary.json").write_text(
            json.dumps(
                {
                    "forwards": self.forwards,
                    "written_tokens": self.written_tokens,
                    "retained_slots": len(self.written_rows),
                    "checked_tokens": self.checked_tokens,
                    "layers": len(self.pool.model_layer_ids),
                    "pool_class": type(self.pool).__name__,
                    "pool_layout": self.pool.kv_cache_layout,
                },
                indent=2,
            )
        )
