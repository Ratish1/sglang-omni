"""Verify captured decode caches against their own original projected history."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass(kw_only=True)
class CacheHistory:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


@dataclass(kw_only=True)
class CacheLayerCheck:
    request_id: str
    frame_index: int
    layer_index: int
    tokens: int
    previous_keys_exact: bool
    previous_values_exact: bool
    new_key_exact: bool
    new_value_exact: bool


@dataclass(kw_only=True)
class CacheHistoryRecorder:
    directory: Path
    histories: dict[str, CacheHistory] = field(default_factory=dict)
    saved_failures: set[tuple[str, int]] = field(default_factory=set)

    def record_prefix(
        self, request_id: str, keys: list[torch.Tensor], values: list[torch.Tensor]
    ) -> None:
        self.histories[request_id] = CacheHistory(keys=keys, values=values)

    def verify_decode(
        self,
        request_id: str,
        frame_index: int,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        projected_keys: list[torch.Tensor],
        projected_values: list[torch.Tensor],
    ) -> None:
        history = self.histories[request_id]
        checks: list[CacheLayerCheck] = []
        for layer_index, (
            cached_keys,
            cached_values,
            projected_key,
            projected_value,
        ) in enumerate(
            zip(keys, values, projected_keys, projected_values, strict=True)
        ):
            expected_keys = history.keys[layer_index]
            expected_values = history.values[layer_index]
            assert len(cached_keys) == len(cached_values) == len(expected_keys) + 1
            projected_key = projected_key.reshape_as(cached_keys[-1])
            projected_value = projected_value.reshape_as(cached_values[-1])
            check = CacheLayerCheck(
                request_id=request_id,
                frame_index=frame_index,
                layer_index=layer_index,
                tokens=len(cached_keys),
                previous_keys_exact=torch.equal(cached_keys[:-1], expected_keys),
                previous_values_exact=torch.equal(cached_values[:-1], expected_values),
                new_key_exact=torch.equal(cached_keys[-1], projected_key),
                new_value_exact=torch.equal(cached_values[-1], projected_value),
            )
            checks.append(check)
            failed = not (
                check.previous_keys_exact
                and check.previous_values_exact
                and check.new_key_exact
                and check.new_value_exact
            )
            if failed and (request_id, layer_index) not in self.saved_failures:
                torch.save(
                    {
                        "expected_keys": expected_keys,
                        "expected_values": expected_values,
                        "cached_keys": cached_keys,
                        "cached_values": cached_values,
                        "projected_key": projected_key,
                        "projected_value": projected_value,
                    },
                    self.directory / f"cache-failure-{request_id}-{layer_index}.pt",
                )
                self.saved_failures.add((request_id, layer_index))
            else:
                pass
            history.keys[layer_index] = torch.cat([expected_keys, projected_key[None]])
            history.values[layer_index] = torch.cat(
                [expected_values, projected_value[None]]
            )
        with (self.directory / "cache-history.jsonl").open("a") as output:
            for check in checks:
                output.write(json.dumps(asdict(check)) + "\n")

    def finish(self, request_id: str) -> None:
        self.histories.pop(request_id)
