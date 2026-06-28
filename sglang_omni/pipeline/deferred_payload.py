# SPDX-License-Identifier: Apache-2.0
"""Deferred payload materialization for scheduler outbox data."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DeferredPayload(Protocol):
    def materialize_payload(self) -> Any: ...


def materialize_payload(data: Any) -> Any:
    if isinstance(data, DeferredPayload):
        return data.materialize_payload()
    return data


def is_deferred_payload(data: Any) -> bool:
    return isinstance(data, DeferredPayload)
