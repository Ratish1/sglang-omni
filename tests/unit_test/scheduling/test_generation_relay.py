# SPDX-License-Identifier: Apache-2.0
"""Relay publication/consumption discipline (plan Phase 1, I4 single-stream
form): every read is preceded by exactly one publication; release/abort
invalidates and poisons; nothing reads a poisoned row silently."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.scheduling.async_generation import (
    InvariantViolation,
    OmniGenerationRelay,
)


def _handles(relay, rows):
    return [relay.handle(r) for r in rows]


def test_publish_materialize_roundtrip_consumes():
    relay = OmniGenerationRelay(8)
    hs = _handles(relay, [3, 0, 5])
    relay.publish(hs, torch.tensor([30, 10, 50]))
    assert relay.published_rows() == frozenset({0, 3, 5})
    out = relay.materialize(hs)
    assert out.tolist() == [30, 10, 50]  # gather follows handle order
    assert relay.published_rows() == frozenset()


def test_materialize_without_publication_raises():
    relay = OmniGenerationRelay(4)
    with pytest.raises(InvariantViolation, match="without publication"):
        relay.materialize(_handles(relay, [1]))


def test_double_consume_raises():
    relay = OmniGenerationRelay(4)
    hs = _handles(relay, [1])
    relay.publish(hs, torch.tensor([7]))
    relay.materialize(hs)
    with pytest.raises(InvariantViolation, match="without publication"):
        relay.materialize(hs)


def test_republish_over_unconsumed_value_raises():
    relay = OmniGenerationRelay(4)
    hs = _handles(relay, [2])
    relay.publish(hs, torch.tensor([9]))
    with pytest.raises(InvariantViolation, match="republished"):
        relay.publish(hs, torch.tensor([11]))


def test_steady_state_republish_after_consume_is_normal():
    relay = OmniGenerationRelay(4)
    hs = _handles(relay, [2])
    for step, token in enumerate((9, 11, 13)):
        relay.publish(hs, torch.tensor([token]))
        assert relay.materialize(hs).tolist() == [token], f"step {step}"


def test_invalidate_poisons_and_clears():
    relay = OmniGenerationRelay(4)
    hs = _handles(relay, [1, 2])
    relay.publish(hs, torch.tensor([7, 8]))
    relay.invalidate(hs)
    assert relay.published_rows() == frozenset()
    assert relay.token_ids[1].item() == OmniGenerationRelay.POISON
    with pytest.raises(InvariantViolation, match="without publication"):
        relay.materialize(hs)
    # release -> re-admission on the same pool row starts clean
    relay.publish(hs, torch.tensor([70, 80]))
    assert relay.materialize(hs).tolist() == [70, 80]


def test_partial_batch_reads_leave_other_rows_published():
    relay = OmniGenerationRelay(8)
    all_hs = _handles(relay, [0, 1, 2])
    relay.publish(all_hs, torch.tensor([10, 11, 12]))
    survivors = _handles(relay, [0, 2])  # row 1 finished; filtered out
    assert relay.materialize(survivors).tolist() == [10, 12]
    assert relay.published_rows() == frozenset({1})
    relay.invalidate(_handles(relay, [1]))  # its release invalidates
    assert relay.published_rows() == frozenset()


def test_shape_and_range_checks():
    relay = OmniGenerationRelay(2)
    with pytest.raises(InvariantViolation, match="out of range"):
        relay.handle(2)
    with pytest.raises(InvariantViolation, match="handles"):
        relay.publish(_handles(relay, [0]), torch.tensor([1, 2]))
    with pytest.raises(InvariantViolation):
        OmniGenerationRelay(0)
