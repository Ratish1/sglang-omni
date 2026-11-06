# SPDX-License-Identifier: Apache-2.0
"""Tensor measurements shared by the model audit commands."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class TensorDifference:
    exact: bool
    equal_fraction: float
    maximum_absolute_error: float
    relative_l2_error: float | None


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> TensorDifference:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference = actual.float() - expected.float()
    expected_norm = expected.float().norm().item()
    return TensorDifference(
        exact=torch.equal(actual, expected),
        equal_fraction=(actual == expected).float().mean().item(),
        maximum_absolute_error=difference.abs().max().item(),
        relative_l2_error=(
            difference.norm().item() / expected_norm if expected_norm else None
        ),
    )
