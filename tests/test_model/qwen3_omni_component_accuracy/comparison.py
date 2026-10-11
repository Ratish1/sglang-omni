# SPDX-License-Identifier: Apache-2.0
"""How far an output strays from the FP32 reference, and the floor a correct BF16 implementation sets."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch

# note (ratish): an omni component may stray from the FP32 reference at most this many times as far as the
# reference's own BF16 copy does on the same input; the first runs record the ratios this bound is reviewed against.
FLOOR_FACTOR = 2.0


@dataclass(frozen=True, kw_only=True)
class Deviation:
    relative_l2: float
    cosine: float
    max_abs: float


def deviation(candidate: torch.Tensor, truth: torch.Tensor) -> Deviation:
    candidate = candidate.detach().float().flatten().cpu()
    truth = truth.detach().float().flatten().cpu()
    if candidate.shape != truth.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} vs {truth.shape}")
    else:
        pass
    return Deviation(
        relative_l2=((candidate - truth).norm() / truth.norm()).item(),
        cosine=torch.nn.functional.cosine_similarity(candidate, truth, dim=0).item(),
        max_abs=(candidate - truth).abs().max().item(),
    )


@dataclass(frozen=True, kw_only=True)
class CaseRecord:
    """One input through omni and through the reference's BF16 copy, both measured against FP32."""

    case: str
    omni: Deviation
    floor: Deviation

    @property
    def floor_ratio(self) -> float:
        return self.omni.relative_l2 / max(self.floor.relative_l2, 1e-12)

    def as_json(self) -> dict[str, object]:
        return {
            "case": self.case,
            "omni": asdict(self.omni),
            "floor": asdict(self.floor),
            "floor_ratio": self.floor_ratio,
        }


def reference_copy(
    module: torch.nn.Module, *, device: str, dtype: torch.dtype
) -> torch.nn.Module:
    return copy.deepcopy(module).to(device=device, dtype=dtype).eval()


def parameter_mismatches(
    omni_state: dict[str, torch.Tensor], reference_state: dict[str, torch.Tensor]
) -> list[str]:
    """Names whose omni tensor is missing or differs bit for bit from the reference checkpoint's."""
    mismatches = [name for name in reference_state if name not in omni_state]
    for name, tensor in omni_state.items():
        reference = reference_state.get(name)
        if reference is None:
            mismatches.append(f"{name} (not in the reference)")
        elif not torch.equal(
            tensor.detach().cpu().to(reference.dtype), reference.detach().cpu()
        ):
            mismatches.append(name)
        else:
            pass
    return mismatches


def floor_violations(records: list[CaseRecord]) -> list[str]:
    return [
        f"{record.case}: omni relative L2 {record.omni.relative_l2:.3e}, "
        f"reference BF16 {record.floor.relative_l2:.3e} (ratio {record.floor_ratio:.2f})"
        for record in records
        if record.floor_ratio > FLOOR_FACTOR
    ]
