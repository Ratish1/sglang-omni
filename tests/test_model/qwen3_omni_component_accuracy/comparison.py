# SPDX-License-Identifier: Apache-2.0
"""How far an output strays from the FP32 reference, and the floor a correct BF16 implementation sets."""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import asdict, dataclass

import torch

# note (ratish): two BF16 implementations of equal accuracy differ case by case in how far they stray from FP32. Over
# the 2000 mmsu-ci-2000 clips, omni's graph-replayed audio encoder strayed 1.00 times as far as HF's own BF16 copy in
# geometric mean, with log standard deviation 0.25 and single clips from 0.12 to 6.6 times, as many below 0.5 as
# above 2. So a component fails when its geometric mean ratio passes 1.25 (over three standard errors at 24 cases),
# a systematic loss, or when one case passes 10 times, beyond the spread equal accuracy produced.
GEOMETRIC_MEAN_BOUND = 1.25
CASE_BOUND = 10.0


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


def geometric_mean_ratio(records: list[CaseRecord]) -> float:
    return math.exp(
        statistics.fmean(math.log(max(record.floor_ratio, 1e-12)) for record in records)
    )


def floor_violations(records: list[CaseRecord]) -> list[str]:
    violations = [
        f"{record.case}: omni relative L2 {record.omni.relative_l2:.3e}, "
        f"reference BF16 {record.floor.relative_l2:.3e} (ratio {record.floor_ratio:.2f})"
        for record in records
        if record.floor_ratio > CASE_BOUND
    ]
    geometric_mean = geometric_mean_ratio(records)
    if geometric_mean > GEOMETRIC_MEAN_BOUND:
        violations.append(
            f"geometric mean ratio {geometric_mean:.3f} over {len(records)} cases passes {GEOMETRIC_MEAN_BOUND}"
        )
    else:
        pass
    return violations
