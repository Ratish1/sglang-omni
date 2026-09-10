# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from tools.minimax_music3.qualify import compare


def comparison_inputs(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    for kind in ("hidden", "condition", "latent"):
        for directory in (base, candidate):
            torch.save(torch.ones(2, 4), directory / f"seed42_chunk000_{kind}.pt")
    return SimpleNamespace(
        base=base,
        candidate=candidate,
        output=tmp_path / "metrics.json",
        require_exact=True,
    )


def test_latent_comparison_detects_drift_and_keeps_diagnostic_metrics(tmp_path):
    args = comparison_inputs(tmp_path)
    compare(args)
    assert json.loads(args.output.read_text())["exact"]
    changed = torch.ones(2, 4)
    changed[0, 0] = 2
    torch.save(changed, args.candidate / "seed42_chunk000_latent.pt")

    with pytest.raises(RuntimeError, match="differs"):
        compare(args)

    metrics = json.loads(args.output.read_text())
    assert not metrics["exact"]
    assert metrics["tensors"]["seed42_chunk000_hidden.pt"]["match"]
    assert metrics["tensors"]["seed42_chunk000_latent.pt"]["max_abs"] == 1
    assert metrics["tensors"]["seed42_chunk000_latent.pt"]["rmse"] == pytest.approx(
        8**-0.5
    )


@pytest.mark.parametrize(
    "defect", ["missing_both", "missing_candidate", "nan", "empty"]
)
def test_incomplete_or_invalid_capture_never_passes_comparison(tmp_path, defect):
    args = comparison_inputs(tmp_path)
    args.require_exact = False
    name = "seed42_chunk000_latent.pt"
    if defect == "missing_both":
        (args.base / name).unlink()
        (args.candidate / name).unlink()
    elif defect == "missing_candidate":
        (args.candidate / name).unlink()
    elif defect == "nan":
        torch.save(torch.full((2, 4), torch.nan), args.candidate / name)
    else:
        for directory in (args.base, args.candidate):
            torch.save(torch.empty(0), directory / name)

    with pytest.raises(RuntimeError, match="Incomplete"):
        compare(args)
    assert not json.loads(args.output.read_text())["exact"]
