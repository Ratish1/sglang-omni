# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from scripts.debug.moss_codec_plumbing_probe import summarize_candidate_cases


def _case(candidate: str, *, max_abs: float, speedup: float) -> dict:
    return {
        "candidate": candidate,
        "parity": {
            "shape_equal": True,
            "max_abs": max_abs,
            "mean_abs": max_abs,
            "max_rel": max_abs,
        },
        "timing": {
            "candidate_speedup_pct": speedup,
        },
    }


def test_plumbing_probe_accepts_only_exact_parity_and_consistent_speedup() -> None:
    rows = summarize_candidate_cases(
        [
            _case("good", max_abs=0.0, speedup=4.0),
            _case("good", max_abs=0.0, speedup=3.5),
            _case("drift", max_abs=0.1, speedup=30.0),
            _case("mixed", max_abs=0.0, speedup=6.0),
            _case("mixed", max_abs=0.0, speedup=-1.0),
        ],
        min_speedup_pct=3.0,
    )

    by_candidate = {row["candidate"]: row for row in rows}

    assert by_candidate["good"]["accepted"] is True
    assert by_candidate["good"]["parity_pass"] == 2
    assert by_candidate["drift"]["accepted"] is False
    assert by_candidate["drift"]["parity_fail"] == 1
    assert by_candidate["mixed"]["accepted"] is False
    assert by_candidate["mixed"]["min_speedup_pct"] < 0.0
