# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from benchmarks.same_gpu_dp.run_study import build_environment


def test_build_environment_translates_strict_study(monkeypatch) -> None:
    monkeypatch.setenv("GPU_UUID", "GPU-test")
    payload = {
        "version": 1,
        "common": {
            "gpu_uuid": "${GPU_UUID}",
            "require_idle_gpu": True,
            "max_samples": 300,
        },
        "matrix": {
            "order": ["1:0", "2:1"],
            "concurrency_values": [32, 64],
        },
        "calibration": {
            "dps": [2, 3],
            "confirmations": 3,
            "token_tolerance": 256,
        },
        "layouts": {
            2: {
                "server_core_sets": ["0-7", "8-15"],
                "client_core_sets": ["32-35", "36-39"],
                "mem_fractions": [0.42, 0.42],
                "initial_cap_tokens": 78015,
                "max_total_tokens": 84328,
            }
        },
    }

    env = build_environment(payload)

    assert env["GPU_UUID"] == "GPU-test"
    assert env["REQUIRE_IDLE_GPU"] == "1"
    assert env["MATRIX_ORDER"] == "1:0,2:1"
    assert env["CONCURRENCY_VALUES"] == "32,64"
    assert env["CALIBRATION_DPS"] == "2,3"
    assert env["CALIBRATION_CONFIRMATIONS"] == "3"
    assert env["CALIBRATION_TOKEN_TOLERANCE"] == "256"
    assert env["DP2_SERVER_CORE_SETS"] == "0-7;8-15"
    assert env["DP2_INITIAL_CAP_TOKENS"] == "78015"
    assert env["DP2_MAX_TOTAL_TOKENS"] == "84328"


def test_build_environment_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown common keys"):
        build_environment({"version": 1, "common": {"gpu": 0}})


def test_build_environment_rejects_unresolved_variable() -> None:
    with pytest.raises(ValueError, match="unresolved environment variable"):
        build_environment({"version": 1, "common": {"gpu_uuid": "${NOT_SET_FOR_TEST}"}})
