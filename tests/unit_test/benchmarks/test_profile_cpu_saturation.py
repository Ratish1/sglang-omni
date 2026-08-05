# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("sglang")

from benchmarks.profiling import profile_cpu_saturation as profile


def _write_wav(path: Path, frames: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * frames)


def _result(sample_count: int) -> dict:
    return {
        "summary": {
            "evaluated": sample_count,
            "total_samples": sample_count,
            "skipped": 0,
            "corpus_wer": 0.0,
        },
        "speed": {
            "throughput_samples_per_s": 10.0,
            "latency_mean_s": 0.1,
            "latency_median_s": 0.1,
            "latency_p95_s": 0.1,
            "latency_p99_s": 0.1,
            "rtf_mean": 0.1,
            "rtfx": 10.0,
        },
        "wall_clock_s": 1.0,
        "per_sample": [
            {
                "id": str(index),
                "is_success": True,
                "error": "",
                "http_status": 200,
                "server_request_id": f"r-{index}",
                "client_timing": {
                    "http_start_ns": index + 1,
                    "client_queue_s": 0.0,
                },
            }
            for index in range(sample_count)
        ],
    }


@pytest.mark.asyncio
async def test_steady_contract_warms_full_shape_population_before_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = []
    for index in range(4):
        wav = tmp_path / f"{index}.wav"
        _write_wav(wav, frames=(index + 1) * 160)
        samples.append(
            SimpleNamespace(
                sample_id=str(index),
                ref_audio=str(wav),
                ref_text="x",
                target_text="x",
            )
        )
    calls: list[int] = []

    async def fake_run(_args, selected, **_kwargs):
        calls.append(len(selected))
        return _result(len(selected))

    monkeypatch.setattr(profile, "_run_pass", fake_run)
    args = SimpleNamespace(
        workload_contract="direct-steady-miss",
        shape_warmup_samples=0,
        shape_warmup_passes=1,
        warmup_samples=2,
        concurrency=1,
        max_warmup_windows=2,
        stability_windows=2,
        stability_tolerance=0.05,
        server_pid=None,
    )
    result = await profile._warm_to_stability(
        args,
        samples,
        artifact_dir=tmp_path / "artifacts",
    )
    assert calls == [4, 2, 2]
    assert result["shape_passes"][0]["samples"] == 4
    assert len(result["stability_windows"]) == 2


def test_summary_preserves_offered_dispatched_rejected_and_client_queue() -> None:
    result = _result(3)
    result["summary"]["evaluated"] = 2
    result["summary"]["skipped"] = 1
    result["per_sample"][2]["is_success"] = False
    result["per_sample"][2]["http_status"] = 500
    result["per_sample"][2]["error"] = "HTTP 500: backlog full"
    result["per_sample"][2]["client_timing"]["client_queue_s"] = 0.25
    summary = profile._summarize_pass(result)
    assert summary["request_accounting"] == {
        "offered": 3,
        "http_dispatched": 3,
        "completed": 2,
        "failed": 1,
        "http_rejected": 1,
        "timed_out": 0,
        "max_client_queue_s": 0.25,
    }
