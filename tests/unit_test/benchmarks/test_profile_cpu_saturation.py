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


def _result(sample_count: int, *, qps: float = 10.0) -> dict:
    return {
        "summary": {
            "evaluated": sample_count,
            "total_samples": sample_count,
            "skipped": 0,
            "corpus_wer": 0.0,
        },
        "speed": {
            "throughput_samples_per_s": qps,
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


def test_nsys_joint_contract_requires_cuda_coverage() -> None:
    reports, coverage = profile._nsys_stats_contract(cpu_only=False)

    assert reports == [
        "nvtx_sum",
        "cuda_api_sum",
        "cuda_gpu_kern_sum",
        "osrt_sum",
    ]
    assert "CUDA API" in coverage
    assert "CUDA kernel" in coverage
    assert coverage["scheduler model execution"] == (
        "scheduler.model_launch",
        "scheduler.model_execute",
    )
    assert coverage["request feature extraction"] == ("request_build.feature_extract",)


def test_nsys_cpu_only_contract_does_not_request_cuda_reports() -> None:
    reports, coverage = profile._nsys_stats_contract(cpu_only=True)

    assert reports == ["nvtx_sum", "osrt_sum"]
    assert "CUDA API" not in coverage
    assert "CUDA kernel" not in coverage
    assert coverage["NVTX capture window"] == ("sglang_omni.capture_window",)
    assert coverage["pre-LM synchronize"] == ("pre_lm.synchronize",)
    assert coverage["scheduler result processing"] == ("scheduler.result_process",)


def test_nsys_nvtx_only_contract_requests_no_injected_runtime_reports() -> None:
    reports, coverage = profile._nsys_stats_contract(
        cpu_only=False,
        nvtx_only=True,
    )

    assert reports == ["nvtx_sum"]
    assert "CUDA API" not in coverage
    assert "CUDA kernel" not in coverage
    assert "OS runtime" not in coverage
    assert coverage["request-build total"] == ("request_build.total",)
    assert coverage["scheduler model execution"] == (
        "scheduler.model_launch",
        "scheduler.model_execute",
    )


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


@pytest.mark.asyncio
async def test_stability_characterization_preserves_all_unstable_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [SimpleNamespace(sample_id=str(index)) for index in range(2)]
    qps = iter([10.0, 14.0, 9.0, 13.0])
    cpu = iter([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0])

    async def fake_run(_args, selected, **_kwargs):
        return _result(len(selected), qps=next(qps))

    def fake_pressure(_args, *, target_pid):
        return {"psi": None, "cgroup_psi": None}

    def fake_threads(pid):
        return {"monotonic_ns": 1, "pid": pid, "threads": []}

    frequency_windows: list[tuple[int, int]] = []

    def fake_frequency(
        _path,
        *,
        start_monotonic_ns=None,
        stop_monotonic_ns=None,
    ):
        if start_monotonic_ns is not None and stop_monotonic_ns is not None:
            frequency_windows.append((start_monotonic_ns, stop_monotonic_ns))
        return {"samples": 2}

    monkeypatch.setattr(profile, "_run_pass", fake_run)
    monkeypatch.setattr(profile, "_build_collectors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(profile, "_read_pressure_snapshot", fake_pressure)
    monkeypatch.setattr(profile, "read_thread_snapshot", fake_threads)
    monkeypatch.setattr(profile, "_process_cpu_seconds", lambda _pid: next(cpu))
    monkeypatch.setattr(profile, "parse_cpu_frequency", fake_frequency)
    args = SimpleNamespace(
        server_pid=123,
        collectors="",
        workload_contract="direct-cold",
        shape_warmup_samples=0,
        shape_warmup_passes=1,
        warmup_samples=2,
        characterization_windows=4,
        stability_windows=3,
        stability_tolerance=0.05,
        max_thread_cpu_accounting_error=0.05,
        cpu_frequency_cpus="",
        run_id="unstable",
        mode="stability",
    )
    artifact_dir = tmp_path / "unstable"
    artifact_dir.mkdir()
    (artifact_dir / "cpu_frequency.jsonl").write_text("{}\n", encoding="utf-8")
    result = await profile._run_stability_characterization(
        args,
        samples,
        artifact_dir=artifact_dir,
    )
    characterization = result["stability_characterization"]
    assert result["capture_complete"] is True
    assert result["accepted"] is True
    assert result["request_integrity"]["valid"] is True
    assert characterization["completed_windows"] == 4
    assert characterization["ever_stable"] is False
    assert len(frequency_windows) == 4
    assert all(start <= stop for start, stop in frequency_windows)
    assert all(
        {
            "started_monotonic_ns",
            "stopped_monotonic_ns",
            "started_wall_ns",
            "stopped_wall_ns",
        }
        <= window.keys()
        for window in characterization["windows"]
    )
    assert len(list((artifact_dir / "stability_windows").glob("window_??.json"))) == 4
    assert (artifact_dir / "result.json").is_file()
    assert (artifact_dir / "artifact_index.json").is_file()


def test_stability_request_integrity_rejects_incomplete_window() -> None:
    incomplete = profile._summarize_pass(_result(3))
    incomplete["evaluated"] = 2
    incomplete["skipped"] = 1
    incomplete["request_accounting"].update(
        {
            "completed": 2,
            "failed": 1,
            "http_rejected": 1,
        }
    )
    integrity = profile._request_integrity(
        [],
        [{"window": 1, "summary": incomplete}],
    )
    assert integrity["valid"] is False
    assert integrity["completed"] == 2
    assert "completed 2/3" in integrity["errors"][0]


def test_thread_accounting_accepts_attributable_birth_but_rejects_large_gap() -> None:
    accounting = profile._thread_accounting(
        process_cpu_ms=100.0,
        thread_delta={
            "cpu_ms": 80.0,
            "threads_observed": 2,
            "threads_started": [13],
            "threads_exited": [],
        },
        max_relative_error=0.05,
    )
    assert accounting["valid"] is False
    assert accounting["relative_error"] == pytest.approx(0.2)
    assert not any("threads started" in error for error in accounting["errors"])
    assert any("relative error" in error for error in accounting["errors"])


def test_stability_system_integrity_rejects_empty_requested_evidence(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        collectors="thread-snapshot,gpu-dmon,cpu-frequency,psi,cgroup-psi",
        cpu_frequency_cpus="2-3",
        required_thread_comms="sched-asr,fun-asr-audio-e",
    )
    errors = profile._stability_system_integrity_errors(
        args,
        tmp_path,
        {
            "thread_summary": {"samples": 1, "threads": []},
            "gpu_dmon": {"samples": 0, "columns": {}},
            "cpu_frequency": {
                "samples": 1,
                "cpu_samples": 0,
                "scope": {"observed_cpus": [2]},
                "busy_weighted_sampled_scaling_frequency_mhz": None,
            },
        },
        [
            {
                "window": 1,
                "pressure": {},
                "thread_accounting": {"errors": ["CPU accounting is incomplete"]},
                "continuous_telemetry": {
                    "cpu_frequency": {
                        "samples": 1,
                        "coverage": {
                            "brackets_start": False,
                            "brackets_stop": False,
                        },
                    },
                },
            }
        ],
    )
    assert "thread snapshot collector produced fewer than two samples" in errors
    assert any("required comms" in error for error in errors)
    assert "nvidia-smi dmon produced no parseable SM samples" in errors
    assert "CPU frequency collector produced no usable frequencies" in errors
    assert "CPU frequency collector did not observe selected CPUs [3]" in errors
    assert "stability window 1 has no global CPU PSI" in errors
    assert "stability window 1 has no cgroup CPU PSI" in errors
    assert "stability window 1: CPU accounting is incomplete" in errors
    assert "stability window 1 lacks bracketing CPU frequency samples" in errors


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


def test_profile_perturbation_uses_midpoint_of_bracketed_controls() -> None:
    before = profile._summarize_pass(_result(3))
    after = profile._summarize_pass(_result(3))
    measured = profile._summarize_pass(_result(3))
    before["throughput_samples_per_s"] = 30.0
    after["throughput_samples_per_s"] = 32.0
    measured["throughput_samples_per_s"] = 34.0

    perturbation = profile._build_profile_perturbation(
        before,
        after,
        [measured],
    )

    assert perturbation["baseline_qps"] == 31.0
    assert perturbation["baseline_relative_drift"] == pytest.approx(2.0 / 31.0)
    assert perturbation["relative_qps_change"] == pytest.approx(3.0 / 31.0)


def test_unstable_controls_make_probe_effect_inconclusive() -> None:
    args = SimpleNamespace(
        mode="events",
        max_adjacent_baseline_drift=0.02,
        max_event_overhead=0.02,
        allow_event_overhead=False,
    )
    errors = profile._perturbation_integrity_errors(
        args,
        {
            "baseline_relative_drift": 0.08,
            "relative_qps_change": 0.12,
        },
    )
    assert len(errors) == 1
    assert "baseline drift" in errors[0]
    assert "inconclusive" in errors[0]
    assert "event-enabled" not in errors[0]


def test_stable_controls_expose_material_event_probe_effect() -> None:
    args = SimpleNamespace(
        mode="events",
        max_adjacent_baseline_drift=0.02,
        max_event_overhead=0.02,
        allow_event_overhead=False,
    )
    errors = profile._perturbation_integrity_errors(
        args,
        {
            "baseline_relative_drift": 0.01,
            "relative_qps_change": 0.12,
        },
    )
    assert errors == [
        "event-enabled QPS change +12.00% exceeds 2.00%; "
        "the trace may have a material probe effect"
    ]
