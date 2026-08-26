from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_request_seed_fn,
    _build_results_config,
    _config_from_args,
    _parse_concurrencies,
    _validate_args,
    derive_sample_specific_seed,
    plan_sustained_overshoot,
    run_tts_concurrency_sweep,
    run_tts_seedtts_transcribe,
    run_tts_sustained_overshoot,
)
from sglang_omni.admission import QueueFullError


def _config_from_cli(*args: str) -> TtsSeedttsBenchmarkConfig:
    parser = _build_arg_parser()
    parsed = parser.parse_args(list(args))
    _validate_args(parser, parsed)
    return _config_from_args(parsed)


def test_seedtts_benchmark_batch_args_default_to_64() -> None:
    config = _config_from_cli()

    assert config.max_running_requests == 64
    assert config.cuda_graph_max_bs == 64
    assert config.max_queued_requests is None

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 64
    assert results_config["cuda_graph_max_bs"] == 64
    assert results_config["max_queued_requests"] is None
    assert results_config["seed_mode"] == "none"


def test_seedtts_benchmark_batch_args_are_independent() -> None:
    config = _config_from_cli(
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "128",
        "--max-queued-requests",
        "16",
    )

    assert config.max_running_requests == 32
    assert config.cuda_graph_max_bs == 128
    assert config.max_queued_requests == 16

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 32
    assert results_config["cuda_graph_max_bs"] == 128
    assert results_config["max_queued_requests"] == 16


def test_seedtts_benchmark_records_managed_server_engine_stage() -> None:
    config = _config_from_cli("--server-engine-stage", "tts_engine")

    assert config.server_engine_stage == "tts_engine"
    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["server_engine_stage"] == "tts_engine"


def test_seedtts_benchmark_records_quantization() -> None:
    config = _config_from_cli("--quantization", "fp8")
    assert config.quantization == "fp8"
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["quantization"] == "fp8"


def test_seedtts_benchmark_quantization_defaults_to_none() -> None:
    config = _config_from_cli()
    assert config.quantization is None
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["quantization"] is None


def test_sample_specific_seed_is_stable_paired_and_sample_dependent() -> None:
    first = derive_sample_specific_seed(20260823, "sample-a")
    assert first == 636208485
    assert derive_sample_specific_seed(20260823, "sample-a") == first
    assert derive_sample_specific_seed(20260823, "sample-b") != first
    assert derive_sample_specific_seed(20260824, "sample-a") != first
    assert 0 <= first <= 0x7FFFFFFF


def test_sample_specific_seed_mode_records_and_resolves_request_seeds() -> None:
    config = _config_from_cli(
        "--seed",
        "20260823",
        "--sample-specific-seeds",
        "--temperature",
        "0.9",
        "--top-p",
        "0.95",
        "--top-k",
        "50",
        "--repetition-penalty",
        "1.05",
    )
    seed_fn = _build_request_seed_fn(config)
    assert seed_fn is not None
    assert seed_fn(_sample("sample-a")) == 636208485
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["seed"] == 20260823
    assert results_config["seed_mode"] == "sample_specific_blake2b_v1"
    assert results_config["temperature"] == 0.9
    assert results_config["top_p"] == 0.95
    assert results_config["top_k"] == 50
    assert results_config["repetition_penalty"] == 1.05


def test_sample_specific_seed_mode_requires_non_negative_panel_seed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_arg_parser()
    for argv, needle in (
        (["--sample-specific-seeds"], "requires --seed"),
        (["--sample-specific-seeds", "--seed", "-1"], "must be non-negative"),
    ):
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit) as exc:
            _validate_args(parser, args)
        assert exc.value.code == 2
        assert needle in capsys.readouterr().err

    with pytest.raises(ValueError, match="require a panel seed"):
        _build_request_seed_fn(
            TtsSeedttsBenchmarkConfig(
                model="model",
                meta="meta",
                sample_specific_seeds=True,
            )
        )


def test_transcribe_artifact_records_sampling_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_transcribe(config, *, wer_config, **kwargs):
        del config, kwargs
        captured.update(wer_config)
        return {"wer_summary": {}, "asr_speed": {}, "per_sample": []}

    monkeypatch.setattr(
        "benchmarks.eval.benchmark_tts_seedtts.run_seedtts_transcribe",
        _fake_transcribe,
    )
    config = _config_from_cli(
        "--seed",
        "20260823",
        "--sample-specific-seeds",
        "--temperature",
        "0.9",
        "--top-p",
        "0.95",
        "--top-k",
        "50",
        "--repetition-penalty",
        "1.05",
    )

    run_tts_seedtts_transcribe(config, asr_router_port=9000)

    assert captured["seed"] == 20260823
    assert captured["seed_mode"] == "sample_specific_blake2b_v1"
    assert captured["temperature"] == 0.9
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 50
    assert captured["repetition_penalty"] == 1.05


def test_parse_concurrencies() -> None:
    assert _parse_concurrencies("16,32,48,64") == [16, 32, 48, 64]


def test_plan_sustained_overshoot() -> None:
    plan = plan_sustained_overshoot(
        max_running_requests=32,
        max_queued_requests=16,
        duration_s=10,
    )
    assert plan.capacity == 48
    assert plan.concurrency == 0
    assert plan.request_rate == 96
    assert plan.request_count == 960
    assert plan.overshoot_ratio == 2.0
    with pytest.raises(ValueError, match="request_rate > admission capacity"):
        plan_sustained_overshoot(
            max_running_requests=32,
            max_queued_requests=16,
            duration_s=10,
            request_rate=48,
        )


def _sample(sample_id: str) -> SampleInput:
    return SampleInput(
        sample_id=sample_id, ref_text="r", ref_audio="r.wav", target_text="t"
    )


def _stub_seedtts_benchmark(monkeypatch, run_fn, *, samples=None) -> None:
    monkeypatch.setattr(
        "benchmarks.eval.benchmark_tts_seedtts.run_tts_seedtts_benchmark",
        run_fn,
    )
    monkeypatch.setattr(
        "benchmarks.eval.benchmark_tts_seedtts.print_speed_summary",
        lambda *args, **kwargs: None,
    )
    if samples is not None:
        monkeypatch.setattr(
            "benchmarks.eval.benchmark_tts_seedtts._load_benchmark_samples",
            lambda _config: samples,
        )


@pytest.mark.parametrize(
    "argv, needle",
    [
        (
            [
                "--generate-only",
                "--sustained-overshoot",
                "--max-running-requests",
                "32",
            ],
            "max-queued-requests",
        ),
        (
            [
                "--generate-only",
                "--sustained-overshoot",
                "--max-queued-requests",
                "16",
                "--concurrencies",
                "16,32",
            ],
            "cannot be combined",
        ),
    ],
)
def test_cli_rejects_invalid_overshoot(
    capsys: pytest.CaptureFixture[str], argv: list[str], needle: str
) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        _validate_args(parser, args)
    assert exc.value.code == 2
    assert needle in capsys.readouterr().err


@pytest.mark.asyncio
async def test_concurrency_sweep_writes_per_point_output_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_output_dirs: list[str] = []

    async def _fake_run(config: TtsSeedttsBenchmarkConfig, **_kwargs) -> dict:
        seen_output_dirs.append(config.output_dir)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.output_dir) / "speed_results.json").write_text(
            "{}", encoding="utf-8"
        )
        return {
            "summary": {
                "completed_requests": 1,
                "failed_requests": 0,
                "latency_p95_s": 0.1,
                "audio_ttfp_p95_s": 0.05,
            }
        }

    _stub_seedtts_benchmark(monkeypatch, _fake_run)

    config = _config_from_cli("--output-dir", str(tmp_path))
    payload = await run_tts_concurrency_sweep(config, [16, 32])
    expected = [str(tmp_path / "c16"), str(tmp_path / "c32")]
    assert seen_output_dirs == expected
    assert [row["output_dir"] for row in payload["rows"]] == expected
    assert (tmp_path / "c16" / "speed_results.json").is_file()
    assert (tmp_path / "c32" / "speed_results.json").is_file()


@pytest.mark.asyncio
async def test_sustained_overshoot_uses_open_loop_and_unique_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[TtsSeedttsBenchmarkConfig] = []
    seen_samples: list[list[str]] = []

    async def _fake_run(config: TtsSeedttsBenchmarkConfig, **kwargs) -> dict:
        seen.append(config)
        seen_samples.append([sample.sample_id for sample in kwargs["samples"]])
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        return {
            "summary": {
                "completed_requests": 1,
                "failed_requests": 1,
                "latency_p95_s": 0.4,
                "audio_ttfp_p95_s": 0.05,
            },
            "per_request": [
                {
                    "is_success": True,
                    "error": None,
                    "audio_ttfp_s": 0.05,
                    "latency_s": 0.4,
                },
                {
                    "is_success": False,
                    "error": f"HTTP 503: {QueueFullError.MESSAGE}",
                    "audio_ttfp_s": None,
                    "latency_s": 0.01,
                },
                {
                    "is_success": False,
                    "error": "HTTP 500: boom",
                    "audio_ttfp_s": None,
                    "latency_s": 1.0,
                },
            ],
            "config": {"concurrency": config.concurrency},
        }

    _stub_seedtts_benchmark(monkeypatch, _fake_run, samples=[_sample("s0")])

    payload = await run_tts_sustained_overshoot(
        _config_from_cli(
            "--output-dir",
            str(tmp_path),
            "--generate-only",
            "--sustained-overshoot",
            "--max-running-requests",
            "32",
            "--max-queued-requests",
            "16",
            "--overshoot-duration-s",
            "1",
            "--request-rate",
            "64",
        )
    )

    assert seen[0].concurrency == 0
    assert seen[0].request_rate == 64
    assert seen[0].output_dir == str(tmp_path / "overshoot")
    assert payload["outcomes"] == {
        "total_requests": 3,
        "success": 1,
        "queue_full": 1,
        "other_failed": 1,
        "success_ttfa_p95_s": 0.05,
        "queue_full_latency_p95_s": 0.01,
    }
    assert len(set(seen_samples[0])) == 64
    assert (tmp_path / "overshoot" / "sustained_overshoot.json").is_file()
