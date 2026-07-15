from __future__ import annotations

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_results_config,
    _config_from_args,
    _select_generation_samples,
)


def _config_from_cli(*args: str) -> TtsSeedttsBenchmarkConfig:
    parser = _build_arg_parser()
    return _config_from_args(parser.parse_args(list(args)))


def test_seedtts_benchmark_batch_args_default_to_64() -> None:
    config = _config_from_cli()

    assert config.max_running_requests == 64
    assert config.cuda_graph_max_bs == 64

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 64
    assert results_config["cuda_graph_max_bs"] == 64


def test_seedtts_benchmark_batch_args_are_independent() -> None:
    config = _config_from_cli(
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "128",
    )

    assert config.max_running_requests == 32
    assert config.cuda_graph_max_bs == 128

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 32
    assert results_config["cuda_graph_max_bs"] == 128


def test_seedtts_benchmark_accepts_internal_pool_coordination_args() -> None:
    config = _config_from_cli(
        "--sample-rotation-index",
        "1",
        "--sample-rotation-count",
        "3",
        "--barrier-ready-file",
        "/tmp/ready-1",
        "--barrier-release-file",
        "/tmp/start",
    )

    assert config.sample_rotation_index == 1
    assert config.sample_rotation_count == 3
    assert config.barrier_ready_file == "/tmp/ready-1"
    assert config.barrier_release_file == "/tmp/start"


def test_seedtts_benchmark_repeats_one_exact_sample_with_unique_ids() -> None:
    config = _config_from_cli(
        "--sample-id",
        "target-sample",
        "--sample-repetitions",
        "3",
    )
    samples = [
        SampleInput("other", "other ref", "/tmp/other.wav", "other target"),
        SampleInput("target-sample", "ref", "/tmp/target.wav", "target"),
    ]

    selected = _select_generation_samples(
        samples,
        sample_id=config.sample_id,
        repetitions=config.sample_repetitions,
    )

    assert [sample.sample_id for sample in selected] == [
        "target-sample__repeat_0000",
        "target-sample__repeat_0001",
        "target-sample__repeat_0002",
    ]
    assert all(sample.ref_audio == "/tmp/target.wav" for sample in selected)
    assert all(sample.target_text == "target" for sample in selected)
