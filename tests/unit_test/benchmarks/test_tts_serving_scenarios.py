# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.tts_serving.scenarios import BATCH_OVERSIZED_SIZE, build_scenarios
from benchmarks.tts_serving.spec import (
    BenchmarkParams,
    BenchmarkSpec,
    LoadStage,
    load_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STRESS_SPEC = PROJECT_ROOT / "benchmarks/tts_serving/examples/stress.json"
GENERATED_AUDIO_FORMATS = {"wav", "pcm", "mp3", "flac", "opus", "aac"}


def test_batch_performance_scenario_seeds_every_item() -> None:
    stage = LoadStage(
        id="batch",
        mode="closed_loop",
        request_count=32,
        max_concurrency=16,
        enabled_endpoints=("batch",),
    )
    spec = BenchmarkSpec(
        base_url="http://127.0.0.1:8000",
        model_name="test-model",
        seed=17,
        params=BenchmarkParams(
            enabled_endpoints=("batch",),
            load_stages=(stage,),
        ),
    )

    scenario = next(
        item for item in build_scenarios(spec) if item.workload == "batch_32_all_valid"
    )
    seeds = [item["seed"] for item in scenario.payload["items"]]

    assert len(seeds) == 32
    assert len(set(seeds)) == 32
    assert min(seeds) >= spec.seed


def test_stress_performance_scenarios_seed_every_generation() -> None:
    spec = load_spec(STRESS_SPEC)
    scenarios = build_scenarios(spec)

    for scenario in scenarios:
        if not scenario.expect_success:
            continue
        index = int(scenario.id.rsplit("-", 1)[1])

        if scenario.endpoint in {"speech", "speech_stream"}:
            assert scenario.payload["seed"] == spec.seed + index, scenario.id
            continue

        if scenario.endpoint == "batch":
            expected_failures = set(
                scenario.planned_metadata.get("expected_item_failures", ())
            )
            generated_seeds = []
            for item_index, item in enumerate(scenario.payload["items"]):
                response_format = item.get(
                    "response_format", scenario.payload["response_format"]
                )
                item_generates = (
                    item_index not in expected_failures
                    and isinstance(item.get("input"), str)
                    and bool(item["input"].strip())
                    and response_format in GENERATED_AUDIO_FORMATS
                )
                if not item_generates:
                    continue
                expected_seed = (
                    spec.seed + index * BATCH_OVERSIZED_SIZE + item_index
                )
                assert item["seed"] == expected_seed, (
                    scenario.id,
                    item_index,
                )
                generated_seeds.append(item["seed"])
            assert len(generated_seeds) == len(set(generated_seeds)), scenario.id
            continue

        if scenario.endpoint == "websocket":
            session_configs = [
                action["payload"]
                for action in scenario.script
                if action.get("action") == "send_json"
                and action.get("payload", {}).get("type") == "session.config"
            ]
            if session_configs:
                assert session_configs[0]["seed"] == spec.seed + index, scenario.id


def test_seeded_generation_does_not_change_validation_failure_payloads() -> None:
    scenarios = build_scenarios(load_spec(STRESS_SPEC))

    validation_failures = [
        scenario
        for scenario in scenarios
        if not scenario.expect_success
        and scenario.category
        in {"speech_length_extreme", "speech_malformed", "speech_reference"}
    ]

    assert validation_failures
    assert all("seed" not in scenario.payload for scenario in validation_failures)
