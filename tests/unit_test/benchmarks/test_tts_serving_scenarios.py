# SPDX-License-Identifier: Apache-2.0

from benchmarks.tts_serving.scenarios import build_scenarios
from benchmarks.tts_serving.spec import BenchmarkParams, BenchmarkSpec, LoadStage


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
