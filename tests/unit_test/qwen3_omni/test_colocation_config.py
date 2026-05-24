# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import build_process_topology_plan, build_stage_placement_plan
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
    Variants,
)
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _set_colocated_runtime(
    config: Qwen3OmniSpeechColocatedPipelineConfig,
    *,
    include_mem_fraction: bool = True,
    conflicting_mem_fraction: bool = False,
) -> None:
    fractions = {
        "image_encoder": 0.025,
        "audio_encoder": 0.025,
        "thinker": 0.75,
        "talker_ar": 0.12,
        "code2wav": 0.02,
    }
    for stage_name, fraction in fractions.items():
        _stage(config, stage_name).runtime.resources.total_gpu_memory_fraction = (
            fraction
        )
    if include_mem_fraction:
        _stage(config, "thinker").runtime.sglang_server_args.mem_fraction_static = (
            0.74 if conflicting_mem_fraction else 0.75
        )
        _stage(config, "talker_ar").runtime.sglang_server_args.mem_fraction_static = (
            0.11 if conflicting_mem_fraction else 0.12
        )


def test_default_speech_topology_stays_disaggregated() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    assert len(config.stages) == 8
    assert _stage(config, "thinker").gpu == 0
    assert _stage(config, "talker_ar").gpu == 1
    assert _stage(config, "code2wav").gpu == 1
    assert config.placement.require_memory_fraction_for_colocation is False
    assert {stage.name: stage.process for stage in config.stages} == {
        "preprocessing": "preprocessing",
        "image_encoder": "image_encoder",
        "audio_encoder": "audio_encoder",
        "mm_aggregate": "mm_aggregate",
        "thinker": "thinker",
        "decode": "decode",
        "talker_ar": "talker_ar",
        "code2wav": "code2wav",
    }
    assert "code_predictor" not in {stage.name for stage in config.stages}

    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]


def test_colocated_topology_is_opt_in_and_uses_one_gpu() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    assert Variants["speech-colocated"] is Qwen3OmniSpeechColocatedPipelineConfig
    for stage_name in (
        "image_encoder",
        "audio_encoder",
        "thinker",
        "talker_ar",
        "code2wav",
    ):
        assert _stage(config, stage_name).gpu == 0
        assert _stage(config, stage_name).process == stage_name


def test_colocated_config_passes_with_explicit_budgets_without_ar_mem_fraction() -> (
    None
):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config, include_mem_fraction=False)

    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]


def test_colocated_config_marks_same_gpu_stream_targets() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)

    plan = build_stage_placement_plan(config)

    assert plan.same_gpu_stream_targets["thinker"] == frozenset({"talker_ar"})
    assert plan.same_gpu_stream_targets["talker_ar"] == frozenset({"code2wav"})


def test_default_speech_marks_only_talker_to_code2wav_same_gpu_stream() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    plan = build_stage_placement_plan(config)

    assert "thinker" not in plan.same_gpu_stream_targets
    assert plan.same_gpu_stream_targets["talker_ar"] == frozenset({"code2wav"})


def test_colocated_config_rejects_conflicting_ar_mem_fraction() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config, conflicting_mem_fraction=True)

    with pytest.raises(ValueError, match="conflicting memory fractions"):
        build_stage_placement_plan(config)


def test_colocated_config_rejects_missing_stage_budgets() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


@pytest.mark.parametrize("stage_name", ["talker_ar", "code2wav"])
def test_colocated_config_rejects_moving_gpu_stage_away(stage_name: str) -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    _stage(config, stage_name).gpu = 1

    with pytest.raises(ValueError, match="share one GPU"):
        build_stage_placement_plan(config)


def test_colocated_config_rejects_topology_override_before_runtime_validation() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    _stage(config, "talker_ar").gpu = 1

    with pytest.raises(ValueError, match="share one GPU"):
        build_stage_placement_plan(config)


def test_default_speech_rejects_same_gpu_thinker_and_talker_colocation() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    _stage(config, "talker_ar").gpu = 0
    _stage(config, "code2wav").gpu = 0
    for stage_name in (
        "image_encoder",
        "audio_encoder",
        "thinker",
        "talker_ar",
        "code2wav",
    ):
        _stage(config, stage_name).runtime.resources.total_gpu_memory_fraction = 0.10

    with pytest.raises(ValueError, match="Qwen3OmniSpeechColocatedPipelineConfig"):
        build_stage_placement_plan(config)


def test_default_speech_allows_thinker_tp_placement() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, "thinker")
    thinker.tp_size = 2
    thinker.parallelism.tp = 2
    thinker.gpu = [0, 1]

    plan = build_stage_placement_plan(config)

    assert plan.stages["thinker"].gpu_ids == (0, 1)


def test_default_speech_allows_talker_tp_with_single_rank_code2wav() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker = _stage(config, "talker_ar")
    talker.tp_size = 2
    talker.parallelism.tp = 2
    talker.gpu = [1, 2]
    _stage(config, "code2wav").gpu = 2

    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert plan.stages["talker_ar"].gpu_ids == (1, 2)
    assert plan.stages["code2wav"].gpu_ids == (2,)
    assert topology.tp_stage_to_processes["talker_ar"] == (
        "talker_ar_tp0",
        "talker_ar_tp1",
    )
    assert "talker_ar" not in topology.stage_to_process
    assert topology.stage_to_process["code2wav"] == "code2wav"


def test_default_speech_talker_tp_builds_rank_specific_stage_specs() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker = _stage(config, "talker_ar")
    talker.tp_size = 2
    talker.parallelism.tp = 2
    talker.gpu = [1, 2]
    _stage(config, "code2wav").gpu = 2

    prep = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            stages_cfg=prep.stages_cfg,
            name_map=prep.name_map,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
    finally:
        prep.runtime_dir.close()

    specs = [
        spec
        for group in groups
        for process in group.process_specs
        for spec in process.stage_specs
        if spec.stage_name == "talker_ar"
    ]
    assert [spec.role for spec in specs] == ["leader", "follower"]
    assert [spec.gpu_id for spec in specs] == [1, 2]
    assert [spec.tp_rank for spec in specs] == [0, 1]
    assert {spec.tp_size for spec in specs} == {2}
    assert specs[0].recv_endpoint
    assert specs[1].recv_endpoint == ""
    assert specs[0].nccl_port is not None
    assert specs[0].nccl_port == specs[1].nccl_port
    assert [spec.factory_args["gpu_id"] for spec in specs] == [1, 2]
    assert [spec.factory_args["tp_rank"] for spec in specs] == [0, 1]
    assert {spec.factory_args["tp_size"] for spec in specs} == {2}
    assert specs[0].factory_args["nccl_port"] == specs[0].nccl_port
    assert specs[1].factory_args["nccl_port"] == specs[0].nccl_port


def test_default_speech_rejects_talker_tp_overlap_with_thinker() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, "thinker")
    thinker.tp_size = 2
    thinker.parallelism.tp = 2
    thinker.gpu = [0, 1]
    talker = _stage(config, "talker_ar")
    talker.tp_size = 2
    talker.parallelism.tp = 2
    talker.gpu = [1, 2]

    with pytest.raises(ValueError, match="may share a GPU only"):
        build_stage_placement_plan(config)


def test_default_speech_rejects_code2wav_tp() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    code2wav = _stage(config, "code2wav")
    code2wav.tp_size = 2
    code2wav.parallelism.tp = 2
    code2wav.gpu = [1, 2]

    with pytest.raises(ValueError, match="code2wav does not support TP"):
        build_stage_placement_plan(config)


def test_colocated_config_rejects_thinker_tp() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    thinker = _stage(config, "thinker")
    thinker.tp_size = 2
    thinker.parallelism.tp = 2
    thinker.gpu = [0, 1]

    with pytest.raises(ValueError, match="thinker TP"):
        build_stage_placement_plan(config)


def test_colocated_config_rejects_talker_tp() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    talker = _stage(config, "talker_ar")
    talker.tp_size = 2
    talker.parallelism.tp = 2
    talker.gpu = [0, 1]

    with pytest.raises(ValueError, match="talker_ar TP"):
        build_stage_placement_plan(config)
