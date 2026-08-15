# SPDX-License-Identifier: Apache-2.0
"""CLI override tests for ``--prefill-admission eager|batched``.

The flag maps onto the stage factories' ``defer_prefill_during_decode``
argument for the supported OmniScheduler AR stages and must survive the
static-args, defaults, and factory-signature resolvers the launch path uses.
"""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_prefill_admission_cli_overrides
from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_factory_signature_args,
    resolve_stage_factory_arg_defaults,
    resolve_stage_static_factory_args,
)
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from sglang_omni.utils.imports import import_string


def _ar_stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == stage_name)
    return resolve_factory_signature_args(
        import_string(stage.factory),
        resolve_stage_static_factory_args(stage, config),
        defaults=resolve_stage_factory_arg_defaults(stage, config),
    )


@pytest.mark.parametrize(
    ("config_cls", "stage_name", "mode", "expected"),
    [
        (Qwen3ASRPipelineConfig, "asr", "eager", False),
        (WhisperASRPipelineConfig, "asr", "eager", False),
        (MossTranscribeDiarizePipelineConfig, "asr", "eager", False),
        (FunASRPipelineConfig, "asr", "batched", True),
        (HiggsTtsPipelineConfig, "tts_engine", "batched", True),
        (MossTTSLocalPipelineConfig, "tts_engine", "batched", True),
        (Qwen3OmniPipelineConfig, "thinker", "batched", True),
    ],
)
def test_cli_overrides_the_model_default(config_cls, stage_name, mode, expected):
    config = config_cls(model_path="dummy")
    apply_prefill_admission_cli_overrides(config, prefill_admission=mode)

    assert _ar_stage_args(config, stage_name)["defer_prefill_during_decode"] is expected


def test_per_stage_yaml_runtime_override_reaches_factory(tmp_path):
    config_path = tmp_path / "moss_local.yaml"
    config_path.write_text(
        """
config_cls: MossTTSLocalPipelineConfig
model_path: dummy
runtime_overrides:
  tts_engine:
    defer_prefill_during_decode: true
"""
    )
    config = ConfigManager.from_file(str(config_path)).config
    assert config is not None

    assert _ar_stage_args(config, "tts_engine")["defer_prefill_during_decode"] is True


def test_rejects_unknown_mode():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter, match="eager, batched"):
        apply_prefill_admission_cli_overrides(config, prefill_admission="lazy")


def test_rejects_unsupported_pipeline():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_prefill_admission_cli_overrides(config, prefill_admission="batched")
