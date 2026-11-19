"""Trace a fractional YAML override to the actual Qwen3-TTS codec arena."""

import json

import pytest
import torch

from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.qwen3_tts import stages
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
)
from tests.unit_test.qwen3_tts.test_pipeline import (
    FakeIncrementalQwen3TTSDecoder,
    FakeQwen3TTSSpeechTokenizer,
    admit_reference_stream,
    qwen3_streaming_vocoder,
)


def probe() -> dict[str, int | float | str | bool]:
    config = Qwen3TTSPipelineConfig(model_path="model")
    vocoder = config.stage_named("vocoder")
    vocoder.factory = FactoryArgs(
        dtype="bfloat16", async_decode=True, codec_state_slots=0.5
    )
    factory_args = resolve_stage_factory_args(vocoder, config, gpu_id=0)
    factory_args["device"] = "cpu"
    assert factory_args["codec_state_slots"] == 0.5
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            stages,
            "load_qwen3_tts_checkpoint_config",
            lambda model_path: {"tts_model_type": "custom_voice"},
        )
        patch.setattr(
            stages,
            "load_qwen3_tts_tokenizer",
            lambda *args, **kwargs: FakeQwen3TTSSpeechTokenizer(),
        )
        patch.setattr(
            qwen3_streaming_vocoder,
            "Qwen3TTSIncrementalDecoder",
            FakeIncrementalQwen3TTSDecoder,
        )
        patch.setattr(
            Qwen3TTSStreamingVocoderScheduler, "warmup_now", lambda scheduler: None
        )
        try:
            scheduler = stages.create_vocoder_executor(**factory_args)
        except ValueError as error:
            return {"started": False, "message": str(error)}
        codes = torch.tensor([[10, 1], [20, 2]], dtype=torch.long)
        request_id = "first"
        state = admit_reference_stream(scheduler, request_id, codes, ref_code_len=0)
        with scheduler.state_lock:
            plan, use_incremental = scheduler.plan_stream_decode(
                request_id, state, is_final=True, max_generated_frames=None
            )
        assert plan is not None
        arena = scheduler.codec_arena
        assert arena is not None
        return {
            "started": True,
            "resolved_override": 0.5,
            "allocated_stream_slots": arena.num_slots,
            "allocated_whole_slots": len(arena.reserved_slots),
            "first_request_uses_incremental": use_incremental,
            "first_request_fell_back": state.incremental_codec_fallback,
        }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
