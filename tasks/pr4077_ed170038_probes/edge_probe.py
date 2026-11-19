"""Verify capacity and rollback boundaries with the model and transport doubled."""

import asyncio
import json
import sys

import pytest
import torch

from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import EngineArgs
from sglang_omni.models.qwen3_tts import stages
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane
from tests.unit_test.qwen3_tts.test_pipeline import (
    FakeIncrementalQwen3TTSDecoder,
    FakeQwen3TTSSpeechTokenizer,
    admit_reference_stream,
    qwen3_streaming_vocoder,
)


def rollback_case() -> dict[str, bool | str | None]:
    config = Qwen3TTSPipelineConfig(model_path="org/model")
    config.stage_named("tts_engine").engine = EngineArgs(max_queued_requests=None)
    vocoder = config.stage_named("vocoder")
    vocoder.factory.enable_stateful_codec_decoder = False

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            stages, "load_qwen3_tts_checkpoint_config",
            lambda model_path: {"tts_model_type": "base"},
        )
        patch.setattr(
            stages, "load_qwen3_tts_tokenizer",
            lambda *args, **kwargs: FakeQwen3TTSSpeechTokenizer(),
        )
        factory_args = resolve_stage_factory_args(vocoder, config, gpu_id=0)
        factory_args["device"] = "cpu"
        try:
            scheduler = stages.create_vocoder_executor(**factory_args)
        except ValueError as error:
            return {"started": False, "error": str(error)}
        try:
            return {"started": True, "has_arena": scheduler.codec_arena is not None}
        finally:
            scheduler.stop()


async def overlapping_abort_case() -> dict[str, int | bool]:
    config = Qwen3TTSPipelineConfig(model_path="org/model")
    config.stage_named("tts_engine").engine = EngineArgs(
        max_running_requests=1, max_queued_requests=1
    )
    resolved_vocoder_args = resolve_stage_factory_args(
        config.stage_named("vocoder"), config, gpu_id=0
    )
    slot_count = resolved_vocoder_args["codec_state_slots"]
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocessing",
        terminal_stages=["vocoder"],
        max_in_flight=resolved_vocoder_args["max_in_flight"],
    )
    coordinator.control_plane = RecordingCoordinatorControlPlane()
    coordinator.register_stage("preprocessing", "inproc://preprocess")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            qwen3_streaming_vocoder,
            "Qwen3TTSIncrementalDecoder",
            FakeIncrementalQwen3TTSDecoder,
        )
        scheduler = Qwen3TTSStreamingVocoderScheduler(
            FakeQwen3TTSSpeechTokenizer(),
            device="cpu",
            async_decode=True,
            enable_stateful_codec_decoder=True,
            codec_state_slots=slot_count,
            initial_chunk_frames=1,
            suppress_bootstrap_silence=False,
        )
        codes = torch.tensor([[10, 1], [20, 2], [30, 3]], dtype=torch.long)
        for request_id in ("cancelled", "running"):
            await coordinator.submit_request(
                request_id, OmniRequest(inputs="hello", params={"stream": True}),
                stream_queue=asyncio.Queue(),
            )
            state = admit_reference_stream(scheduler, request_id, codes, ref_code_len=0)
            plan, incremental = scheduler.plan_stream_decode(
                request_id, state, is_final=True, max_generated_frames=None
            )
            assert plan is not None and incremental
            if request_id == "cancelled":
                cancelled_slot = plan.slot
            else:
                scheduler.finish_codec_slots([plan.slot])

        # The stage has processed the abort. Its in-flight decode still owns the row.
        assert await coordinator.abort("cancelled")
        scheduler.abort("cancelled")
        assert cancelled_slot in scheduler.codec_slots_deferred
        await coordinator.submit_request(
            "replacement", OmniRequest(inputs="hello", params={"stream": True}),
            stream_queue=asyncio.Queue(),
        )
        replacement = admit_reference_stream(
            scheduler, "replacement", codes, ref_code_len=0
        )
        plan, incremental = scheduler.plan_stream_decode(
            "replacement", replacement, is_final=True, max_generated_frames=None
        )
        assert plan is not None
        evidence = {
            "configured_slots": slot_count,
            "coordinator_owned_requests": len(coordinator.requests),
            "occupied_slots": scheduler.codec_arena.active_slots(),
            "deferred_slots": len(scheduler.codec_slots_deferred),
            "replacement_incremental": incremental,
            "replacement_fell_back": replacement.incremental_codec_fallback,
            "exhausted_count": scheduler.codec_arena.exhausted_count,
        }
        scheduler.finish_codec_slots([cancelled_slot])
        assert scheduler.codec_arena.active_slots() == 1
        return evidence


def synchronous_case() -> dict[str, int | bool]:
    config = Qwen3TTSPipelineConfig(model_path="org/model")
    resolved_vocoder_args = resolve_stage_factory_args(
        config.stage_named("vocoder"), config, gpu_id=0
    )
    resolved_vocoder_args.pop("max_in_flight", None)
    resolved_vocoder_args.pop("running_stream_count", None)
    resolved_vocoder_args.pop("shares_engine_speech_tokenizer")
    resolved_vocoder_args.pop("gpu_id")
    resolved_vocoder_args.pop("model_path")
    resolved_vocoder_args.pop("dtype", None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            qwen3_streaming_vocoder,
            "Qwen3TTSIncrementalDecoder",
            FakeIncrementalQwen3TTSDecoder,
        )
        scheduler = Qwen3TTSStreamingVocoderScheduler(
            FakeQwen3TTSSpeechTokenizer(),
            **{
                **resolved_vocoder_args,
                "device": "cpu",
                "enable_stateful_codec_decoder": True,
                "enable_deterministic_inference": True,
                "initial_chunk_frames": 1,
            },
        )
        codes = torch.tensor([[10, 1], [20, 2], [30, 3]], dtype=torch.long)
        state = admit_reference_stream(scheduler, "synchronous", codes, ref_code_len=0)
        messages = scheduler.decode_and_emit("synchronous", state)
        assert messages and state.incremental_codec_state is not None
        assert state.codec_slot is None and scheduler.codec_arena.active_slots() == 0
        return {
            "allocated_rows": scheduler.codec_arena.num_slots,
            "active_rows": scheduler.codec_arena.active_slots(),
            "decoded_frames": state.emitted_generated_frames,
            "async_decode": scheduler.async_decode,
            "has_per_request_state": state.incremental_codec_state is not None,
        }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rollback":
        evidence = rollback_case()
    elif mode == "synchronous":
        evidence = synchronous_case()
    else:
        evidence = asyncio.run(overlapping_abort_case())
    print(json.dumps(evidence, indent=2))
