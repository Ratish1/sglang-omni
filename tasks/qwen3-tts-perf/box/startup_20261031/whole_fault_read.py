"""A whole request whose window replay fails partway, on the real codec. For each case decodes the same codes clean, then with the
failure, and prints whether the request completed, the samples before the failure against the clean run, the rest's relative L2
against the clean run, and the sample step at the join against the clean run's steps. The codes are the tree's reference recordings
encoded by the codec and repeated to length, so they are speech. The last case tears the graphs down as a real replay failure does.

usage: cd TREE && PYTHONPATH=TREE python3 whole_fault_read.py
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile
import torch

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.streaming_vocoder import Qwen3TTSStreamingVocoderScheduler
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer  # noqa: E402

CASES = ((301, 0, 2, False), (129, 400, 2, False), (2047, 0, 5, False), (301, 0, 3, True))


def speech_codes(tokenizer) -> torch.Tensor:
    pieces = []
    for path in sorted((Path.cwd() / "docs/_static/audio").glob("*.wav")):
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        pieces.append(tokenizer.encode(samples.mean(axis=1), sr=sample_rate).audio_codes[0].cpu())
    return torch.cat(pieces, dim=0)


def message(request_id: str, codes: torch.Tensor, reference_frames: int) -> IncomingMessage:
    data = Qwen3TTSState(audio_codes=codes, ref_code_len=reference_frames, finish_reason="length").to_dict()
    return IncomingMessage(request_id, "new_request", StagePayload(request_id, OmniRequest(inputs="fault"), data))


def decode(scheduler, loop, request_id: str, codes: torch.Tensor, reference_frames: int):
    scheduler.run_non_streaming_batch([message(request_id, codes, reference_frames)], loop)
    for _ in range(20000):
        batch = scheduler.collect_async_batch(scheduler.initial_queue, max_batch_size=8, batch_wait_s=0, should_block=False)
        scheduler.run_initial_work(batch)
        if not scheduler.followup_queue.empty():
            scheduler.run_followup_batch(scheduler.collect_followup_batch())
            scheduler.drain_pending_incremental(keep=0)
        else:
            pass
        if scheduler.initial_queue.empty() and not scheduler.whole_request_rows and scheduler.followup_queue.empty():
            break
        else:
            pass
    else:
        raise AssertionError("whole request did not finish")
    result = scheduler.outbox.get_nowait()
    if result.type != "result":
        return None, str(result.data)
    else:
        return np.frombuffer(result.data.data["audio_waveform"], np.float32).copy(), None


def main() -> None:
    torch.manual_seed(3477)
    tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0", dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer, device="cuda:0", tts_model_size="1b7", async_decode=True, enable_stateful_codec_decoder=True,
        incremental_codec_cuda_graph=True, incremental_codec_compile=True, fused_snake_activation=True, initial_max_batch_size=8,
        followup_max_batch_size=8, followup_worker_count=1, codec_state_slots=8, has_reference_prefixed_first_chunks=True,
    )
    scheduler.warmup_now()
    assert scheduler.initial_window_decode_graphs.enabled
    scheduler.followup_worker = object()
    loop = asyncio.new_event_loop()
    runner = scheduler.initial_window_decode_graphs
    real_decode_slots = runner.decode_slots
    with torch.inference_mode():
        speech = speech_codes(tokenizer)
    print(json.dumps({"speech_frames": speech.shape[0], "quantizers": speech.shape[1]}), flush=True)
    for index, (generated, reference_frames, failed_replay, disables) in enumerate(CASES):
        frames = generated + reference_frames
        codes = speech.repeat(frames // speech.shape[0] + 1, 1)[:frames].clone()
        with torch.inference_mode(), torch.cuda.stream(scheduler.decode_stream):
            clean, error = decode(scheduler, loop, f"clean-{index}", codes, reference_frames)
            assert error is None, error
            calls = []
            emitted_before_failure = []

            def failing(window_codes, slots, **kwargs):
                calls.append(1)
                if len(calls) == failed_replay:
                    state = scheduler.stream_states[f"fault-{index}"]
                    emitted_before_failure.append(state.emitted_generated_frames)
                    if disables:
                        runner.disable_runtime("injected")
                    else:
                        pass
                    raise RuntimeError("injected replay failure")
                else:
                    return real_decode_slots(window_codes, slots, **kwargs)

            runner.decode_slots = failing
            faulted, error = decode(scheduler, loop, f"fault-{index}", codes, reference_frames)
            runner.decode_slots = real_decode_slots
        row = {"generated": generated, "reference": reference_frames, "failed_replay": failed_replay, "disables": disables,
               "fallbacks": scheduler.codec_fallback_count, "slots_left": scheduler.codec_arena.active_slots()}
        if faulted is None:
            print(json.dumps({**row, "completed": False, "error": error}), flush=True)
            continue
        else:
            pass
        assert emitted_before_failure, "the failed replay was never reached"
        join = emitted_before_failure[0] * scheduler.samples_per_frame
        rest = faulted[join:].astype(np.float64) - clean[join:].astype(np.float64)
        steps = np.abs(np.diff(clean))
        print(json.dumps({
            **row, "completed": True, "samples": int(faulted.size), "clean_samples": int(clean.size),
            "finite": bool(np.isfinite(faulted).all()), "frames_before_failure": emitted_before_failure[0],
            "before_failure_identical": bool(np.array_equal(faulted[:join], clean[:join])),
            "rest_relative_l2": float(np.linalg.norm(rest) / max(np.linalg.norm(clean[join:]), 1e-30)),
            "join_step": float(abs(faulted[join] - faulted[join - 1])) if join else None,
            "clean_step_p999": float(np.quantile(steps, 0.999)), "clean_step_max": float(steps.max()),
        }), flush=True)
    loop.close()


if __name__ == "__main__":
    main()
