"""A whole request whose window replay fails, on the real codec and speech codes: the tree's reference recordings encoded by the
codec and repeated to length. Decodes each case clean, then with the failure, and prints whether every request completed, each
faulted row's sample count, finiteness and relative L2 against the clean run. The last case holds nine rows, so the replay that
fails and tears the graphs down holds eight and the ninth continues without them.

usage: cd TREE && PYTHONPATH=TREE python3 whole_fault_read.py [--only-case N] [--no-clean] [--no-native]
--no-clean skips the clean decode and the comparison, and --no-native returns silence from the whole decode, to bisect a fault's sequence.
"""

import argparse
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

# generated frames, reference frames, rows, the replay that fails, whether it tears the graphs down
CASES = ((301, 0, 1, 2, False), (129, 400, 1, 2, False), (2047, 0, 1, 5, False), (130, 0, 9, 1, True))


def speech_codes(tokenizer) -> torch.Tensor:
    pieces = []
    for path in sorted((Path.cwd() / "docs/_static/audio").glob("*.wav")):
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        pieces.append(tokenizer.encode(samples.mean(axis=1), sr=sample_rate).audio_codes[0].cpu())
    return torch.cat(pieces, dim=0)


def message(request_id: str, codes: torch.Tensor, reference_frames: int) -> IncomingMessage:
    data = Qwen3TTSState(audio_codes=codes, ref_code_len=reference_frames, finish_reason="length").to_dict()
    return IncomingMessage(request_id, "new_request", StagePayload(request_id, OmniRequest(inputs="fault"), data))


def decode(scheduler, loop, rows: list[tuple[str, torch.Tensor]], reference_frames: int) -> dict:
    scheduler.run_non_streaming_batch([message(name, codes, reference_frames) for name, codes in rows], loop)
    for _ in range(20000):
        batch = scheduler.collect_async_batch(scheduler.initial_queue, max_batch_size=8, batch_wait_s=0, should_block=False)
        scheduler.run_initial_work(batch)
        if scheduler.initial_queue.empty() and not scheduler.whole_request_rows:
            break
        else:
            pass
    else:
        raise AssertionError("whole requests did not finish")
    results = {}
    while not scheduler.outbox.empty():
        result = scheduler.outbox.get_nowait()
        results[result.request_id] = (
            np.frombuffer(result.data.data["audio_waveform"], np.float32).copy() if result.type == "result" else str(result.data)
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-case", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--no-native", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(3477)
    tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0", dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer, device="cuda:0", tts_model_size="1b7", async_decode=True, enable_stateful_codec_decoder=True,
        incremental_codec_cuda_graph=True, incremental_codec_compile=True, fused_snake_activation=True, initial_max_batch_size=8,
        followup_max_batch_size=8, followup_worker_count=1, codec_state_slots=16, has_reference_prefixed_first_chunks=True,
    )
    scheduler.warmup_now()
    runner = scheduler.initial_window_decode_graphs
    assert runner.enabled
    print(json.dumps({"stream_slots": scheduler.codec_arena.num_slots, "whole_request_slots": len(scheduler.free_whole_request_slots),
                      "scratch_slot": scheduler.codec_arena.scratch_slot}), flush=True)
    loop = asyncio.new_event_loop()
    real_decode_slots = runner.decode_slots
    with torch.inference_mode():
        speech = speech_codes(tokenizer)
    if args.no_native:
        scheduler.decode_state_audio = lambda state: torch.zeros(
            (len(state.audio_codes) - state.ref_code_len) * scheduler.samples_per_frame
        )
    else:
        pass
    for index, (generated, reference_frames, row_count, failed_replay, disables) in enumerate(CASES):
        if args.only_case is not None and index != args.only_case:
            continue
        else:
            pass
        frames = generated + reference_frames
        rows = [(f"{index}-{row}", speech[row:].repeat(frames // speech.shape[0] + 1, 1)[:frames].clone()) for row in range(row_count)]
        with torch.inference_mode(), torch.cuda.stream(scheduler.decode_stream):
            clean = {} if args.no_clean else decode(scheduler, loop, rows, reference_frames)
            calls = []

            def failing(window_codes, slots, **kwargs):
                calls.append(1)
                if len(calls) == failed_replay:
                    if disables:
                        runner.disable_runtime("injected")
                    else:
                        pass
                    raise RuntimeError("injected replay failure")
                else:
                    return real_decode_slots(window_codes, slots, **kwargs)

            runner.decode_slots = failing
            faulted = decode(scheduler, loop, rows, reference_frames)
            runner.decode_slots = real_decode_slots
        for name, _ in rows:
            row = {"case": index, "row": name, "generated": generated, "reference": reference_frames, "failed_replay": failed_replay,
                   "disables": disables, "free_slots": len(scheduler.free_whole_request_slots), "window_calls": len(calls)}
            if isinstance(faulted[name], str):
                print(json.dumps({**row, "completed": False, "error": faulted[name][:200]}), flush=True)
            elif args.no_clean:
                print(json.dumps({**row, "completed": True, "samples": int(faulted[name].size)}), flush=True)
            else:
                delta = faulted[name].astype(np.float64) - clean[name].astype(np.float64)
                print(json.dumps({**row, "completed": True, "samples": int(faulted[name].size), "clean_samples": int(clean[name].size),
                                  "finite": bool(np.isfinite(faulted[name]).all()),
                                  "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(clean[name]), 1e-30))}), flush=True)
    loop.close()


if __name__ == "__main__":
    main()
