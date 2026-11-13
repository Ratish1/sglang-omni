"""bf16 whole requests on reserved slots and main's bf16 non-streaming decode, each against an fp32 decode of the same codes.

Seeded random codes cover each path a whole request takes: a reference then one padded window after the tail decode,
many 64-frame windows after the tail, a reference past the first decode's reach, no reference, and a short request.
Each row gives the relative L2 error of the generated audio and of its worst 64-frame window for this tree's whole
requests (card 0), main's path one request at a time and main's path with every request in one batch (card 1). The
fp32 reference is one stateful decode of all frames with TF32 off; the whole-utterance fp32 forward is printed beside it.

usage: PYTHONPATH=TREE python3 bf16_whole_requests.py [--checkpoint Qwen/Qwen3-TTS-12Hz-1.7B-Base]
"""

import argparse
import asyncio
import json

import numpy as np
import torch

from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

CASES = {
    "one_window": (60, 45),
    "many_windows": (120, 400),
    "past_reach": (700, 150),
    "no_reference": (0, 300),
    "short": (30, 5),
}
WINDOW_FRAMES = 64


def message(
    request_id: str, codes: torch.Tensor, reference_frames: int
) -> IncomingMessage:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=request_id, params={}, metadata={"tts_params": {}}),
        data=Qwen3TTSState(audio_codes=codes, ref_code_len=reference_frames).to_dict(),
    )
    return IncomingMessage(request_id, "new_request", payload)


def drain(scheduler) -> dict[str, np.ndarray]:
    audio = {}
    while not scheduler.outbox.empty():
        result = scheduler.outbox.get_nowait()
        assert (
            result.type == "result"
        ), f"{result.request_id}: {result.type} {result.data}"
        audio[result.request_id] = np.frombuffer(
            result.data.data["audio_waveform"], dtype=np.float32
        ).copy()
    return audio


def errors(
    actual: np.ndarray, expected: np.ndarray, window_samples: int
) -> tuple[float, float]:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    whole = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    worst = max(
        float(
            np.linalg.norm(
                actual[start : start + window_samples]
                - expected[start : start + window_samples]
            )
            / np.linalg.norm(expected[start : start + window_samples])
        )
        for start in range(0, expected.shape[0], window_samples)
    )
    return whole, worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    args = parser.parse_args()
    reference_tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
        args.checkpoint,
        device="cuda:0",
        dtype="float32",
        attn_implementation=None,
        float32_encoder=False,
    )
    reference_decoder = reference_tokenizer.model.decoder
    generator = torch.Generator().manual_seed(42)
    codes = {
        name: torch.randint(
            reference_decoder.config.codebook_size,
            (reference + generated, reference_decoder.config.num_quantizers),
            generator=generator,
        )
        for name, (reference, generated) in CASES.items()
    }
    samples_per_frame = int(reference_decoder.total_upsample)
    expected, whole_forward = {}, {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        for name, (reference, _) in CASES.items():
            decoder_input = codes[name].T.unsqueeze(0).to("cuda:0")
            expected[name] = (
                Qwen3TTSIncrementalDecoder(reference_decoder)
                .decode(decoder_input, Qwen3TTSIncrementalCodecState())
                .flatten()[reference * samples_per_frame :]
                .cpu()
                .numpy()
            )
            whole_forward[name] = (
                reference_decoder(decoder_input)
                .flatten()[reference * samples_per_frame :]
                .cpu()
                .numpy()
            )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    del reference_tokenizer, reference_decoder
    torch.cuda.empty_cache()
    messages = [
        message(name, codes[name], reference) for name, (reference, _) in CASES.items()
    ]

    main_scheduler = qwen3_stages.create_vocoder_executor(
        args.checkpoint,
        device="cuda",
        gpu_id=1,
        incremental_codec_cuda_graph_window_frames=(),
        shares_engine_speech_tokenizer=False,
    )
    assert (
        not main_scheduler.decodes_whole_requests
        and main_scheduler.full_decode_graphs.graphs
    )
    for request in messages:
        main_scheduler.run_non_streaming_batch([request], asyncio.new_event_loop())
    main_alone = drain(main_scheduler)
    main_scheduler.run_non_streaming_batch(messages, asyncio.new_event_loop())
    main_batched = drain(main_scheduler)

    scheduler = qwen3_stages.create_vocoder_executor(
        args.checkpoint, device="cuda", gpu_id=0, shares_engine_speech_tokenizer=False
    )
    assert scheduler.decodes_whole_requests and not scheduler.full_decode_graphs.graphs
    assert scheduler.codec_arena.key_rings, "the bf16 arena keeps its K/V in rings"
    scheduler.activate_decode_worker(scheduler.decode_stream)
    scheduler.run_non_streaming_batch(messages, asyncio.new_event_loop())
    while not scheduler.initial_queue.empty() or scheduler.whole_requests:
        scheduler.run_initial_pass()
    whole = drain(scheduler)
    graph_stats = scheduler.codec_state_stats()["cuda_graphs"]
    print(
        f"arena ring frames {scheduler.codec_arena.ring_frames}, reserved slots {list(scheduler.codec_arena.reserved_slots)[:2]}..., "
        f"tail replays {graph_stats['tail']['runtime']['replays']}, window replays {graph_stats['window']['runtime']['replays']}, "
        f"idle whole slots {len(scheduler.idle_whole_request_slots)}",
        flush=True,
    )

    window_samples = WINDOW_FRAMES * samples_per_frame
    rows = []
    for name, (reference, generated) in CASES.items():
        row = {"case": name, "reference": reference, "generated": generated}
        row["forward_vs_reference"] = errors(
            whole_forward[name], expected[name], window_samples
        )[0]
        for arm, audio in (
            ("whole_bf16", whole),
            ("main_alone_bf16", main_alone),
            ("main_batched_bf16", main_batched),
        ):
            row[arm], row[f"{arm}_worst_window"] = errors(
                audio[name], expected[name], window_samples
            )
        rows.append(row)
        print(
            f"{name:13s} ref {reference:4d} gen {generated:4d}: whole {row['whole_bf16']:.2e} (worst window "
            f"{row['whole_bf16_worst_window']:.2e}), main alone {row['main_alone_bf16']:.2e} (worst "
            f"{row['main_alone_bf16_worst_window']:.2e}), main batched {row['main_batched_bf16']:.2e} (worst "
            f"{row['main_batched_bf16_worst_window']:.2e}), fp32 forward vs reference {row['forward_vs_reference']:.2e}",
            flush=True,
        )
    print(json.dumps(rows), flush=True)
    print("bf16 whole requests done", flush=True)


if __name__ == "__main__":
    main()
