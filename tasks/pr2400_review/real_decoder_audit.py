import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.qwen3_tts.stages import create_vocoder_executor
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
)
from sglang_omni.profiler.event_recorder import get_recorder


def run_schedule(
    scheduler: Qwen3TTSStreamingVocoderScheduler,
    codes: list[torch.Tensor],
    slack_milliseconds: int,
    output_directory: Path,
) -> dict[str, np.ndarray]:
    scheduler.followup_urgent_slack_s = slack_milliseconds / 1000
    scheduler.initial_worker = threading.current_thread()
    scheduler.followup_worker = threading.current_thread()
    scheduler.worker_ctx.stream = scheduler.followup_decode_streams[0]
    scheduler.worker_ctx.incremental_graphs = (
        scheduler.followup_incremental_graph_holders[0]
    )
    scheduler.event_stage_name = "vocoder"
    reference_lengths = (0, 5, 2, 0)
    preparation_counts = (2, 1, 0, 0)
    streams = []
    recorder = get_recorder()
    recorder.start(str(output_directory.name), str(output_directory), "vocoder")
    with torch.cuda.stream(scheduler.decode_stream):
        for index, (frames, reference_length) in enumerate(
            zip(codes, reference_lengths)
        ):
            request_id = f"request-{index}"
            state = scheduler.create_stream_state(request_id)
            scheduler.stream_states[request_id] = state
            scheduler.latch_stream_contract(
                request_id,
                state,
                {"num_quantizers": frames.shape[1], "ref_code_len": reference_length},
                origin="stream metadata",
            )
            scheduler.ingest(
                request_id, state, scheduler.validate_chunk(request_id, state, frames)
            )
            state.initial_pending = True
            scheduler.run_initial_batch([(request_id, state)])
            streams.append((request_id, state))
    with torch.cuda.stream(scheduler.worker_ctx.stream):
        for stream, preparation_count in zip(streams, preparation_counts):
            for _ in range(preparation_count):
                scheduler.run_followup_batch([stream])
                scheduler.drain_pending_incremental(keep=0)
        while not scheduler.followup_queue.empty():
            scheduler.followup_queue.get_nowait()
        for index, (_, state) in enumerate(streams):
            state.playback_deadline_s = time.monotonic() + (
                0 if index == 3 else 10 + index
            )
        scheduler.run_followup_batch(streams)
        scheduler.drain_pending_incremental(keep=0)
        for _, state in streams:
            state.final_pending = True
        while True:
            remaining = [
                stream
                for stream in streams
                if stream[1].emitted_generated_frames
                < stream[1].total_frames - stream[1].ref_frames
            ]
            if not remaining:
                break
            else:
                pass
            scheduler.run_followup_batch(remaining)
            scheduler.drain_pending_incremental(keep=0)
    recorder.stop()
    chunks: dict[str, list[np.ndarray]] = {request_id: [] for request_id, _ in streams}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        assert message.type == "stream", message
        chunks[message.request_id].append(
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32).copy()
        )
    audio = {request_id: np.concatenate(parts) for request_id, parts in chunks.items()}
    for request_id, state in streams:
        assert (
            len(audio[request_id])
            == (state.total_frames - state.ref_frames) * scheduler.samples_per_frame
        )
        assert np.isfinite(audio[request_id]).all()
        assert state.codec_frame_position == state.total_frames
        scheduler.clear_stream_state(request_id)
    while not scheduler.followup_queue.empty():
        scheduler.followup_queue.get_nowait()
    assert scheduler.pending_incremental() == []
    assert scheduler.codec_slots_in_flight == set()
    assert scheduler.codec_arena.active_slots() == 0
    np.savez(output_directory / "pcm.npz", **audio)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    scheduler = create_vocoder_executor(arguments.checkpoint, device="cuda:0")
    generator = torch.Generator().manual_seed(731)
    codes = [
        torch.randint(
            2048, (length, scheduler.decoder.config.num_quantizers), generator=generator
        )
        for length in (18, 17, 11, 21)
    ]
    runs = []
    with torch.inference_mode():
        for index, slack in enumerate((0, 30, 30, 0)):
            directory = arguments.output / f"run-{index}-slack-{slack}"
            directory.mkdir(parents=True, exist_ok=True)
            runs.append(run_schedule(scheduler, codes, slack, directory))
    comparisons = []
    for index, audio in enumerate(runs[1:], 1):
        for request_id, reference in runs[0].items():
            difference = audio[request_id].astype(np.float64) - reference.astype(
                np.float64
            )
            comparisons.append(
                {
                    "run": index,
                    "request_id": request_id,
                    "samples": len(reference),
                    "exact": bool(np.array_equal(reference, audio[request_id])),
                    "max_abs": float(np.abs(difference).max()),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                }
            )
    summary = {"comparisons": comparisons, "codec_stats": scheduler.codec_state_stats()}
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
