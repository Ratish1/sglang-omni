"""Measure graph captures that exceed a one-stream codec pool's capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    create_code2wav_scheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(8)
    start = time.perf_counter()
    scheduler = create_code2wav_scheduler(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device="cuda:0",
        dtype="bfloat16",
        codec_state_slots=1,
        max_replay_rows=arguments.rows,
        enable_cuda_graph=True,
        total_gpu_memory_fraction=0.014,
    )
    elapsed = time.perf_counter() - start
    pool = scheduler.codec_pool
    runner = scheduler.cuda_graph_runner
    assert runner.enabled
    unreachable = [
        key
        for key in runner.graphs
        if key.decode != "window" and key.batch_size > pool.num_slots
    ]
    scheduler.stream_payloads["only"] = make_qwen_payload(request_id="only")
    torch.manual_seed(2909)
    codes = torch.randint(
        scheduler.model.config.codebook_size,
        (scheduler.model.config.num_quantizers, 20),
        device="cuda",
    )
    for position in (0, 10):
        scheduler.handle_stream_chunk(
            "only",
            StreamItem(
                position,
                codes[:, position : position + 10].T,
                "talker",
                metadata={"stream": False},
            ),
        )
        assert len(scheduler.ready_window_group()) == 1
        scheduler.run_ready_step()
    scheduler.handle_stream_done("only")
    message = scheduler.outbox.get_nowait()
    assert message.type == "result"
    assert pool.available_size() == 1
    statistics = {
        "max_replay_rows": arguments.rows,
        "stream_slots": pool.num_slots,
        "capture_slots": len(pool.scratch_slots),
        "unreachable_stateful_graphs": len(unreachable),
        "published_graphs": len(runner.graphs),
        "pool_bytes": pool.bytes_per_slot * (pool.num_slots + len(pool.scratch_slots)),
        "graph_footprint_bytes": runner.stats()["memory"]["graph_footprint_bytes"],
        "factory_seconds": elapsed,
        "served_audio_bytes": len(message.data.data["audio_waveform"]),
        "served_audio_sha256": hashlib.sha256(
            message.data.data["audio_waveform"]
        ).hexdigest(),
        "runtime": runner.stats()["runtime"],
    }
    Path(f"/data/review2909/capacity-{arguments.rows}.json").write_text(
        json.dumps(statistics, indent=2)
    )
    print(json.dumps(statistics), flush=True)
    scheduler.stop()


if __name__ == "__main__":
    main()
