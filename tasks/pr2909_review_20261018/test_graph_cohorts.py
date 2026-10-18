"""Guard slot ownership across batched graph replays and a ring wrap."""

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload
from tests.unit_test.qwen3_omni.test_code2wav import (
    decode_in_parts,
    make_fused_tiny_code2wav,
    make_incremental_scheduler,
)


def test_two_slots_keep_their_streams_across_batched_next_graphs() -> None:
    torch.manual_seed(2909)
    model = make_fused_tiny_code2wav()
    scheduler = make_incremental_scheduler(model, codec_state_slots=2, cuda_graph=True)
    codes = torch.randint(0, 16, (2, 2, 11), device="cuda")
    request_ids = ["cohort-0", "cohort-1"]
    for request_id in request_ids:
        scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    for position in (0, 3, 6):
        scheduler.on_stream_chunk_batch(
            [
                (
                    request_id,
                    StreamItem(
                        position,
                        row_codes[:, position : position + 3].T,
                        "talker",
                        metadata={"stream": False},
                    ),
                )
                for request_id, row_codes in zip(request_ids, codes)
            ]
        )
        assert len(scheduler.ready_window_group()) == 2
        scheduler.run_ready_step()
    for request_id, row_codes in zip(request_ids, codes):
        scheduler.handle_stream_chunk(
            request_id,
            StreamItem(9, row_codes[:, 9:].T, "talker", metadata={"stream": False}),
        )
        scheduler.handle_stream_done(request_id)
    with torch.inference_mode():
        expected = decode_in_parts(
            Qwen3OmniIncrementalCodec(model), codes, (3, 3, 3, 2)
        )
    for waveform in expected.float().cpu().reshape(2, -1):
        message = scheduler.outbox.get_nowait()
        assert message.type == "result"
        actual = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
        np.testing.assert_allclose(actual, waveform.numpy(), atol=0.002, rtol=0.02)
    assert scheduler.codec_pool.available_size() == 2
    assert scheduler.cuda_graph_runner.stats()["runtime"]["fallback_counts"] == {}
