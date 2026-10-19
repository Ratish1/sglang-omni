"""Exercise codec-slot reuse after an in-flight abort and a cohort failure."""

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    CapturedGraph,
    GraphKey,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload
from tests.unit_test.qwen3_omni.test_code2wav import (
    decode_in_parts,
    make_fused_tiny_code2wav,
    make_incremental_scheduler,
    run_streams,
)


def main() -> None:
    torch.manual_seed(2909)
    model = make_fused_tiny_code2wav()
    scheduler = make_incremental_scheduler(model, codec_state_slots=2, cuda_graph=True)
    codes = torch.randint(0, 16, (2, 2, 9), device="cuda")
    for request_id in ("abort-0", "abort-1"):
        scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    for position in (0, 3):
        for request_id, stream_item in [
            (
                f"abort-{row}",
                StreamItem(
                    position,
                    row_codes[:, position : position + 3].T,
                    "talker",
                    metadata={"stream": False},
                ),
            )
            for row, row_codes in enumerate(codes)
        ]:
            scheduler.handle_stream_chunk(request_id, stream_item)
        if position == 3:
            torch.cuda._sleep(500_000_000)
        else:
            pass
        scheduler.run_ready_step()
    assert not scheduler.stream_states["abort-0"].pending.slot.query()
    scheduler.abort("abort-0")
    scheduler.abort("abort-1")
    assert scheduler.codec_pool.available_size() == 2
    waveforms = run_streams(scheduler, codes, first_request=10)
    with torch.inference_mode():
        expected = decode_in_parts(Qwen3OmniIncrementalCodec(model), codes, (3, 3, 3))
    for actual, wanted in zip(waveforms, expected.float().cpu().reshape(2, -1)):
        np.testing.assert_allclose(actual, wanted.numpy(), atol=0.002, rtol=0.02)
    print("in-flight cohort abort and immediate codec-slot reuse: passed", flush=True)
    for request_id in ("failure-0", "failure-1"):
        scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    for request_id, stream_item in [
        (
            f"failure-{row}",
            StreamItem(0, row_codes[:, :3].T, "talker", metadata={"stream": False}),
        )
        for row, row_codes in enumerate(codes)
    ]:
        scheduler.handle_stream_chunk(request_id, stream_item)
    original_replay = scheduler.cuda_graph_runner.replay

    def fail_after_state_write(
        captured: CapturedGraph,
        key: GraphKey,
        input_codes: torch.Tensor,
        *,
        slots: list[int] | None,
    ) -> None:
        original_replay(captured, key, input_codes, slots=slots)
        raise RuntimeError("review: failure after graph state write")

    scheduler.cuda_graph_runner.replay = fail_after_state_write
    scheduler.run_ready_step()
    scheduler.cuda_graph_runner.replay = original_replay
    errors = [scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()]
    assert all(message.type == "error" for message in errors)
    assert {message.request_id for message in errors} == {"failure-0", "failure-1"}
    assert not scheduler.stream_states
    assert scheduler.codec_pool.available_size() == 2
    run_streams(scheduler, codes, first_request=20)
    scheduler.stop()
    assert scheduler.codec_pool.available_size() == 2
    assert not scheduler.retired_chunks
    assert not scheduler.pinned_retired
    print(
        "failure after real state writes aborts the cohort and returns every slot: passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
