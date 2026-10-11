from __future__ import annotations

import asyncio
import json
import threading
from queue import Queue

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.scheduling.message import OutgoingMessage, put_messages
from tests.unit_test.pipeline.test_stage_streaming import (
    frame_message,
    make_local_stream_pair,
)
from tests.unit_test.qwen3_tts.test_incremental_codec import Decoder


def check_blocked_consumer() -> None:
    outbox: Queue[OutgoingMessage] = Queue()
    entered_wait = threading.Event()
    received_message = threading.Event()
    original_wait = outbox.not_empty.wait
    received: list[OutgoingMessage] = []

    def waiting(timeout: float | None = None) -> bool:
        entered_wait.set()
        return original_wait(timeout)

    def consume() -> None:
        received.append(outbox.get(timeout=3.0))
        received_message.set()

    outbox.not_empty.wait = waiting
    consumer = threading.Thread(target=consume)
    consumer.start()
    assert entered_wait.wait(1.0)
    message = OutgoingMessage(request_id="blocked", type="stream", data=3)
    put_messages(outbox, [message])
    assert received_message.wait(1.0)
    consumer.join(1.0)
    assert not consumer.is_alive() and received == [message]


async def check_drain_boundary() -> None:
    dispatcher, sender, receivers = make_local_stream_pair()
    sender.active_requests.add("request")
    messages = [frame_message("request", frame) for frame in range(70)]
    messages.append(OutgoingMessage(request_id="request", type="result", data={}))
    put_messages(sender.scheduler.outbox, messages)
    await sender.drain_outbox_external()
    delivered = []
    inbox = receivers["vocoder"].scheduler.inbox
    while not inbox.empty():
        delivered.append(inbox.get_nowait())
    assert [len(batch) for _, batch in dispatcher.chunk_batches] == [64, 6]
    assert [message.data.data for message in delivered[:-1]] == list(range(70))
    assert [message.data.chunk_id for message in delivered[:-1]] == list(range(70))
    assert delivered[-1].type == "stream_done"


async def check_mixed_targets() -> None:
    dispatcher, sender, receivers = make_local_stream_pair(
        targets=("vocoder", "remote")
    )
    sender.same_process_targets.remove("remote")
    sender.active_requests.add("request")
    messages = [
        OutgoingMessage(
            request_id="request",
            type="stream",
            data=torch.tensor([value]),
            target=target,
        )
        for value, target in ((0, "vocoder"), (1, "remote"), (2, "vocoder"))
    ]
    await sender.route_stream_run(messages)
    assert dispatcher.chunk_batches == [
        ("vocoder", [("request", 0)]),
        ("vocoder", [("request", 1)]),
    ]
    inbox = receivers["vocoder"].scheduler.inbox
    assert [inbox.get_nowait().data.data.item() for _ in range(2)] == [0, 2]
    assert len(sender.control_plane.stage_messages) == 1
    target, _, remote_message = sender.control_plane.stage_messages[0]
    assert target == "remote" and remote_message.chunk_id == 0
    assert sender.stream_chunk_counters == {
        ("request", "vocoder"): 2,
        ("request", "remote"): 1,
    }


def check_cuda_slot_reuse() -> None:
    torch.manual_seed(119)
    decoder = Qwen3TTSIncrementalDecoder(Decoder().cuda())
    arena = Qwen3TTSCodecStateArena(
        decoder, num_slots=2, device=torch.device("cuda"), dtype=torch.float32
    )
    storage = arena.storage
    buffers = [
        *storage.conv_histories.values(),
        *storage.transconv_overlaps.values(),
        *storage.transformer_keys.values(),
        *storage.transformer_values.values(),
    ]
    for tensor in buffers:
        tensor.fill_(7)
    storage.frame_positions.fill_(11)
    slot = arena.acquire()
    assert slot == 0
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        for tensor in buffers:
            tensor[slot].fill_(9)
        storage.frame_positions[slot].fill_(17)
        arena.release(slot)
    with torch.cuda.stream(consumer):
        assert arena.acquire() == slot
        complete = torch.cuda.Event()
        complete.record()
    torch.cuda.current_stream().wait_event(complete)
    for tensor in buffers:
        assert torch.count_nonzero(tensor[0]).item() == 0
        assert bool((tensor[1:] == 7).all())
    assert storage.frame_positions.tolist() == [0, 11, 11]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        arena.zero_slot(slot)
    for tensor in buffers:
        tensor[slot].fill_(5)
    storage.frame_positions[slot].fill_(19)
    graph.replay()
    torch.cuda.synchronize()
    for tensor in buffers:
        assert torch.count_nonzero(tensor[0]).item() == 0
        assert bool((tensor[1:] == 7).all())
    assert storage.frame_positions.tolist() == [0, 11, 11]


def run() -> None:
    check_blocked_consumer()
    asyncio.run(check_drain_boundary())
    asyncio.run(check_mixed_targets())
    check_cuda_slot_reuse()
    print(
        json.dumps(
            {
                "blocked_consumer": "pass",
                "drain_boundary": "pass",
                "mixed_targets": "pass",
                "cuda_reuse_and_capture": "pass",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
