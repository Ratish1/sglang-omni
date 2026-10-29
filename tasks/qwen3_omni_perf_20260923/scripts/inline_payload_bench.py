"""A stage payload sent over the SHM relay against the same payload pickled into its control message.

Two processes, the tree's real stage_io, ShmRelay and control-plane sockets (ZMQ PUSH and PULL
over ipc://, msgpack messages), one payload in flight at a time, as a stage hop runs:

- shm: the sender runs write_payload and sends the DataReadyMessage; the receiver reads the
  payload with read_payload and sends the DataAckMessage; the sender completes the put on it.
- inline: the sender pickles the StagePayload into the DataReadyMessage; the receiver unpickles it.

For every payload size (one float32 tensor plus a small request dict) and path it prints the
median and p90 of the sender's thread CPU, the receiver's thread CPU and the delivery time (sender
start to the receiver holding the payload, wall clocks of one host), over ITERATIONS hops after
WARMUP, the two paths alternating hop by hop.

usage: python inline_payload_bench.py OUT_DIR [ITERATIONS] [WARMUP]
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import pickle
import statistics
import sys
import time

import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.pipeline.control_plane import PullSocket, PushSocket
from sglang_omni.proto import DataReadyMessage, StagePayload
from sglang_omni.proto.messages import DataAckMessage
from sglang_omni.relay.shm import ShmRelay

SIZES = (8, 64, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216)
PATHS = ("shm", "inline")


def make_payload(size_bytes: int, hop: int) -> StagePayload:
    return StagePayload(
        request_id=f"r{hop}",
        request={"prompt_tokens": 128, "stage": "bench"},
        data={"features": torch.randn(max(size_bytes // 4, 1), dtype=torch.float32)},
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


async def sender_main(out_dir: str, iterations: int, warmup: int) -> None:
    relay = ShmRelay(engine_id="sender", device="cpu", credits=2)
    data_socket = PushSocket(f"ipc://{out_dir}/data.sock")
    await data_socket.connect()
    ack_socket = PullSocket(f"ipc://{out_dir}/ack.sock", bind=True)
    await ack_socket.start()
    await asyncio.sleep(0.5)
    records: list[dict[str, object]] = []
    for size_bytes in SIZES:
        for hop in range(warmup + iterations):
            for path in PATHS:
                payload = make_payload(size_bytes, hop)
                start_ns = time.time_ns()
                cpu_start = time.thread_time_ns()
                if path == "shm":
                    data_ref, put_operation = await stage_io.write_payload(
                        relay, payload.request_id, payload, transport=TransportKind.SHM
                    )
                    reference = data_ref.to_dict()
                else:
                    put_operation = None
                    reference = {
                        "_type": "InlinePayload",
                        "version": 1,
                        "payload": pickle.dumps(payload),
                    }
                await data_socket.send(
                    DataReadyMessage(
                        request_id=payload.request_id,
                        from_stage="sender",
                        to_stage="receiver",
                        data_ref=reference,
                    )
                )
                send_cpu_ns = time.thread_time_ns() - cpu_start
                ack_cpu_start = time.thread_time_ns()
                ack = await ack_socket.recv()
                if put_operation is not None:
                    put_operation.mark_receiver_done()
                    await put_operation.wait_for_completion(timeout=10.0)
                    send_cpu_ns += time.thread_time_ns() - ack_cpu_start
                else:
                    pass
                if hop >= warmup:
                    records.append(
                        {
                            "size": size_bytes,
                            "path": path,
                            "start_ns": start_ns,
                            "sender_cpu_us": send_cpu_ns / 1000,
                            "ack_object_id": ack.object_id,
                        }
                    )
                else:
                    pass
    await data_socket.send(
        DataReadyMessage(
            request_id="stop",
            from_stage="sender",
            to_stage="receiver",
            data_ref={"_type": "Stop"},
        )
    )
    with open(f"{out_dir}/sender.json", "w") as sender_file:
        json.dump(records, sender_file)


async def receiver_main(out_dir: str) -> None:
    relay = ShmRelay(engine_id="receiver", device="cpu")
    data_socket = PullSocket(f"ipc://{out_dir}/data.sock", bind=True)
    await data_socket.start()
    ack_socket = PushSocket(f"ipc://{out_dir}/ack.sock")
    await ack_socket.connect()
    records: list[dict[str, object]] = []
    while True:
        message = await data_socket.recv()
        cpu_start = time.thread_time_ns()
        reference = message.data_ref
        if reference["_type"] == "Stop":
            break
        elif reference["_type"] == "InlinePayload":
            payload = pickle.loads(reference["payload"])
            path = "inline"
        else:
            data_ref = DataRef.from_dict(reference)
            payload = await stage_io.read_payload(relay, message.request_id, data_ref)
            path = "shm"
        ready_ns = time.time_ns()
        receive_cpu_ns = time.thread_time_ns() - cpu_start
        ack_cpu_start = time.thread_time_ns()
        await ack_socket.send(
            DataAckMessage(
                request_id=message.request_id,
                from_stage="receiver",
                to_stage="sender",
                object_id=f"{path}:{message.request_id}",
            )
        )
        if path == "shm":
            receive_cpu_ns += time.thread_time_ns() - ack_cpu_start
        else:
            pass
        records.append(
            {
                "path": path,
                "bytes": int(payload.data["features"].numel() * 4),
                "ready_ns": ready_ns,
                "receiver_cpu_us": receive_cpu_ns / 1000,
            }
        )
        del payload
    with open(f"{out_dir}/receiver.json", "w") as receiver_file:
        json.dump(records, receiver_file)


def run_receiver(out_dir: str) -> None:
    torch.set_num_threads(8)
    asyncio.run(receiver_main(out_dir))


def main() -> None:
    out_dir = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(8)
    receiver = multiprocessing.get_context("spawn").Process(
        target=run_receiver, args=(out_dir,)
    )
    receiver.start()
    asyncio.run(sender_main(out_dir, iterations, warmup))
    receiver.join()
    sent = json.load(open(f"{out_dir}/sender.json"))
    received = [
        record
        for record in json.load(open(f"{out_dir}/receiver.json"))
        if record["path"] in PATHS
    ]
    warm = len(received) - len(sent)
    received = received[warm:] if warm > 0 else received
    print(f"torch {torch.__version__}, {iterations} hops a size and path after {warmup}")
    print(
        f"{'bytes':>10} {'path':>7}  {'sender cpu us':>20}  {'receiver cpu us':>20}  {'delivery us':>20}"
    )
    rows: dict[tuple[int, str], dict[str, list[float]]] = {}
    for sent_record, received_record in zip(sent, received):
        if sent_record["path"] != received_record["path"]:
            raise RuntimeError("sender and receiver records are out of step")
        else:
            pass
        key = (sent_record["size"], sent_record["path"])
        row = rows.setdefault(key, {"sender": [], "receiver": [], "delivery": []})
        row["sender"].append(sent_record["sender_cpu_us"])
        row["receiver"].append(received_record["receiver_cpu_us"])
        row["delivery"].append(
            (received_record["ready_ns"] - sent_record["start_ns"]) / 1000
        )
    for (size_bytes, path), row in sorted(rows.items()):
        cells = [
            f"{statistics.median(row[name]):9.1f} / {percentile(row[name], 0.9):8.1f}"
            for name in ("sender", "receiver", "delivery")
        ]
        print(f"{size_bytes:>10} {path:>7}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
