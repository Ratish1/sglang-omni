import asyncio
import glob

import torch

import sglang_omni
from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.proto import StagePayload
from sglang_omni.relay.shm import ShmRelay

print("imported", sglang_omni.__file__)


async def round_trip(name: str, data: dict[str, torch.Tensor]) -> None:
    relay = ShmRelay(engine_id="e", device="cpu")
    before = set(glob.glob("/dev/shm/psm_*"))
    payload = StagePayload(request_id="r0", request=None, data=data)
    try:
        data_ref, put_operation = await stage_io.write_payload(
            relay, "r0", payload, transport=TransportKind.SHM
        )
    except Exception as error:
        leaked = set(glob.glob("/dev/shm/psm_*")) - before
        print(
            f"{name}: put raised {type(error).__name__}: {error}; leaked blocks {len(leaked)}; free credits {relay.sem._value}"
        )
        for path in leaked:
            import os

            os.unlink(path)
        return
    restored = await stage_io.read_payload(relay, "r0", data_ref)
    equal = all(
        (
            (
                torch.equal(restored.data[key], value)
                and restored.data[key].dtype == value.dtype
            )
            if isinstance(value, torch.Tensor)
            else restored.data[key] == value
        )
        for key, value in data.items()
    )
    put_operation.mark_receiver_done()
    await put_operation.wait_for_completion(timeout=1.0)
    print(
        f"{name}: size {data_ref.buffer.length} equal {equal} leaked {len(set(glob.glob('/dev/shm/psm_*')) - before)}"
    )


async def main() -> None:
    await round_trip("one empty float32", {"x": torch.empty(0)})
    await round_trip(
        "two empty tensors",
        {"x": torch.empty(0), "y": torch.empty(0, 3, dtype=torch.int64)},
    )
    await round_trip(
        "empty then a byte",
        {"x": torch.empty(0), "y": torch.ones(1, dtype=torch.uint8)},
    )
    await round_trip("no tensors", {"text": "hello"})


asyncio.run(main())
