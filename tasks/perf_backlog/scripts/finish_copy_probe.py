import time

import torch

device = torch.device("cuda")
stream = torch.cuda.current_stream(device)
ref = torch.arange(6, device=device).reshape(2, 3)
codes = [torch.full((3,), 100 + i, dtype=torch.long, device=device) for i in range(5)]
torch.empty((7, 3), dtype=torch.long, pin_memory=True)
torch.stack(codes, 0)
torch.cuda.synchronize()


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    ms = 1e3 * (time.perf_counter() - t0)
    print(f"{label:14s} {ms:8.2f} ms   stream still busy: {not stream.query()}")
    return out


torch.cuda._sleep(1_000_000_000)
print("after sleep enqueue, stream busy:", not stream.query())
stacked = timed("stack", lambda: torch.stack(codes, dim=0))
cat = timed("cat", lambda: torch.cat([ref, stacked], dim=0))
host = timed(
    "pinned empty", lambda: torch.empty(cat.shape, dtype=torch.long, pin_memory=True)
)
timed("copy_ D2H", lambda: host.copy_(cat, non_blocking=True))
event = timed("Event()", lambda: torch.cuda.Event())
timed("record", lambda: event.record())
print("event done:", event.query(), " stream busy:", not stream.query())
torch.cuda.synchronize()
print("values", host.flatten().tolist()[:6])
