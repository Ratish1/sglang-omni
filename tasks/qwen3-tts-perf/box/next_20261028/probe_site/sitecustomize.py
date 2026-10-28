"""Block 6 probe, loaded into every Python process of a server through PYTHONPATH.

With OMNI_PROBE_CUDNN_SDP_OFF=1 the process runs with torch's cuDNN SDPA backend off. With
OMNI_PROBE_MEMORY_DIR set, a daemon thread appends, every second once CUDA is initialized, the
process's PyTorch reserved and allocated bytes, its allocator retries and OOMs, and the device's
free bytes, one line per device the process holds memory on.
"""

import multiprocessing
import os
import sys
import threading
import time

if os.environ.get("OMNI_PROBE_CUDNN_SDP_OFF") == "1":
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
else:
    pass


def log_memory(directory: str) -> None:
    path = os.path.join(directory, f"torchmem_{os.getpid()}.log")
    is_named = False
    while True:
        time.sleep(1.0)
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            continue
        else:
            pass
        lines = []
        for device in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(device)
            reserved_bytes = int(stats.get("reserved_bytes.all.current", 0))
            if reserved_bytes == 0:
                continue
            else:
                pass
            free_bytes, _ = torch.cuda.mem_get_info(device)
            lines.append(
                f"{time.time():.1f} {device} reserved {reserved_bytes} "
                f"allocated {int(stats.get('allocated_bytes.all.current', 0))} "
                f"peak_reserved {int(stats.get('reserved_bytes.all.peak', 0))} "
                f"retries {int(stats.get('num_alloc_retries', 0))} "
                f"ooms {int(stats.get('num_ooms', 0))} free {free_bytes}\n"
            )
        if lines:
            with open(path, "a") as handle:
                if not is_named:
                    handle.write(
                        f"# pid {os.getpid()} process {multiprocessing.current_process().name} "
                        f"argv {' '.join(sys.argv)[:200]}\n"
                    )
                    is_named = True
                else:
                    pass
                handle.writelines(lines)
        else:
            pass


if os.environ.get("OMNI_PROBE_MEMORY_DIR"):
    threading.Thread(
        target=log_memory,
        args=(os.environ["OMNI_PROBE_MEMORY_DIR"],),
        name="omni-probe-memory",
        daemon=True,
    ).start()
else:
    pass
