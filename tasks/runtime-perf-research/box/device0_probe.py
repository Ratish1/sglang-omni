"""Which CUDA calls made from a fresh Python thread create a context on device 0 when the process works on device 1.

usage: python3 device0_probe.py    (run with both cards idle; each case runs in its own process)

Per case: a child process sets device 1 on its main thread and allocates there (as a stage process does), then a new
thread runs one call. The child counts compute processes on device 0 through NVML before and after; a new one means
that call created a context on device 0. Prints one line per case: case, created or not, device 0 memory delta.
"""

import subprocess
import sys

CASES = {
    "control_main_only": "pass",
    "thread_stream_context": "with torch.cuda.stream(stream1):\n        pass",
    "thread_stream_context_after_set_device": "torch.cuda.set_device(1)\n    with torch.cuda.stream(stream1):\n        pass",
    "thread_current_stream": "torch.cuda.current_stream()",
    "thread_synchronize": "torch.cuda.synchronize()",
    "thread_alloc_cuda1": "torch.empty(4, device='cuda:1')",
    "thread_copy_to_host": "tensor1.cpu()",
    "thread_new_stream_dev1": "torch.cuda.Stream(device=1)",
    "thread_event_record_on_stream1": "event = torch.cuda.Event()\n    event.record(stream1)\n    event.synchronize()",
    "thread_launch_on_stream1": "with torch.cuda.stream(stream1):\n        (tensor1 + 1).sum()",
}

CHILD = """
import threading, pynvml, torch
pynvml.nvmlInit()
handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
torch.cuda.set_device(1)
tensor1 = torch.ones(1024, device="cuda:1")
stream1 = torch.cuda.Stream(device=1)
torch.cuda.synchronize(1)
before = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle0))
used_before = pynvml.nvmlDeviceGetMemoryInfo(handle0).used
def body():
    {body}
thread = threading.Thread(target=body)
thread.start()
thread.join()
torch.cuda.synchronize(1)
after = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle0))
used_after = pynvml.nvmlDeviceGetMemoryInfo(handle0).used
print(f"{{after > before}} {{(used_after - used_before) / 2**20:.0f}}")
"""


def main() -> None:
    # note (ratish): run under PYTHONPATH=hop_probe OMNI_BIND_THREAD_DEVICE=1 to check the binding what-if on each case
    for name, body in CASES.items():
        result = subprocess.run([sys.executable, "-c", CHILD.format(body=body)], capture_output=True, text=True,
                                timeout=300)
        line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr.strip()[-300:]
        print(f"{name:42s} {line}", flush=True)


if __name__ == "__main__":
    main()
