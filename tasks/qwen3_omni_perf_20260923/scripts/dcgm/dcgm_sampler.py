"""Host-side DCGM sampler: every sample of the watched profiling fields, with DCGM's own sample
timestamp (microseconds since the epoch), one TSV row per value, until SIGTERM or --seconds.

DCGM runs as root on the node and reads the counters the driver keeps from users
(RmProfilingAdminOnly: 1), so this works where nsys GPU metrics and ncu are refused. SM issue
has no DCGM field; the pipe fields (fp32, fp16, integer, tensor) are its nearest proxies. DCGM GPU
ids equal the container's card indices on the H100 host (checked by UUID); the H200 node runs
no host engine.

Run on the host with the system python3 (the bindings live in /usr/local/dcgm).

usage: python3 dcgm_sampler.py --gpus 0 1 --interval-ms 10 --out samples.tsv [--seconds S]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

sys.path.append("/usr/local/dcgm/bindings/python3")
from DcgmReader import DcgmReader  # noqa: E402

FIELDS = {
    1001: "gr_active",
    1002: "sm_active",
    1003: "sm_occupancy",
    1004: "tensor_active",
    1005: "dram_active",
    1007: "fp32_active",
    1008: "fp16_active",
    1016: "integer_active",
}
POLL_SECONDS = 0.2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--interval-ms", type=float, default=10.0)
    parser.add_argument("--fields", type=int, nargs="+", default=list(FIELDS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=0.0)
    args = parser.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stop.append(signum))
    reader = DcgmReader(
        fieldIds=args.fields,
        updateFrequency=int(args.interval_ms * 1000),
        maxKeepAge=60.0,
        gpuIds=args.gpus,
    )
    deadline = time.time() + args.seconds if args.seconds > 0 else float("inf")
    rows = 0
    with open(args.out, "w") as out:
        out.write("ts_us\tgpu\tfield\tvalue\n")
        # the first call arms the watch and returns nothing
        reader.GetAllGpuValuesAsFieldIdDictSinceLastCall()
        while not stop and time.time() < deadline:
            time.sleep(POLL_SECONDS)
            batch = reader.GetAllGpuValuesAsFieldIdDictSinceLastCall()
            for gpu, fields in batch.items():
                for field, values in fields.items():
                    name = FIELDS.get(field, str(field))
                    for value in values:
                        out.write(f"{value.ts}\t{gpu}\t{name}\t{value.value}\n")
                        rows += 1
            out.flush()
    reader.Shutdown()
    print(f"{rows} values written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
