"""cuDNN's cost per frame of the DiT's causal conv as a function of width.

The layout benches found the same grouped conv (1024 channels, 16 groups,
kernel 31, bfloat16) costing 0.126 microseconds per frame at width 680 and 0.19
at most other widths. This sweeps the width at fixed batch sizes to see whether
the fast widths follow a rule (a multiple, a threshold) a tile width could be
rounded to. The first conv of the real module, inference mode, autocast.

  python conv_width_sweep.py --model .../snapshots/master
"""

from __future__ import annotations

import argparse
import json

import torch

from sglang_omni.models.fun_cosyvoice3.stages import load_cosyvoice3_flow_hift

ITERATIONS = 20
BATCHES = (2, 8, 32)
WIDTHS = tuple(range(320, 1441, 4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    flow, _ = load_cosyvoice3_flow_hift(args.model, device="cuda:0")
    conv = flow.decoder.estimator.input_embed.conv_pos_embed.conv1

    report = {}
    for batch in BATCHES:
        points = []
        for width in WIDTHS:
            x = torch.randn(batch, 1024, width, device="cuda", dtype=torch.bfloat16)

            def run():
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    return conv(x)

            for _ in range(5):
                run()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERATIONS):
                run()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / ITERATIONS
            points.append(
                {
                    "width": width,
                    "ms": round(ms, 4),
                    "ns_per_frame": round(ms * 1e6 / (batch * width), 2),
                }
            )
        report[str(batch)] = points
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cudnn": torch.backends.cudnn.version(),
                "batches": report,
            }
        )
    )


if __name__ == "__main__":
    main()
