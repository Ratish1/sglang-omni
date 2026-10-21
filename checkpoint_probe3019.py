"""Compare trained HiFT readbacks across consecutive graph-pool replays."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftDecodeCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import (
    HiftStepRow,
    create_vocoder_executor,
)
from sglang_omni.platforms import current_platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    label = parser.parse_args().label
    torch.set_num_threads(1)
    torch.manual_seed(3019)
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=4,
        flow_prefix_cache_gb=0.0,
        enable_dit_torch_compile=False,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False,
        enable_hift_cuda_graph=False,
    )
    vocoder = scheduler.vocoder
    device = torch.device("cuda", 0)
    root = Path("/data/review3019")
    stored = {}
    offsets = {}
    patterns = [
        [(120, 50, False), (60, 0, False)],
        [(200, 140, True), (160, 100, False), (30, 0, True)],
        [(120, 50, False), (60, 0, False)],
        [(240, 180, True)],
    ]
    prepared_rows = []
    for pattern in patterns:
        rows = []
        for row, (frames, emitted_frames, is_final) in enumerate(pattern):
            generator = torch.Generator().manual_seed(3019 + frames + row)
            history = (torch.randn(1, 80, frames, generator=generator) * 3).cuda()
            rows.append(
                HiftStepRow(
                    history=history,
                    emitted_samples=emitted_frames * 480,
                    is_final=is_final,
                )
            )
        prepared_rows.append(rows)
    torch.cuda.synchronize()
    with torch.inference_mode(), vocoder.stream_context:
        for graph_mode in (False, True):
            if graph_mode:
                runner = HiftDecodeCudaGraphRunner(
                    vocoder.hift,
                    graphs=SolveGraphCapture(
                        current_platform.get_device_graph_backend(device),
                        device=device,
                        autocast_dtype=torch.bfloat16,
                    ),
                    mel_channels=80,
                    samples_per_frame=480,
                    row_tiers=[1, 2, 4],
                    frame_tiers=[64, 128],
                )
                runner.capture()
                vocoder.hift_decode_cuda_graph_runner = runner
            pending = None
            pending_name = None
            for step, rows in enumerate(prepared_rows):
                current = vocoder.hift_step(rows)
                name = f"graph_{graph_mode}_step_{step}"
                if pending is not None:
                    samples = (
                        pending.wait()
                        if label == "head"
                        else [row[0] for row in pending]
                    )
                    stored[pending_name] = [sample.clone() for sample in samples]
                pending, pending_name = current, name
                offsets[name] = (
                    current.emitted_samples
                    if label == "head"
                    else [row[1] for row in current]
                )
            samples = pending.wait() if label == "head" else [row[0] for row in pending]
            stored[pending_name] = [sample.clone() for sample in samples]
    scheduler.stop()
    torch.save(stored, root / f"{label}.pt")
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": {
            name: [list(sample.shape) for sample in samples]
            for name, samples in stored.items()
        },
        "offsets": offsets,
        "all_finite": all(
            torch.isfinite(sample).all().item()
            for samples in stored.values()
            for sample in samples
        ),
    }
    (root / f"{label}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
