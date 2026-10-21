"""Exercise the trained HiFT decoder, graph rows and shape-dependent arithmetic."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftDecodeCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import create_vocoder_executor
from sglang_omni.platforms import current_platform


def decode_inputs(batch_size: int, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    mels = []
    sources = []
    for row in range(batch_size):
        generator = torch.Generator().manual_seed(3029 + frames * 10 + row)
        mels.append(torch.randn(1, 80, frames, generator=generator) * 3)
        sources.append(torch.randn(1, 1, frames * 480, generator=generator) * 0.1)
    return torch.cat(mels).cuda(), torch.cat(sources).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parsed = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(3029)
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=16,
        flow_prefix_cache_gb=0.0,
        enable_dit_torch_compile=False,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False,
        enable_hift_cuda_graph=False,
    )
    vocoder = scheduler.vocoder
    assert vocoder.has_hift_kernels
    torch.cuda.synchronize()
    tensors = {}
    records = []
    device = torch.device("cuda", 0)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    with torch.inference_mode(), vocoder.stream_context:
        for allow_tf32 in (False, True):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            for batch_size, frames, finalize in (
                (1, 18, True),
                (1, 64, True),
                (3, 64, True),
                (3, 64, False),
                (1, 100, True),
                (16, 100, True),
                (1, 257, True),
            ):
                mel, source = decode_inputs(batch_size, frames)
                decoded = vocoder.hift.decode(x=mel, s=source, finalize=finalize)
                repeated = vocoder.hift.decode(x=mel, s=source, finalize=finalize)
                assert decoded.shape == repeated.shape
                assert torch.isfinite(decoded).all()
                expected_frames = frames if finalize else frames - 5
                assert decoded.shape == (batch_size, expected_frames * 480)
                name = f"tf32_{allow_tf32}_b{batch_size}_f{frames}_final_{finalize}"
                tensors[name] = decoded.cpu()
                records.append(
                    {
                        "name": name,
                        "shape": list(decoded.shape),
                        "same_call_bit_identical": torch.equal(decoded, repeated),
                    }
                )
            runner = HiftDecodeCudaGraphRunner(
                vocoder.hift,
                graphs=SolveGraphCapture(
                    backend, device=device, autocast_dtype=torch.bfloat16
                ),
                mel_channels=80,
                samples_per_frame=480,
                row_tiers=[1, 4, 16],
                frame_tiers=[64, 100, 128],
            )
            runner.capture()
            first_replay = None
            for batch_size, frames in ((3, 64), (16, 100), (1, 128), (3, 64)):
                mel, source = decode_inputs(batch_size, frames)
                replayed = runner.run(mel, source).clone()
                own_shape = vocoder.hift.decode(x=mel, s=source, finalize=True)
                name = f"tf32_{allow_tf32}_graph_b{batch_size}_f{frames}"
                tensors[name] = replayed.cpu()
                records.append(
                    {
                        "name": name,
                        "graph_eager_max_abs_error": float(
                            (replayed - own_shape).abs().max()
                        ),
                    }
                )
                if (batch_size, frames) == (3, 64):
                    if first_replay is None:
                        first_replay = replayed.clone()
                    else:
                        assert torch.equal(first_replay, replayed)
                else:
                    pass
            for batch_size in (1, 16):
                mel, source = decode_inputs(batch_size, 100)
                for _ in range(5):
                    runner.run(mel, source)
                latencies = []
                for _ in range(5):
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(20):
                        runner.run(mel, source)
                    end.record()
                    end.synchronize()
                    latencies.append(start.elapsed_time(end) / 20)
                records.append(
                    {
                        "name": f"tf32_{allow_tf32}_graph_b{batch_size}_f100_latency_ms",
                        "measurements": latencies,
                    }
                )
    scheduler.stop()
    root = Path("/data/review3029")
    torch.save(tensors, root / f"{parsed.label}.pt")
    summary = {
        "label": parsed.label,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "production_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD:sglang_omni"], text=True
        ).strip(),
        "decode_type": type(vocoder.hift.decode).__name__,
        "cases": records,
    }
    (root / f"{parsed.label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
