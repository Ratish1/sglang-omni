"""Review probes for trained HiFT numerics, tier reuse and decode costs."""

import copy
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftDecodeCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture, row_tiers
from sglang_omni.platforms import current_platform
from tests.unit_test.fun_cosyvoice3.test_vocoder import HiftFlowStub, make_causal_hift


def difference(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    error = (actual.double() - reference.double()).square().mean()
    signal = reference.double().square().mean()
    return {
        "bit_equal": torch.equal(actual, reference),
        "max_abs": (actual - reference).abs().max().item(),
        "rms_error": error.sqrt().item(),
        "snr_db": (10 * torch.log10(signal / error.clamp_min(1e-30))).item(),
    }


def timing(hift: torch.nn.Module, runner: HiftDecodeCudaGraphRunner,
           mel: torch.Tensor, source: torch.Tensor) -> list[dict[str, float | str]]:
    measurements = []
    for name in ("stock_eager", "patched_eager", "graph_with_copies"):
        for iteration in range(8):
            if name == "graph_with_copies":
                runner.run(mel, source)
            else:
                hift.decode(x=mel, s=source, finalize=True)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        for iteration in range(30):
            if name == "graph_with_copies":
                runner.run(mel, source)
            else:
                hift.decode(x=mel, s=source, finalize=True)
        end.record()
        end.synchronize()
        measurements.append({"arm": name, "wall_ms": (time.perf_counter() - started) * 1000 / 30,
                             "stream_ms": begin.elapsed_time(end) / 30})
        if name == "stock_eager":
            stages.patch_hift_snake(hift)
            stages.patch_hift_spectra(hift, mel.device)
        else:
            pass
    return measurements


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    raw = make_causal_hift(10.0)
    checkpoint = hf_hub_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "hift.pt")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw.load_state_dict({name.removeprefix("generator."): value for name, value in weights.items()}, strict=True)
    patched = copy.deepcopy(raw)
    vocoder = stages.CosyVoice3Vocoder(HiftFlowStub(), patched)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    graphs = SolveGraphCapture(backend, device=device, autocast_dtype=torch.bfloat16)
    tiers = [frames for frames in generate_prefill_cuda_graph_batch_sizes(8000) if 50 <= frames <= 256]
    runner = HiftDecodeCudaGraphRunner(patched, graphs=graphs, mel_channels=80,
                                     samples_per_frame=480, row_tiers=row_tiers(16), frame_tiers=tiers)
    before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    runner.capture()
    report = {"checkpoint": checkpoint, "device": torch.cuda.get_device_name(),
              "tiers": tiers, "graphs": len(runner.captured),
              "capture_seconds": time.perf_counter() - started,
              "capture_allocated_bytes": torch.cuda.memory_allocated() - before,
              "defaults": {"cudnn_tf32": torch.backends.cudnn.allow_tf32,
                           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32}}
    torch.manual_seed(2910)
    report["decode_numerics"] = []
    for rows, frames in ((1, 50), (3, 64), (16, 128), (1, 257)):
        mel = torch.randn(rows, 80, frames, device=device) * 2 - 4
        f0 = raw.f0_predictor(mel.double(), finalize=True).to(mel)
        excitation = raw.f0_upsamp(f0[:, None]).transpose(1, 2)
        excitation, _, _ = raw.m_source(excitation)
        source = excitation.transpose(1, 2).contiguous()
        reference = raw.decode(x=mel, s=source, finalize=True)
        actual = patched.decode(x=mel, s=source, finalize=True)
        assert torch.isfinite(actual).all()
        report["decode_numerics"].append({"rows": rows, "frames": frames,
                                         **difference(actual, reference)})
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    report["reuse_checks"] = []
    for rows, frames in ((16, 256), (3, 64), (1, 80), (8, 128), (2, 256), (1, 64), (16, 80), (3, 64)):
        mel = torch.randn(rows, 80, frames, device=device) * 2 - 4
        source = torch.randn(rows, 1, frames * 480, device=device) * 0.1
        actual = runner.run(mel, source).clone()
        reference = patched.decode(x=mel, s=source, finalize=True)
        metrics = difference(actual, reference)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        report["reuse_checks"].append({"rows": rows, "frames": frames, **metrics})
    report["window_checks"] = []
    for configuration in (((50, 0, False), (150, 42, False)),
                          ((350, 142, False), (90, 42, True), (18, 0, True)),
                          ((410, 342, True), (257, 0, True)),
                          ((700, 42, False),), ((50, 0, False),)):
        rows = [stages.HiftStepRow(history=torch.randn(1, 80, frames, device=device) * 2 - 4,
                                  emitted_samples=emitted * 480, is_final=final)
                for frames, emitted, final in configuration]
        vocoder.hift_decode_cuda_graph_runner = None
        expected = vocoder.hift_step(rows)
        vocoder.hift_decode_cuda_graph_runner = runner
        actual = vocoder.hift_step(rows)
        for (audio, cursor), (reference, expected_cursor) in zip(actual, expected, strict=True):
            assert cursor == expected_cursor
            torch.testing.assert_close(audio, reference, rtol=0, atol=0)
        report["window_checks"].append({"configuration": configuration, "bit_equal": True})
    torch.backends.cudnn.allow_tf32 = report["defaults"]["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = report["defaults"]["matmul_tf32"]
    report["decode_timing"] = []
    for rows in (1, 16):
        fresh = copy.deepcopy(raw)
        mel = torch.randn(rows, 80, 64, device=device) * 2 - 4
        source = torch.randn(rows, 1, 64 * 480, device=device) * 0.1
        report["decode_timing"].append({"rows": rows, "frames": 64,
                                        "arms": timing(fresh, runner, mel, source)})
    destination = Path("/data/review2910/probe.json")
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
