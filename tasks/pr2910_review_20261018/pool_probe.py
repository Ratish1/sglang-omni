"""Alternate real Flow and HiFT runners in one shared graph pool."""

import json

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftDecodeCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.whole_history_cuda_graph import (
    WholeHistoryCudaGraphRunner,
)
from sglang_omni.platforms import current_platform
from tests.unit_test.fun_cosyvoice3.test_prefix_cache_gpu import make_estimator
from tests.unit_test.fun_cosyvoice3.test_vocoder import HiftFlowStub, make_causal_hift


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2910)
    device = torch.device("cuda", torch.cuda.current_device())
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    graphs = SolveGraphCapture(backend, device=device, autocast_dtype=torch.bfloat16)
    estimator = make_estimator()
    unit_span = torch.linspace(0, 1, 11, device=device, dtype=torch.bfloat16)
    time_span = (1 - torch.cos(unit_span * 0.5 * torch.pi)).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        estimator.build_modulations(time_span, torch.bfloat16)
    flow = WholeHistoryCudaGraphRunner(
        estimator,
        graphs=graphs,
        frame_dtype=torch.float32,
        speaker_dtype=torch.bfloat16,
        cfg_rate=0.7,
        euler_steps=10,
        mel_channels=80,
        speaker_channels=80,
        max_rows=2,
        tier_frames=[256, 512],
    )
    flow.capture()
    hift = make_causal_hift(10.0)
    stages.CosyVoice3Vocoder(HiftFlowStub(), hift)
    hift_runner = HiftDecodeCudaGraphRunner(
        hift,
        graphs=graphs,
        mel_channels=80,
        samples_per_frame=480,
        row_tiers=[1, 4],
        frame_tiers=[64, 128],
    )
    hift_runner.capture()
    checks = []
    for lengths, rows, frames in (
        ((70, 80), 3, 64),
        ((200, 100), 1, 128),
        ((70,), 4, 64),
        ((70, 80), 2, 128),
    ):
        inputs = dict(
            noise=torch.randn(1, sum(lengths), 80, device=device),
            time_span=time_span,
            mu=torch.randn(1, sum(lengths), 80, device=device),
            speaker_embeddings=torch.randn(len(lengths), 80, device=device).bfloat16(),
            mel_conditioning=torch.randn(1, sum(lengths), 80, device=device),
            lengths=lengths,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first_flow = flow.run(**inputs)
        assert first_flow is not None
        expected_flow = first_flow.clone()
        mel = torch.randn(rows, 80, frames, device=device)
        source = torch.randn(rows, 1, frames * 480, device=device) * 0.1
        first_hift = hift_runner.run(mel, source).clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            second_flow = flow.run(**inputs)
        torch.testing.assert_close(first_flow, expected_flow, rtol=0, atol=0)
        torch.testing.assert_close(second_flow, expected_flow, rtol=0, atol=0)
        second_hift = hift_runner.run(mel, source).clone()
        eager_hift = hift.decode(x=mel, s=source, finalize=True)
        torch.testing.assert_close(first_hift, eager_hift, rtol=0, atol=0)
        torch.testing.assert_close(second_hift, eager_hift, rtol=0, atol=0)
        checks.append(
            dict(
                flow_lengths=lengths, hift_rows=rows, hift_frames=frames, bit_equal=True
            )
        )
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
