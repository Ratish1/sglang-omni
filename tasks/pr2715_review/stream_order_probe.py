"""Exercise the factory's first packed solve while cache construction is pending."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from cosyvoice.flow.DiT.dit import DiT

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from tests.unit_test.fun_cosyvoice3.test_flow_batch import FakeFlow


torch.cuda.set_device(0)
torch.manual_seed(2715)
flow = FakeFlow(channels=8, max_frames=256).cuda().eval()
flow.decoder.rand_noise = flow.decoder.rand_noise.cuda()
flow.decoder.estimator = DiT(
    dim=32, depth=2, heads=2, dim_head=16, mel_dim=8, mu_dim=8,
    spk_dim=8, out_channels=8, static_chunk_size=4, num_decoding_left_chunks=-1,
).cuda().eval()
estimator = PackedDiT(flow.decoder.estimator, device="cuda")
wrapped = stages.FunCosyVoice3Flow(flow, estimator)
hift = SimpleNamespace(
    upsample_rates=[1], istft_params={"hop_len": 1},
    f0_predictor=SimpleNamespace(condnet=[SimpleNamespace(causal_padding=0)]),
    conv_pre_look_right=0,
)
original_build = PackedDiT.build_modulations
producer_started = torch.cuda.Event()
observations = {}


def delayed_build(self, time_span, dtype):
    # Fill the allocator and native kernels before introducing a deterministic queue delay.
    original_build(self, time_span + 0.23, dtype)
    original_build(self, time_span, dtype)
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000_000)
    producer_started.record()
    original_build(self, time_span, dtype)
    observations["producer_started_before_build_returned"] = producer_started.query()


def first_solve(self):
    item = self.make_warmup_flow_input(28)
    with self.vocoder.stream_context:
        actual = self.vocoder.hop_batch([item])[0]
        torch.cuda.current_stream().synchronize()
        observations["producer_started_before_first_solve_finished"] = producer_started.query()
        snapshot = actual.cpu()
    torch.cuda.synchronize()
    with self.vocoder.stream_context:
        expected = self.vocoder.hop_batch([item])[0].cpu()
    observations["pcm_equal_after_producer_finished"] = torch.equal(snapshot, expected)
    observations["max_mel_difference"] = float((snapshot - expected).abs().max())


with (
    patch.object(stages, "resolve_checkpoint", return_value="diagnostic-native-flow"),
    patch.object(stages, "load_cosyvoice3_flow_hift", return_value=(wrapped, hift)),
    patch.object(PackedDiT, "build_modulations", delayed_build),
    patch.object(FunCosyVoice3StreamingVocoderScheduler, "warmup_now", first_solve),
):
    stages.create_vocoder_executor(
        "diagnostic-native-flow", device="cuda", gpu_id=0, dtype="float32",
        enable_dit_torch_compile=False, enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False, enable_flow_whole_history_cuda_graph=False,
        flow_prefix_cache_gb=0.0,
    )
print(json.dumps(observations), flush=True)
