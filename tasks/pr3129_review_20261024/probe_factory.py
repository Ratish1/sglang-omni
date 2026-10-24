"""Exercise the real factory's no-cache branch at a declared share boundary."""

import json
from unittest.mock import patch

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import FunCosyVoice3StreamingVocoderScheduler
from tests.unit_test.fun_cosyvoice3.test_vocoder import FakeHiFT, RunnableFakeFlow


def main():
    fake = RunnableFakeFlow()
    fake.packed_estimator.is_ragged = True
    flow = stages.FunCosyVoice3Flow(fake, packed_estimator=fake.packed_estimator)
    load_bytes = 2**20
    memory_readings = iter((0, load_bytes))
    recorded = []

    def reserve(device):
        return next(memory_readings)

    def build(*args, **kwargs):
        recorded.append(kwargs)
        # The sizing helper's documented result when a declared stage share has
        # no room for a cache block. Do not mask the caller's failure boundary.
        return None

    with (
        patch.object(stages, "resolve_concrete_device", return_value=torch.device("cuda:0")),
        patch.object(stages, "resolve_checkpoint", return_value="/checkpoint"),
        patch.object(stages, "patch_chunk_mask"),
        patch.object(stages, "load_cosyvoice3_flow_hift", return_value=(flow, FakeHiFT())),
        patch.object(torch.accelerator, "memory_reserved", side_effect=reserve),
        patch.object(torch.accelerator, "get_memory_info", return_value=(2**29, 2**30)),
        patch.object(FunCosyVoice3StreamingVocoderScheduler, "measure_peak_step_bytes", return_value=0),
        patch.object(stages, "build_prefix_pool", side_effect=build),
    ):
        try:
            scheduler = stages.create_vocoder_executor(
                "model", device="cuda:0", dtype="float16", max_batch_size=4,
                flow_prefix_cache_gb=None, enable_flow_cuda_graph=False,
                enable_flow_prefix_cuda_graph=True,
                enable_flow_whole_history_cuda_graph=False,
                enable_hift_cuda_graph=False, enable_dit_torch_compile=False,
                total_gpu_memory_fraction=load_bytes / 2**30,
            )
        except AssertionError:
            print(json.dumps(dict(outcome="startup_AssertionError", pool_calls=recorded)), flush=True)
        else:
            print(json.dumps(dict(outcome="cache_disabled", pool_is_none=scheduler.vocoder.flow.prefix_pool is None)), flush=True)


if __name__ == "__main__":
    main()
