"""Review-only check for dynamic segment metadata across CUDA graph replays."""

import argparse
import json
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    VisionTritonAttention,
)

from sglang_omni.models.qwen3_omni.components import audio_layer_graph
from sglang_omni.models.qwen3_omni.components.audio_layer_graph import (
    AudioLayerGraphRunner,
)
from tests.unit_test.qwen3_omni.test_audio_layer_graph import (
    WINDOW,
    make_cu_seqlens,
    real_tower,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-metadata", action="store_true")
    options = parser.parse_args()
    records = []
    envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.set(True)
    for backend_class in (VisionFlash3Attention, VisionTritonAttention):
        torch.manual_seed(3131)
        device = torch.device("cuda", 0)
        tower = real_tower(device)
        runner = AudioLayerGraphRunner(
            tower,
            device=device,
            window=WINDOW,
            packed_attention=backend_class(use_data_parallel=True),
            token_buckets=(128, 256),
        )
        if options.freeze_metadata:
            builder = audio_layer_graph.prepare_packed_attention_metadata
            cached_metadata = {}

            def build_once(cumulative, longest):
                address = cumulative.data_ptr()
                if address not in cached_metadata:
                    cached_metadata[address] = builder(cumulative, longest)
                return cached_metadata[address]

            with patch.object(
                audio_layer_graph,
                "prepare_packed_attention_metadata",
                side_effect=build_once,
            ):
                runner.capture_all()
        else:
            runner.capture_all()
        assert runner.has_graphs, runner.disabled_reason
        cases = ([104, 23], [37, 90], [63, 64], [104, 24], [104, 25], [1] * 80)
        for segments in cases:
            hidden = torch.randn(
                sum(segments), tower.config.d_model, device=device
            ).bfloat16()
            cumulative = make_cu_seqlens(segments, device)
            with torch.no_grad():
                expected = runner.run_layers(hidden, cumulative, WINDOW)
                actual = runner.maybe_replay(hidden, cumulative, segments)
            record = dict(
                backend=backend_class.__name__,
                segments=segments,
                replayed=actual is not None,
            )
            if len(segments) == 80:
                assert actual is None
            else:
                assert actual is not None
                record["max_abs"] = float((actual - expected).abs().max())
                try:
                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                except AssertionError:
                    records.append(record)
                    print(json.dumps(records), flush=True)
                    raise
            records.append(record)
        del runner, tower
        torch.cuda.empty_cache()
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
