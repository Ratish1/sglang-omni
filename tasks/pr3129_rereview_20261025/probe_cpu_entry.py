"""Check that requesting a non-CUDA vocoder still reaches its model loader."""

import json
from unittest.mock import patch

from sglang_omni.models.fun_cosyvoice3 import stages


class LoaderReached(Exception):
    pass


def main():
    records = []
    for cache_size in (None, 0.0):
        loader = None
        with (
            patch.object(
                stages, "resolve_checkpoint", return_value="/review-checkpoint"
            ),
            patch.object(
                stages, "load_cosyvoice3_flow_hift", side_effect=LoaderReached
            ) as loader,
        ):
            try:
                stages.create_vocoder_executor(
                    "review-checkpoint",
                    device="cpu",
                    dtype="float32",
                    enable_dit_torch_compile=False,
                    enable_flow_cuda_graph=False,
                    enable_flow_prefix_cuda_graph=False,
                    enable_flow_whole_history_cuda_graph=False,
                    enable_hift_cuda_graph=False,
                    flow_prefix_cache_gb=cache_size,
                )
            except Exception as error:
                records.append(
                    dict(
                        cache_size=cache_size,
                        loader_reached=loader.called,
                        outcome=type(error).__name__,
                        error=str(error),
                    )
                )
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
