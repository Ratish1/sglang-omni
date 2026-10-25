"""Observe device and platform initialization caused by a fresh stage import."""

import argparse
import importlib
import json
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="sglang_omni.models.qwen3_omni.stages")
    options = parser.parse_args()
    before = torch.cuda.is_initialized()
    importlib.import_module(options.module)
    print(
        json.dumps(
            dict(
                module=options.module,
                cuda_before=before,
                cuda_after=torch.cuda.is_initialized(),
                omni_platform_loaded="sglang_omni.platforms" in sys.modules,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
