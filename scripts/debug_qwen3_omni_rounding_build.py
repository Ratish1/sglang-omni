# SPDX-License-Identifier: Apache-2.0
"""Build an isolated CUTLASS module for the projection-rounding experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import torch
from flashinfer.jit.env import FLASHINFER_CSRC_DIR
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--round-projections", action="store_true")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    original = gen_cutlass_fused_moe_sm90_module()
    assert original.extra_cuda_cflags is not None
    specification = replace(
        original,
        name=(
            "fused_moe_90_projection_rounding_diagnostic"
            if arguments.round_projections
            else "fused_moe_90_projection_control_diagnostic"
        ),
        extra_cuda_cflags=[
            *original.extra_cuda_cflags,
            *(
                ["-DFLASHINFER_DIAGNOSTIC_ROUND_GATE_UP_BF16=1"]
                if arguments.round_projections
                else []
            ),
        ],
    )
    header = FLASHINFER_CSRC_DIR / (
        "nv_internal/tensorrt_llm/cutlass_extensions/include/"
        "cutlass_extensions/gemm/kernel/fused_moe_kernel_routine.cuh"
    )
    assert b"FLASHINFER_DIAGNOSTIC_ROUND_GATE_UP_BF16" in header.read_bytes()
    print(f"Building {specification.name} in {specification.build_dir}", flush=True)
    specification.build_and_load()
    library = arguments.output_directory / "projection_rounding.so"
    shutil.copyfile(specification.get_library_path(), library)
    manifest = {
        "flashinfer": version("flashinfer-python"),
        "gpu": torch.cuda.get_device_name(arguments.gpu),
        "module": specification.name,
        "round_projections": arguments.round_projections,
        "original_library": str(original.get_library_path()),
        "diagnostic_library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "cuda_flags": specification.extra_cuda_cflags,
        "scope": (
            "BF16 gated SM80-family GEMM1 rounds both FP32 projections to BF16 immediately before the existing activation. No tactic or reduction algorithm is changed."
            if arguments.round_projections
            else "Unmodified arithmetic rebuilt from the same source and compiler as the rounding diagnostic."
        ),
    }
    (arguments.output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
