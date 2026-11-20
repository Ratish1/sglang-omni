# SPDX-License-Identifier: Apache-2.0
"""Build a private CUTLASS BF16 projection-rounding control."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from flashinfer.jit import env
from flashinfer.jit.cpp_ext import run_ninja

from sglang_omni.models.qwen3_omni.cutlass_projection_control import (
    build_specification,
    file_hashes,
)

FUSED_SWIGLU_HEADER = Path(
    "nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/"
    "gemm/kernel/fused_moe_kernel_routine.cuh"
)
FUSED_SWIGLU_HEADER_SHA256 = (
    "3d5ffec8a48ce3d9e0fc7acf84850e3c5cad11b1935ce296311e6427c8e76ee1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--build-jobs", type=int, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir()
    installed_hashes = file_hashes(env.FLASHINFER_CSRC_DIR)
    assert installed_hashes[str(FUSED_SWIGLU_HEADER)] == FUSED_SWIGLU_HEADER_SHA256
    source_directory = output_directory / "csrc"
    shutil.copytree(env.FLASHINFER_CSRC_DIR, source_directory)
    subprocess.run(
        [
            "patch",
            "--batch",
            "--fuzz=0",
            "--no-backup-if-mismatch",
            "-p1",
            "-d",
            str(source_directory),
            "-i",
            str(Path(__file__).with_name("qwen3_omni_cutlass_projection.patch")),
        ],
        check=True,
    )
    patched_hashes = file_hashes(source_directory)
    assert patched_hashes.keys() == installed_hashes.keys()
    assert {
        filename
        for filename, digest in installed_hashes.items()
        if patched_hashes[filename] != digest
    } == {str(FUSED_SWIGLU_HEADER)}
    specification = build_specification(source_directory)
    assert not specification.is_aot
    specification.write_ninja()
    run_ninja(
        specification.build_dir,
        specification.ninja_path,
        verbose=True,
        max_jobs=arguments.build_jobs,
    )
    assert file_hashes(env.FLASHINFER_CSRC_DIR) == installed_hashes
    assert file_hashes(source_directory) == patched_hashes
    with specification.jit_library_path.open("rb") as library:
        library_sha256 = hashlib.file_digest(library, "sha256").hexdigest()
    (output_directory / "build-result.json").write_text(
        json.dumps(
            {
                "library": str(specification.jit_library_path),
                "library_sha256": library_sha256,
                "installed_header_sha256": FUSED_SWIGLU_HEADER_SHA256,
                "patched_header_sha256": patched_hashes[str(FUSED_SWIGLU_HEADER)],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
else:
    pass
