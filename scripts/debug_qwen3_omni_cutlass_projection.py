# SPDX-License-Identifier: Apache-2.0
"""Build or select a private CUTLASS BF16 projection-rounding control."""

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path

from flashinfer.fused_moe import core
from flashinfer.jit import env
from flashinfer.jit.core import JitSpecNvcc
from flashinfer.jit.cpp_ext import run_ninja
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

FUSED_SWIGLU_HEADER = Path(
    "nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/"
    "gemm/kernel/fused_moe_kernel_routine.cuh"
)
FUSED_SWIGLU_HEADER_SHA256 = (
    "3d5ffec8a48ce3d9e0fc7acf84850e3c5cad11b1935ce296311e6427c8e76ee1"
)


def file_hashes(source_directory: Path) -> dict[str, str]:
    return {
        str(source_file.relative_to(source_directory)): hashlib.sha256(
            source_file.read_bytes()
        ).hexdigest()
        for source_file in source_directory.rglob("*")
        if source_file.is_file()
    }


def build_specification(
    source_directory: Path, use_fast_build: bool = False
) -> JitSpecNvcc:
    assert not use_fast_build
    specification = gen_cutlass_fused_moe_sm90_module(use_fast_build)
    assert isinstance(specification, JitSpecNvcc)
    assert specification.extra_include_dirs is not None
    installed_directory = env.FLASHINFER_CSRC_DIR
    source_digest = hashlib.sha256(
        json.dumps(file_hashes(source_directory), sort_keys=True).encode()
    ).hexdigest()
    return replace(
        specification,
        name=f"fused_moe_90_projection_control_{source_digest[:16]}",
        sources=[
            (
                source_directory / source_file.relative_to(installed_directory)
                if source_file.is_relative_to(installed_directory)
                else source_file
            )
            for source_file in specification.sources
        ],
        extra_include_dirs=[
            source_directory,
            *[
                (
                    source_directory
                    / include_directory.relative_to(installed_directory)
                    if include_directory.is_relative_to(installed_directory)
                    else include_directory
                )
                for include_directory in specification.extra_include_dirs
            ],
        ],
    )


def install_projection_control(source_directory: Path) -> None:
    assert core.get_cutlass_fused_moe_module.cache_info().currsize == 0
    specification = build_specification(source_directory)
    assert specification.jit_library_path.is_file()
    core.gen_cutlass_fused_moe_sm90_module = partial(
        build_specification, source_directory
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
