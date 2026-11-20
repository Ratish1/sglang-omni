# SPDX-License-Identifier: Apache-2.0
"""Select the private CUTLASS library for a native diagnostic worker."""

import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from flashinfer.fused_moe import core
from flashinfer.jit import env
from flashinfer.jit.core import JitSpecNvcc
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module


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
