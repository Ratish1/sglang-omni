"""Select a separately built expert contribution-rounding diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from flashinfer.fused_moe import core as expert_core
from flashinfer.jit import env as jit_environment
from flashinfer.jit.core import JitSpecNvcc
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

FINALIZER_PATH = Path("fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh")
GATED_ROUTINE_PATH = Path(
    "nv_internal/tensorrt_llm/cutlass_extensions/include/"
    "cutlass_extensions/gemm/kernel/fused_moe_kernel_routine.cuh"
)


def source_file_fingerprints(source_directory: Path) -> dict[str, str]:
    return {
        filename.relative_to(source_directory)
        .as_posix(): hashlib.sha256(filename.read_bytes())
        .hexdigest()
        for filename in sorted(source_directory.rglob("*"))
        if filename.is_file()
    }


def contribution_build_spec(
    source_directory: Path, use_fast_build: bool = False
) -> JitSpecNvcc:
    """Preserve native code generation while selecting the diagnostic source tree."""
    assert not use_fast_build
    original_specification = gen_cutlass_fused_moe_sm90_module(use_fast_build)
    assert isinstance(original_specification, JitSpecNvcc)
    assert original_specification.extra_include_dirs is not None
    installed_directory = jit_environment.FLASHINFER_CSRC_DIR
    source_files = [
        (
            source_directory / filename.relative_to(installed_directory)
            if filename.is_relative_to(installed_directory)
            else filename
        )
        for filename in original_specification.sources
    ]
    include_directories = [
        (
            source_directory / directory.relative_to(installed_directory)
            if directory.is_relative_to(installed_directory)
            else directory
        )
        for directory in original_specification.extra_include_dirs
    ]
    source_digest = hashlib.sha256(
        json.dumps(source_file_fingerprints(source_directory), sort_keys=True).encode()
    ).hexdigest()
    return replace(
        original_specification,
        name=f"fused_moe_90_diagnostic_{source_digest[:16]}",
        sources=source_files,
        extra_include_dirs=[source_directory, *include_directories],
    )


def install_contribution_provider(source_directory: Path) -> None:
    """Select the diagnostic before any native expert runner is created."""
    assert expert_core.get_cutlass_fused_moe_module.cache_info().currsize == 0
    specification = contribution_build_spec(source_directory)
    assert specification.jit_library_path.is_file()
    expert_core.gen_cutlass_fused_moe_sm90_module = partial(
        contribution_build_spec, source_directory
    )
