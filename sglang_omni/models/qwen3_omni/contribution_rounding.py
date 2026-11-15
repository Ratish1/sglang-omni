"""Select a separately built expert contribution-rounding diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
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


def shared_activation_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


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


def install_contribution_provider(
    source_directory: Path, gemm1_profile: int | None = None
) -> None:
    """Select the diagnostic before any native expert runner is created."""
    assert expert_core.get_cutlass_fused_moe_module.cache_info().currsize == 0
    specification = contribution_build_spec(source_directory)
    assert specification.jit_library_path.is_file()
    expert_core.gen_cutlass_fused_moe_sm90_module = partial(
        contribution_build_spec, source_directory
    )
    if gemm1_profile is None:
        pass
    else:
        install_gemm1_profile(gemm1_profile)


def install_gemm1_profile(gemm1_profile: int) -> None:
    tuner = AutoTuner.get()
    original_choose_one = tuner.choose_one

    def choose_one(
        custom_op: str,
        runners: list[TunableRunner],
        tuning_config: TuningConfig,
        inputs: list[torch.Tensor | None],
        **kernel_arguments: int,
    ) -> tuple[TunableRunner, int | tuple[int, ...]]:
        runner, tactic = original_choose_one(
            custom_op, runners, tuning_config, inputs, **kernel_arguments
        )
        if custom_op == "trtllm::fused_moe::gemm1":
            return runner, gemm1_profile
        else:
            return runner, tactic

    tuner.choose_one = choose_one
