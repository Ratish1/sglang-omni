"""Compile and substitute one captured MoE tile kernel for synchronization controls."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

import torch
from cuda.bindings import driver
from flashinfer.jit import env as jit_environment
from flashinfer.jit.core import sm90a_nvcc_flags

FUSED_TILE_HEADER_SHA256 = (
    "98a2a94d39f477d8e2f3b4ce43b67d7755309d0dfa6961d7f061f7b498f64598"
)


def compile_tile_kernel(
    output_directory: Path, mode: Literal["original", "barrier"]
) -> tuple[driver.CUmodule, driver.CUfunction]:
    include_directory = output_directory / "tile-include"
    relative_header = Path("cutlass_extensions/gemm/kernel/fused_moe_kernel.cuh")
    original_include = (
        jit_environment.FLASHINFER_CSRC_DIR
        / "nv_internal/tensorrt_llm/cutlass_extensions/include"
    )
    header = (original_include / relative_header).read_bytes()
    assert hashlib.sha256(header).hexdigest() == FUSED_TILE_HEADER_SHA256
    header_text = header.decode()
    if mode == "barrier":
        original_line = "      problem_visitor.advance(gridDim.x);"
        assert header_text.count(original_line) == 1
        header_text = header_text.replace(
            original_line, "      __syncthreads();\n" + original_line
        )
    else:
        pass
    target_header = include_directory / relative_header
    target_header.parent.mkdir(parents=True)
    target_header.write_text(header_text)
    source = output_directory / "tile-control.cu"
    source.write_text(
        "#include <cutlass_extensions/gemm/kernel/fused_moe_kernel.cuh>\n"
        "using ActualKernel = fused_moe::Fused_Moe_Kernel_sm80<"
        "cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, "
        "64, 128, 64, 4, fused_moe::Activation_Type::Swiglu>;\n"
        'extern "C" __global__ void qwen_moe_tile_control('
        "__grid_constant__ ActualKernel::Params const parameters) {\n"
        "  ActualKernel kernel;\n"
        "  kernel.run_device(parameters);\n"
        "}\n"
    )
    cubin = output_directory / "tile-control.cubin"
    include_paths = [
        include_directory,
        original_include,
        jit_environment.FLASHINFER_CSRC_DIR / "nv_internal",
        jit_environment.FLASHINFER_CSRC_DIR / "nv_internal/include",
        *jit_environment.CUTLASS_INCLUDE_DIRS,
    ]
    command = [
        "nvcc",
        "--cubin",
        "-std=c++17",
        "-use_fast_math",
        "--expt-relaxed-constexpr",
        "-DNDEBUG",
        "-O3",
        *sm90a_nvcc_flags,
        *(f"-I{directory}" for directory in include_paths),
        str(source),
        "-o",
        str(cubin),
    ]
    (output_directory / "tile-build-command.json").write_text(
        json.dumps(command, indent=2)
    )
    with (output_directory / "tile-build.log").open("w") as build_log:
        subprocess.run(command, check=True, stdout=build_log, stderr=subprocess.STDOUT)
    torch.cuda.init()
    status, module = driver.cuModuleLoad(str(cubin).encode())
    assert status == driver.CUresult.CUDA_SUCCESS, status
    status, function = driver.cuModuleGetFunction(module, b"qwen_moe_tile_control")
    assert status == driver.CUresult.CUDA_SUCCESS, status
    return module, function


def replace_tile_kernel(
    graph: torch.cuda.CUDAGraph, function: driver.CUfunction
) -> None:
    graph_handle = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(graph_handle)
    assert status == driver.CUresult.CUDA_SUCCESS, status
    status, nodes, count = driver.cuGraphGetNodes(graph_handle, count)
    assert status == driver.CUresult.CUDA_SUCCESS, status
    replacements = 0
    for node in nodes:
        status, node_type = driver.cuGraphNodeGetType(node)
        assert status == driver.CUresult.CUDA_SUCCESS, status
        if node_type == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            status, parameters = driver.cuGraphKernelNodeGetParams(node)
            assert status == driver.CUresult.CUDA_SUCCESS, status
            status, name = driver.cuFuncGetName(parameters.func)
            assert status == driver.CUresult.CUDA_SUCCESS, status
            if b"Fused_Moe_Kernel_sm80" in name:
                assert b"Li64ELi128ELi64ELi4E" in name, name
                (status,) = driver.cuFuncSetAttribute(
                    function,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    parameters.sharedMemBytes,
                )
                assert status == driver.CUresult.CUDA_SUCCESS, status
                parameters.func = function
                (status,) = driver.cuGraphKernelNodeSetParams(node, parameters)
                assert status == driver.CUresult.CUDA_SUCCESS, status
                replacements += 1
            else:
                pass
        else:
            pass
    assert replacements == 1, replacements
