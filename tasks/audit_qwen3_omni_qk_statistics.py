# SPDX-License-Identifier: Apache-2.0
"""Expose normalization statistics at matched native Q/K divergence points."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf


@dataclass(kw_only=True)
class ProbePoint:
    sample_id: str
    frame_index: int
    layer_index: int


@dataclass(kw_only=True)
class StatisticsComparison:
    sample_id: str
    frame: int
    layer: int
    operand: Literal["query", "key"]
    arm: Literal["eager", "fused"]
    request_row: int
    graph_rows: int
    selected_changed_coordinates: list[list[int]]
    selected_kernel_statistics: list[list[float]]
    selected_torch_argument: list[float]
    selected_torch_inverse: list[float]
    same_argument_rsqrt_equal: bool
    captured_selected_output_exact: bool
    probe_graph_and_direct_kernel_exact: bool


def replace_source(source: str, expected: str, replacement: str, count: int = 1) -> str:
    assert source.count(expected) == count, expected
    return source.replace(expected, replacement)


def create_warp_probe(output: Path) -> tuple[Path, str]:
    source = (KERNEL_PATH / "csrc/elementwise/rmsnorm_hf.cuh").read_text()
    original_sha256 = sha256(source.encode()).hexdigest()
    source = replace_source(
        source,
        "  float eps;\n};",
        "  float eps;\n  float* statistics = nullptr;\n"
        "  const float* other_argument = nullptr;\n};",
    )
    source = replace_source(
        source,
        "num_tokens, eps] = params;",
        "num_tokens, eps, statistics, other_argument] = params;",
        2,
    )
    source = replace_source(
        source,
        "    const float rstd = math::rsqrt(lsq / kDim + eps);",
        """    const float inverse_argument = lsq / kDim + eps;
    const float rstd = math::rsqrt(inverse_argument);
    if (threadIdx.x == 0) {
      statistics[4 * row] = lsq;
      statistics[4 * row + 1] = inverse_argument;
      statistics[4 * row + 2] = rstd;
      statistics[4 * row + 3] = math::rsqrt(other_argument[row]);
    }""",
    )
    start = source.index("struct HFRMSNormWarpKernel {")
    end = source.index("struct HFRMSNormKernel {")
    launcher = source[start:end]
    launcher = replace_source(
        launcher,
        "      float eps) {",
        "      float eps,\n"
        "      const tvm::ffi::TensorView statistics,\n"
        "      const tvm::ffi::TensorView other_argument) {",
    )
    launcher = replace_source(
        launcher,
        "    const auto num_tokens = static_cast<uint32_t>(N.unwrap());",
        "    TensorMatcher({N, 4}).with_dtype<float>().with_device(device_).verify(statistics);\n"
        "    TensorMatcher({N}).with_dtype<float>().with_device(device_).verify(other_argument);\n"
        "    const auto num_tokens = static_cast<uint32_t>(N.unwrap());",
    )
    launcher = replace_source(
        launcher,
        "        .eps = eps,",
        "        .eps = eps,\n"
        "        .statistics = static_cast<float*>(statistics.data_ptr()),\n"
        "        .other_argument = static_cast<const float*>(other_argument.data_ptr()),",
    )
    destination = output / "qk_statistics.cuh"
    destination.write_text(source[:start] + launcher + source[end:])
    return destination, original_sha256


@torch.inference_mode()
def main(
    baseline: Path,
    intervention: Path,
    checkpoint: Path,
    output: Path,
    points: list[ProbePoint],
) -> None:
    output.mkdir()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    config = json.loads((checkpoint / "config.json").read_text())["talker_config"][
        "text_config"
    ]
    head_channels = config["head_dim"]
    assert head_channels == 128
    epsilon = config["rms_norm_eps"]
    query_channels = config["num_attention_heads"] * head_channels
    key_channels = config["num_key_value_heads"] * head_channels
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    probe_path, source_sha256 = create_warp_probe(output)
    template = make_cpp_args(head_channels, is_arch_support_pdl(), torch.bfloat16)
    module = load_jit(
        "qwen3_omni_qk_statistics",
        *template,
        cuda_files=[str(probe_path)],
        cuda_wrappers=[("read_statistics", f"HFRMSNormWarpKernel<{template}>::run")],
    )
    capture_sha256: dict[str, str] = {}
    for path in sorted(
        {
            root / "captures" / point.sample_id / "trajectory.pt"
            for root in (baseline, intervention)
            for point in points
        }
    ):
        with path.open("rb") as capture:
            capture_sha256[str(path)] = file_digest(capture, "sha256").hexdigest()
    reports: list[StatisticsComparison] = []
    for point in points:
        directories = [
            root / "captures" / point.sample_id for root in (baseline, intervention)
        ]
        frames = [
            torch.load(
                directory / "trajectory.pt", weights_only=True, map_location="cpu"
            )[point.frame_index]
            for directory in directories
        ]
        states = [
            json.loads((directory / "trajectory.json").read_text())["frames"][
                point.frame_index
            ]
            for directory in directories
        ]
        assert [state["qk_trial"] for state in states] == [
            "checkpoint",
            "fused_checkpoint",
        ]
        prefix = f"layer{point.layer_index}_"
        assert torch.equal(
            frames[0][prefix + "qkv_projection"], frames[1][prefix + "qkv_projection"]
        )
        for operand, channels, start in (
            ("query", query_channels, 0),
            ("key", key_channels, query_channels),
        ):
            name = f"talker.model.layers.{point.layer_index}.self_attn.{operand[0]}_norm.weight"
            with safe_open(
                checkpoint / weight_map[name], framework="pt", device="cpu"
            ) as shard:
                weight = shard.get_tensor(name).cuda()
            for arm, frame, state in zip(
                ("eager", "fused"), frames, states, strict=True
            ):
                raw = frame[prefix + "peer_qkv_projection"].cuda()[
                    :, start : start + channels
                ]
                expanded = raw.reshape(len(raw), -1, head_channels).float()
                argument = expanded.pow(2).mean(-1) + epsilon
                inverse = torch.rsqrt(argument)
                eager_output = (expanded * inverse[..., None]).to(raw.dtype) * weight
                kernel_input = raw.reshape(-1, head_channels).contiguous()
                direct = rmsnorm_hf(kernel_input, weight, epsilon)
                observed = torch.empty_like(direct)
                statistics = torch.full(
                    (len(kernel_input), 4), float("nan"), device="cuda"
                )
                other_argument = argument.flatten().contiguous()
                module.read_statistics(
                    kernel_input, weight, observed, epsilon, statistics, other_argument
                )
                assert torch.equal(observed, direct)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    module.read_statistics(
                        kernel_input,
                        weight,
                        observed,
                        epsilon,
                        statistics,
                        other_argument,
                    )
                graph.replay()
                assert (
                    torch.equal(observed, direct) and torch.isfinite(statistics).all()
                )
                reconstructed = (kernel_input.float() * statistics[:, 2, None]).to(
                    raw.dtype
                ) * weight
                assert torch.equal(reconstructed, direct)
                expected = (
                    eager_output.reshape_as(raw)
                    if arm == "eager"
                    else direct.reshape_as(raw)
                )
                row = state["request_row"]
                assert torch.equal(
                    expected[row].cpu(), frame[prefix + operand + "_before_rope"]
                )
                heads = channels // head_channels
                selected = slice(row * heads, (row + 1) * heads)
                changed = (
                    (eager_output.reshape_as(direct)[selected] != direct[selected])
                    .nonzero()
                    .tolist()
                )
                reports.append(
                    StatisticsComparison(
                        sample_id=point.sample_id,
                        frame=point.frame_index,
                        layer=point.layer_index,
                        operand=operand,
                        arm=arm,
                        request_row=row,
                        graph_rows=len(raw),
                        selected_changed_coordinates=changed,
                        selected_kernel_statistics=statistics[selected].tolist(),
                        selected_torch_argument=argument[row].tolist(),
                        selected_torch_inverse=inverse[row].tolist(),
                        same_argument_rsqrt_equal=torch.equal(
                            statistics[:, 3], inverse.flatten()
                        ),
                        captured_selected_output_exact=True,
                        probe_graph_and_direct_kernel_exact=True,
                    )
                )
    report = {
        "source_sha256": source_sha256,
        "capture_sha256": capture_sha256,
        "probe_sha256": sha256(probe_path.read_bytes()).hexdigest(),
        "statistic_columns": [
            "sum_of_squares",
            "inverse_argument",
            "inverse_rms",
            "inverse_on_torch_argument",
        ],
        "epsilon": epsilon,
        "scope": "Held native peer operands and selected outputs only; no serving intervention or speech-quality claim.",
        "points": [asdict(report) for report in reports],
    }
    (output / "statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"operator_comparisons": len(reports), "all_output_checks_passed": True}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("intervention", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--point",
        nargs=3,
        action="append",
        required=True,
        metavar=("SAMPLE", "FRAME", "LAYER"),
    )
    arguments = parser.parse_args()
    main(
        arguments.baseline,
        arguments.intervention,
        arguments.checkpoint,
        arguments.output,
        [
            ProbePoint(sample_id=sample, frame_index=int(frame), layer_index=int(layer))
            for sample, frame, layer in arguments.point
        ],
    )
else:
    pass
