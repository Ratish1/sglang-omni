"""Separate RMS reduction order from BF16 stores on actual speech operands."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRMSNorm,
)

# note (ratish): these are the scalar CTA and warp widths in rmsnorm_hf.cuh.
NORMALIZATION_THREADS = 512
WARP_LANES = 32


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    bundle = torch.load(arguments.operands_file, map_location="cpu", weights_only=True)
    assert bundle["complete_request_count"] == 4
    provenance = {
        "operands_sha256": hashlib.file_digest(
            arguments.operands_file.open("rb"), "sha256"
        ).hexdigest(),
        "source": bundle["source"],
        "checkpoint_forward": inspect.getsource(Qwen3OmniMoeThinkerTextRMSNorm.forward),
        "native_entrypoint": inspect.getsource(rmsnorm_hf),
    }
    (arguments.output_directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    summaries = []
    for normalization in bundle["normalizations"]:
        residual = normalization["residual"].to(device)
        if normalization["contribution"] is not None:
            residual = residual + normalization["contribution"].to(device)
        else:
            pass
        observed = normalization["normalized"].to(device)
        weight = normalization["weight"].to(device)
        assert residual.dtype == observed.dtype == weight.dtype == torch.bfloat16
        assert residual.shape == observed.shape
        assert residual.shape[-1] == weight.numel()
        assert len(normalization["origins"]) == len(residual)
        assert residual.isfinite().all() and observed.isfinite().all()
        width = residual.shape[-1]
        assert width >= NORMALIZATION_THREADS and width % NORMALIZATION_THREADS == 0
        assert width & (width - 1) == 0
        epsilon = normalization["epsilon"]
        native = rmsnorm_hf(residual, weight, epsilon)
        assert torch.equal(native.view(torch.uint8), observed.view(torch.uint8)), (
            normalization["name"],
            "native operand replay",
        )
        checkpoint = Qwen3OmniMoeThinkerTextRMSNorm(width, eps=epsilon).to(
            device=device, dtype=torch.bfloat16
        )
        checkpoint.weight.copy_(weight)
        eager = checkpoint(residual)
        values = residual.float()
        partial_squares = torch.zeros(
            (len(values), NORMALIZATION_THREADS), device=device, dtype=torch.float32
        )
        for offset in range(0, width, NORMALIZATION_THREADS):
            partial_squares = (
                partial_squares
                + values[:, offset : offset + NORMALIZATION_THREADS].square()
            )
        warp_squares = partial_squares.reshape(len(values), -1, WARP_LANES)
        lanes = torch.arange(WARP_LANES, device=device)
        for offset in (16, 8, 4, 2, 1):
            warp_squares = warp_squares + warp_squares.index_select(-1, lanes ^ offset)
        block_squares = torch.zeros(
            (len(values), WARP_LANES), device=device, dtype=torch.float32
        )
        block_squares[:, : NORMALIZATION_THREADS // WARP_LANES] = warp_squares[:, :, 0]
        for offset in (16, 8, 4, 2, 1):
            block_squares = block_squares + block_squares.index_select(
                -1, lanes ^ offset
            )
        native_variance = block_squares[:, :1] / width
        eager_variance = values.pow(2).mean(-1, keepdim=True)
        native_inverse_rms = torch.rsqrt(native_variance + epsilon)
        eager_inverse_rms = torch.rsqrt(eager_variance + epsilon)
        reconstructed = (values * native_inverse_rms).bfloat16() * weight
        if not torch.equal(reconstructed.view(torch.uint8), native.view(torch.uint8)):
            torch.save(
                {
                    "residual": residual.cpu(),
                    "weight": weight.cpu(),
                    "native": native.cpu(),
                    "reconstructed": reconstructed.cpu(),
                    "eager": eager.cpu(),
                    "native_variance": native_variance.cpu(),
                    "eager_variance": eager_variance.cpu(),
                    "native_inverse_rms": native_inverse_rms.cpu(),
                    "origins": normalization["origins"],
                    "epsilon": epsilon,
                },
                arguments.output_directory / f"{normalization['name']}-failed.pt",
            )
            raise AssertionError(
                f"Source-derived reduction replay: {normalization['name']}"
            )
        else:
            pass
        eager_formula = (values * eager_inverse_rms).bfloat16() * weight
        assert torch.equal(eager_formula.view(torch.uint8), eager.view(torch.uint8))
        precise_values = residual.double()
        precise_variance = precise_values.square().mean(-1, keepdim=True)
        precise_inverse_rms = torch.rsqrt(precise_variance + epsilon)
        precise = (precise_values * precise_inverse_rms).bfloat16() * weight
        changed = native != eager
        changed_rows = changed.any(dim=-1)
        row_records = []
        for row, origin in enumerate(normalization["origins"]):
            row_records.append(
                {
                    "request_id": origin[0],
                    "frame": origin[1],
                    "token_row": origin[2],
                    "changed_values": int(changed[row].sum()),
                    "variance_difference": float(
                        native_variance[row, 0] - eager_variance[row, 0]
                    ),
                    "inverse_rms_difference": float(
                        native_inverse_rms[row, 0] - eager_inverse_rms[row, 0]
                    ),
                    "native_matches_fp64_values": int(
                        (native[row] == precise[row]).sum()
                    ),
                    "eager_matches_fp64_values": int(
                        (eager[row] == precise[row]).sum()
                    ),
                }
            )
        name = normalization["name"]
        (arguments.output_directory / f"{name}.json").write_text(
            json.dumps(row_records, indent=2)
        )
        torch.save(
            {
                "native_variance": native_variance.cpu(),
                "eager_variance": eager_variance.cpu(),
                "fp64_variance": precise_variance.cpu(),
                "native_inverse_rms": native_inverse_rms.cpu(),
                "eager_inverse_rms": eager_inverse_rms.cpu(),
                "fp64_inverse_rms": precise_inverse_rms.cpu(),
                "changed_rows": changed_rows.cpu(),
                "native_normalized_changed_rows": native[changed_rows].cpu(),
                "eager_normalized_changed_rows": eager[changed_rows].cpu(),
                "fp64_normalized_changed_rows": precise[changed_rows].cpu(),
            },
            arguments.output_directory / f"{name}.pt",
        )
        summary = {
            "name": name,
            "rows": len(residual),
            "values": native.numel(),
            "native_operand_replay_exact": True,
            "source_derived_reduction_replay_exact": True,
            "changed_values": int(changed.sum()),
            "changed_rows": int(changed_rows.sum()),
            "native_matches_fp64_values": int((native == precise).sum()),
            "eager_matches_fp64_values": int((eager == precise).sum()),
        }
        summaries.append(summary)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(
                {
                    "normalizations": summaries,
                    "scope": "All retained native residual-normalization operands. Exact native and source-derived reduction reproduction are mandatory. FP64 measures numerical distance, not universal speech quality or a production-fix criterion.",
                },
                indent=2,
            )
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
