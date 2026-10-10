import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageTransformer2DModelWrapper,
    decoder_config,
)
from sglang_omni.models.llada2_uni.components.decoder_runtime import (
    initialize_decoder_runtime,
)


@dataclass
class Comparison:
    batch_size: int
    height: int
    width: int
    caption_length: int
    max_absolute_error: float
    root_mean_squared_error: float
    tolerance_passed: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = decoder_config(
        {
            "dim": 768,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 6,
            "n_kv_heads": 6,
            "cap_feat_dim": 16,
            "axes_dims": (32, 48, 48),
            "axes_lens": (128, 64, 64),
        }
    )
    comparisons: list[Comparison] = []
    with initialize_decoder_runtime(
        str(arguments.checkpoint), gpu_id=0, attention_backend="torch_sdpa"
    ) as runtime:
        with runtime.compute_context(), torch.inference_mode():
            reference = ZImageTransformer2DModelWrapper(
                str(arguments.checkpoint), config, runtime.device, runtime.dtype
            )
            native = ZImageTransformer2DModelWrapper(
                str(arguments.checkpoint),
                config,
                runtime.device,
                runtime.dtype,
                backend="sglang",
                runtime=runtime,
            )
            generator = torch.Generator(device=runtime.device).manual_seed(2026)
            for batch_size, height, width, caption_length in (
                (1, 14, 18, 7),
                (1, 14, 18, 7),
                (2, 16, 18, 33),
                (1, 18, 14, 33),
            ):
                latents = torch.randn(
                    batch_size,
                    16,
                    1,
                    height,
                    width,
                    device=runtime.device,
                    dtype=runtime.dtype,
                    generator=generator,
                )
                captions = torch.randn(
                    batch_size,
                    caption_length,
                    16,
                    device=runtime.device,
                    dtype=runtime.dtype,
                    generator=generator,
                )
                times = torch.linspace(0.125, 0.875, batch_size, device=runtime.device)
                expected = torch.stack(
                    reference(
                        list(latents.unbind(0)),
                        times,
                        list(captions.unbind(0)),
                        return_dict=False,
                    )[0]
                )
                actual = torch.stack(
                    native(
                        list(latents.unbind(0)),
                        times,
                        list(captions.unbind(0)),
                        return_dict=False,
                    )[0]
                )
                assert actual.shape == expected.shape
                assert torch.isfinite(actual).all()
                difference = actual.float() - expected.float()
                passed = bool(
                    torch.isclose(
                        actual.float(), expected.float(), rtol=0.02, atol=0.02
                    )
                    .all()
                    .item()
                )
                comparison = Comparison(
                    batch_size,
                    height,
                    width,
                    caption_length,
                    float(difference.abs().max().item()),
                    float(difference.square().mean().sqrt().item()),
                    passed,
                )
                comparisons.append(comparison)
                print(json.dumps(asdict(comparison)), flush=True)
    assert not torch.distributed.is_initialized()
    arguments.output.write_text(
        json.dumps([asdict(comparison) for comparison in comparisons], indent=2) + "\n"
    )
    assert all(comparison.tolerance_passed for comparison in comparisons)


if __name__ == "__main__":
    main()
