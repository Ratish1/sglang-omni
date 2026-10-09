import argparse
import importlib.util
import json
from pathlib import Path

import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CARD, DEPFORMER
from sglang_omni.models.personaplex.components.depformer import Depformer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    baseline_spec = importlib.util.spec_from_file_location(
        "baseline_depformer", arguments.baseline_source
    )
    assert baseline_spec is not None and baseline_spec.loader is not None
    baseline_module = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline_module)
    fixture_spec = importlib.util.spec_from_file_location(
        "depformer_test", "tests/unit_test/personaplex/test_depformer.py"
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixtures = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixtures)
    fixtures.SPEC = DEPFORMER
    weights = {
        name: (
            tensor / tensor.shape[-1] ** 0.5
            if tensor.ndim == 2 and "emb" not in name
            else tensor
        )
        for name, tensor in fixtures.reference_weights(DEPFORMER.steps).items()
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    cases: list[dict[str, str | int | float | bool]] = []
    for dtype in (torch.float32, torch.bfloat16):
        with torch.device("cuda"):
            serial = baseline_module.Depformer(DEPFORMER).to(dtype=dtype).eval()
            fused = Depformer(DEPFORMER).to(dtype=dtype).eval()
        serial.load_reference_weights(weights)
        fused.load_reference_weights(weights)
        for batch_size in (1, 4, 8):
            generator = torch.Generator(device="cuda").manual_seed(7)
            hidden = torch.randn(
                batch_size,
                DEPFORMER.input_dim,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            text = torch.arange(batch_size, device="cuda") + 3
            forced = torch.arange(batch_size * DEPFORMER.steps, device="cuda").view(
                batch_size, DEPFORMER.steps
            )
            for mode in ("forced", "mixed", "free"):
                if mode == "forced":
                    codes = forced
                elif mode == "mixed":
                    codes = torch.where(forced % 3 == 0, -1, forced)
                else:
                    codes = torch.full_like(forced, -1)
                outputs: list[torch.Tensor] = []
                logits: list[torch.Tensor] = []
                for model in (serial, fused):
                    recorded: list[torch.Tensor] = []

                    def record(values: torch.Tensor) -> torch.Tensor:
                        recorded.append(values)
                        return values.argmax(dim=-1)

                    output = model.generate(text, hidden, codes, record)
                    stacked = torch.stack(recorded, dim=1)
                    assert bool(torch.isfinite(stacked).all())
                    assert output.shape == (batch_size, DEPFORMER.steps)
                    assert bool(((output >= 0) & (output < AUDIO_CARD)).all())
                    torch.testing.assert_close(output[codes >= 0], codes[codes >= 0])
                    repeated = model.generate(text, hidden, codes, record)
                    torch.testing.assert_close(repeated, output, atol=0, rtol=0)
                    outputs.append(output)
                    logits.append(stacked)
                if dtype == torch.float32:
                    torch.testing.assert_close(
                        logits[1], logits[0], atol=1e-4, rtol=1e-4
                    )
                    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
                else:
                    pass
                cases.append(
                    {
                        "dtype": str(dtype),
                        "batch_size": batch_size,
                        "mode": mode,
                        "finite": True,
                        "logit_max_difference": (logits[1] - logits[0])
                        .abs()
                        .max()
                        .item(),
                        "code_disagreements": (outputs[1] != outputs[0]).sum().item(),
                        "code_positions": outputs[0].numel(),
                        "same_arm_repeated_codes": True,
                    }
                )
                print(json.dumps(cases[-1]), flush=True)
        del serial, fused
    arguments.output.write_text(
        json.dumps(
            {
                "scope": "full-shape random-weight CUDA mechanics; not checkpoint quality or serving performance",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
