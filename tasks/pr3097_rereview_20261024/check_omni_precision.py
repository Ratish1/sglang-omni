"""Compare the old/new activation kernel in trained Omni slot decodes on identical codes."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni.components import incremental_codec
from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.platforms import current_platform


@torch.inference_mode()
def main() -> None:
    folder = Path("/data/review3097oct24")
    torch.manual_seed(3097)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = load_code2wav_model(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", device="cuda:0", dtype="float32"
    )
    model = copy.deepcopy(reference).bfloat16()
    model.use_fused_transformer(
        current_platform.get_joint_rope_inplace_kernel(), max_positions=128
    )
    codec = Qwen3OmniIncrementalCodec(model)
    pools = {
        name: CodecStatePool(
            codec.spec,
            num_slots=3,
            capture_slots=0,
            ring_frames=codec.spec.retained_context + 10,
            device=torch.device("cuda", 0),
            dtype=torch.bfloat16,
        )
        for name in ("previous", "head")
    }
    source = subprocess.run(
        [
            "git",
            "show",
            "a248446c7802cdca8525a72bfd7e705b0b60fe9a:sglang_omni/kernels/bias_snake_beta.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = folder / "previous_omni_bias.py"
    path.write_text(source)
    specification = importlib.util.spec_from_file_location("previous_omni_bias", path)
    assert specification is not None and specification.loader is not None
    previous = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = previous
    specification.loader.exec_module(previous)
    current_kernel = incremental_codec.fused_bias_snake_beta
    generator = torch.Generator(device="cuda:0").manual_seed(3097)
    history: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    positions = [0, 0]
    records = []
    for step, (slots, frames) in enumerate(
        (([0, 1], 10), ([1], 1), ([1, 0], 10), ([0], 3))
    ):
        codes = torch.randint(
            model.config.codebook_size,
            (len(slots), model.config.num_quantizers, frames),
            device="cuda:0",
            generator=generator,
        )
        outputs = {}
        for name, kernel in (
            ("previous", previous.fused_bias_snake_beta),
            ("head", current_kernel),
        ):
            incremental_codec.fused_bias_snake_beta = kernel
            outputs[name] = codec.decode_slots(
                codes,
                pools[name],
                torch.tensor(slots, device="cuda"),
                first_decode=step == 0,
            ).clone()
        incremental_codec.fused_bias_snake_beta = current_kernel
        expected_rows = []
        for row, slot in enumerate(slots):
            history[slot].append(codes[row : row + 1])
            whole = Qwen3OmniMoeCode2Wav.forward(
                reference, torch.cat(history[slot], dim=-1)
            )
            expected_rows.append(whole[..., -outputs["head"].shape[-1] :])
            positions[slot] += frames
        expected = torch.cat(expected_rows)
        record = {"slots": slots, "frames": frames, "snr_to_fp32_db": {}}
        for name, output in outputs.items():
            assert output.shape == expected.shape
            assert torch.isfinite(output).all()
            assert pools[name].frame_positions[:2].tolist() == positions
            signal = expected.square().sum()
            error = (output.float() - expected).square().sum()
            record["snr_to_fp32_db"][name] = float(
                10 * torch.log10(signal / error.clamp_min(1e-30))
            )
        record["gain_db"] = (
            record["snr_to_fp32_db"]["head"] - record["snr_to_fp32_db"]["previous"]
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    (folder / "trained-omni-precision.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(
        "Trained Omni kernel comparison complete; generated code IDs, not a WER/SIM evaluation",
        flush=True,
    )


if __name__ == "__main__":
    main()
