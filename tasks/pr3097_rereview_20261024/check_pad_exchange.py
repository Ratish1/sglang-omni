"""Check shared graph-padding slots against independent live-row state transitions."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

from sglang_omni.kernels import codec_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("history", "overlap"))
    parser.add_argument("--previous", action="store_true")
    parser.add_argument("--distinct", action="store_true")
    parser.add_argument("--cold", action="store_true")
    arguments = parser.parse_args()
    production = codec_state
    if arguments.previous:
        source = subprocess.run(
            [
                "git",
                "show",
                "837251cff78b7b2c6b3f08a36fad3fc0771c014f:sglang_omni/kernels/codec_state.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        path = Path("/data/review3097oct24/previous_codec_state.py")
        path.write_text(source)
        specification = importlib.util.spec_from_file_location(
            "previous_codec_state", path
        )
        assert specification is not None and specification.loader is not None
        production = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = production
        specification.loader.exec_module(production)
    else:
        pass
    torch.manual_seed(3097)
    pad_slot = 4
    slots = [0, 1, 2, 3] if arguments.distinct else [pad_slot, 1, pad_slot, pad_slot]
    slot_index = torch.tensor(slots, device="cuda")
    pool_samples = 54 if arguments.operation == "history" else 4
    pool = torch.randn(5, pool_samples, 200, device="cuda", dtype=torch.bfloat16)
    expected_pool = pool.clone()
    if arguments.operation == "history":
        fresh_samples = 1
        values = torch.randn(
            4, pool_samples + fresh_samples + 2, 200, device="cuda", dtype=pool.dtype
        )
        expected_values = values.clone()
        for row, slot in enumerate(slots):
            if slot == pad_slot or arguments.cold:
                expected_values[row, :pool_samples].zero_()
            else:
                expected_values[row, :pool_samples].copy_(pool[slot])
            expected_values[row, pool_samples + fresh_samples :].zero_()
            if slot != pad_slot:
                expected_pool[slot].copy_(
                    expected_values[row, fresh_samples : fresh_samples + pool_samples]
                )
            else:
                pass
        if arguments.previous:
            production.history_exchange(
                values, pool, slot_index, fresh_samples, arguments.cold
            )
        else:
            production.history_exchange(
                values,
                pool,
                slot_index,
                fresh_samples,
                arguments.cold,
                pad_slot=pad_slot,
            )
    else:
        emitted_samples = 4
        values = torch.randn(
            4, emitted_samples + pool_samples, 200, device="cuda", dtype=pool.dtype
        )
        expected_values = values.clone()
        for row, slot in enumerate(slots):
            if slot != pad_slot:
                if not arguments.cold:
                    expected_values[row, :pool_samples] += pool[slot]
                else:
                    pass
                expected_pool[slot].copy_(values[row, emitted_samples:])
            else:
                pass
        if arguments.previous:
            production.overlap_exchange(
                values, pool, slot_index, emitted_samples, arguments.cold
            )
        else:
            production.overlap_exchange(
                values,
                pool,
                slot_index,
                emitted_samples,
                arguments.cold,
                pad_slot=pad_slot,
            )
    torch.cuda.synchronize()
    assert torch.equal(
        values, expected_values
    ), "padding must take zero state; live output must match its own slot"
    assert torch.equal(pool, expected_pool), "only live slots may change"
    print(
        f"PASS {arguments.operation}, {slots}, cold={arguments.cold}, previous={arguments.previous}",
        flush=True,
    )


if __name__ == "__main__":
    main()
