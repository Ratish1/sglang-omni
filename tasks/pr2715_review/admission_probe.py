"""Check that the native solve-parity test detects incorrect timestep-table selection."""

import json

from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT
from tests.unit_test.fun_cosyvoice3.test_packed_dit import (
    test_packed_solve_matches_the_padded_solve_per_row,
)


original = PackedDiT.build_modulations
outcomes = {}

for mutation in ("wrong_block_step", "wrong_final_step"):
    def changed_build(self, time_span, dtype):
        original(self, time_span, dtype)
        if mutation == "wrong_block_step":
            self.block_modulations = self.block_modulations[1:] + self.block_modulations[:1]
        else:
            self.final_modulations = self.final_modulations[1:] + self.final_modulations[:1]

    PackedDiT.build_modulations = changed_build
    try:
        test_packed_solve_matches_the_padded_solve_per_row(streaming=True)
    except AssertionError:
        outcomes[mutation] = "detected by actual native solve comparison"
    else:
        raise AssertionError(f"the existing test did not detect {mutation}")
    finally:
        PackedDiT.build_modulations = original

test_packed_solve_matches_the_padded_solve_per_row(streaming=True)
test_packed_solve_matches_the_padded_solve_per_row(streaming=False)
outcomes["restored"] = "both native solve comparisons pass"
print(json.dumps(outcomes), flush=True)
