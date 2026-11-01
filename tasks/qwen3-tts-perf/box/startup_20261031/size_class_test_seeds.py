# Block 75: the size-class test's fp32 bound across seeds, then the review's mutation. Runs
# test_compiled_decode_shares_a_program_per_size_class_and_tracks_fp32 from the tree on PYTHONPATH with torch.manual_seed
# pinned to each seed. In the sweep every (compiled, eager) error pair the test compares is recorded and the compiled
# value handed back as zero, so no assertion stops the record: per cohort the waveform pair, then the retained-state pair.
# Then the real test at seed 33 and with the final conv zeroed only while Dynamo traces, which the test must reject.
# usage: python3 size_class_test_seeds.py <tree> <first seed> <seed count>
import sys
import traceback

import pytest
import torch

tree, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, tree)
sys.path.insert(0, f"{tree}/tests/unit_test/qwen3_tts")
import test_incremental_codec as module  # noqa: E402

from sglang_omni.models.qwen3_tts import incremental_codec  # noqa: E402

original_seed = torch.manual_seed
original_error = module.relative_error
recorded = []


def recording_error(actual, expected):
    value = original_error(actual, expected)
    recorded.append(value)
    return 0.0 if len(recorded) % 2 == 1 else value


def run(seed, measure):
    recorded.clear()
    module.relative_error = recording_error if measure else original_error
    torch.manual_seed = lambda _: original_seed(seed)
    monkeypatch = pytest.MonkeyPatch()
    try:
        module.test_compiled_decode_shares_a_program_per_size_class_and_tracks_fp32(monkeypatch)
        outcome = "passed"
    except AssertionError:
        outcome = "FAILED " + " | ".join(line.strip() for line in traceback.format_exc().splitlines()[-3:])[:300]
    finally:
        monkeypatch.undo()
        torch.manual_seed = original_seed
        module.relative_error = original_error
    return outcome, list(zip(recorded[0::2], recorded[1::2]))


labels = ("cohort 2x(8,4,3,8,4) waveform", "cohort 2x(8,4,3,8,4) state", "cohort 1x(3,3,3) waveform", "cohort 1x(3,3,3) state")
worst = {label: 0.0 for label in labels}
for seed in range(first, first + count):
    outcome, pairs = run(seed, measure=True)
    cells = []
    for label, (compiled, eager) in zip(labels, pairs):
        ratio = compiled / eager
        worst[label] = max(worst[label], ratio)
        cells.append(f"{label} {compiled:.2e}/{eager:.2e} = {ratio:.3f}")
    print(f"seed {seed}: {outcome}; " + "; ".join(cells), flush=True)
print(f"seeds {first} to {first + count - 1}, worst compiled/eager ratio: " + "; ".join(f"{label} {value:.3f}" for label, value in worst.items()))

outcome, pairs = run(33, measure=False)
print(f"real test, seed 33: {outcome}")
direct_conv = incremental_codec.clamped_single_output_conv


def silent_when_traced(padded, conv, length):
    output = direct_conv(padded, conv, length)
    if torch.compiler.is_dynamo_compiling():
        return torch.zeros_like(output)
    else:
        return output


incremental_codec.clamped_single_output_conv = silent_when_traced
outcome, pairs = run(33, measure=False)
print(f"mutation, final conv zeroed only in the trace: {outcome}; pairs {pairs}")
