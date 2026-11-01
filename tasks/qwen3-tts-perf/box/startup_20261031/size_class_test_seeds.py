# Block 75: the size-class test's fp32 bound across seeds, then the review's mutation. Runs
# test_compiled_decode_shares_a_program_per_size_class_and_tracks_fp32 from the tree on PYTHONPATH with torch.manual_seed
# pinned to each seed, recording each (compiled, eager) error pair the test compares; then once with the final conv
# zeroed only while Dynamo traces, which the test must reject.
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
    return value


def run(seed):
    recorded.clear()
    torch.manual_seed = lambda _: original_seed(seed)
    monkeypatch = pytest.MonkeyPatch()
    try:
        module.test_compiled_decode_shares_a_program_per_size_class_and_tracks_fp32(monkeypatch)
        outcome = "passed"
    except AssertionError as error:
        outcome = f"FAILED {str(error).splitlines()[0][:120] if str(error) else traceback.format_exc().splitlines()[-2][:120]}"
    finally:
        monkeypatch.undo()
        torch.manual_seed = original_seed
    pairs = list(zip(recorded[0::2], recorded[1::2]))
    return outcome, pairs


module.relative_error = recording_error
worst_wave, worst_state = 0.0, 0.0
for seed in range(first, first + count):
    outcome, pairs = run(seed)
    ratios = [compiled / eager for compiled, eager in pairs]
    state_keys = (len(pairs) - 2) // 2
    wave_indices = {0, 1 + state_keys}
    wave_ratios = [ratios[index] for index in sorted(wave_indices) if index < len(ratios)]
    state_ratios = [ratio for index, ratio in enumerate(ratios) if index not in wave_indices]
    worst_wave = max([worst_wave, *wave_ratios])
    worst_state = max([worst_state, *state_ratios])
    print(f"seed {seed}: {outcome}; waveform ratios {' '.join(f'{ratio:.3f}' for ratio in wave_ratios)} "
          f"(errors {' '.join(f'{pairs[index][0]:.2e}/{pairs[index][1]:.2e}' for index in sorted(wave_indices) if index < len(pairs))}); "
          f"state ratios over {len(state_ratios)} keys max {max(state_ratios, default=0.0):.3f} "
          f"median {sorted(state_ratios)[len(state_ratios) // 2] if state_ratios else 0.0:.3f}", flush=True)
print(f"seeds {first} to {first + count - 1}: worst ratio, waveforms {worst_wave:.3f}, state rows {worst_state:.3f}")

direct_conv = incremental_codec.clamped_single_output_conv


def silent_when_traced(padded, conv, length):
    output = direct_conv(padded, conv, length)
    if torch.compiler.is_dynamo_compiling():
        return torch.zeros_like(output)
    else:
        return output


incremental_codec.clamped_single_output_conv = silent_when_traced
outcome, pairs = run(33)
print(f"mutation, final conv zeroed only in the trace: {outcome}; first pair {pairs[0] if pairs else None}")
