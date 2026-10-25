"""Review-only mutation: corrupt the batcher's real encode results after execution."""

import torch

from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher

original = Qwen3TTSRefCodeBatcher.encode_waveform


def zero_codes(self, waveform, sample_rate):
    return torch.zeros_like(original(self, waveform, sample_rate))


def pytest_runtest_setup(item):
    if item.name == 'test_reference_code_batcher_replays_captured_buckets_and_encodes_the_rest':
        Qwen3TTSRefCodeBatcher.encode_waveform = zero_codes


def pytest_runtest_teardown(item):
    Qwen3TTSRefCodeBatcher.encode_waveform = original
