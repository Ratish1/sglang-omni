"""Review counterfactual: ring row offsets ignore the requested arena slots."""

from dataclasses import replace

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena


def pytest_configure(config):
    original = Qwen3TTSCodecStateArena.gather_by_index

    def wrong_slot_rows(self, index):
        state = original(self, index)
        if state.kv_rings is not None:
            state.kv_rings = replace(
                state.kv_rings,
                slot_rows=torch.arange(index.numel(), device=index.device)
                * self.ring_frames,
            )
        return state

    Qwen3TTSCodecStateArena.gather_by_index = wrong_slot_rows
