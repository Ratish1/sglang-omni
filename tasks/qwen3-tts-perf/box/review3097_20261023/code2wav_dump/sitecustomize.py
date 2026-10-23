"""Box only: every request's talker codes and served waveform saved when code2wav finishes it, for
omni_fp32_probe.py (the pr2800 program's code2wav_dump hook).

Each stage process imports sitecustomize at start while this directory is on PYTHONPATH. Once the code2wav scheduler
module loads, Code2WavScheduler.final_result_data is wrapped: before it runs, the request's code frames (state.chunks)
and the audio parts it emitted (state.audio_parts) are saved under CODE2WAV_DUMP_DIR as <request id>.pt; the call
then runs unchanged.
"""

import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"


def install(module):
    import numpy as np
    import torch

    original = module.Code2WavScheduler.final_result_data
    directory = os.environ["CODE2WAV_DUMP_DIR"]
    os.makedirs(directory, exist_ok=True)

    def final_result_data(self, request_id, payload, state):
        frames = [chunk.reshape(-1).to("cpu") for chunk in state.chunks]
        audio = [np.asarray(part, dtype=np.float32) for part in state.audio_parts]
        torch.save(
            {
                "request_id": request_id,
                "codes": torch.stack(frames) if frames else torch.zeros(0, dtype=torch.long),
                "audio": np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32),
                "sample_rate": self.sample_rate,
                "stream_enabled": state.stream_enabled,
            },
            f"{directory}/{request_id}.pt",
        )
        return original(self, request_id, payload, state)

    module.Code2WavScheduler.final_result_data = final_result_data


class DumpAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_install
        return spec


if os.environ.get("CODE2WAV_DUMP_DIR"):
    sys.meta_path.insert(0, DumpAfterLoad())
else:
    pass
