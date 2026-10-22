"""Box only: omni's audio loader resamples as SGLang's load_audio does, for resample_quality_check.py.

Every stage process imports sitecustomize at start while this directory is on PYTHONPATH. Once
sglang_omni.preprocessing.audio loads, its resample_linear (np.interp, no anti-alias filter) is replaced by
torchaudio.functional.resample, SGLang's fallback resampler (srt/utils/common.py load_audio), so every request's
audio reaches the thinker band limited.
"""

import importlib.abc
import importlib.machinery
import sys

TARGET = "sglang_omni.preprocessing.audio"


def band_limited(module):
    import numpy as np
    import torch
    import torchaudio

    def resample_linear(audio, orig_sr, target_sr):
        if orig_sr == target_sr or audio.size == 0:
            return audio.astype(np.float32, copy=False)
        else:
            pass
        samples = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return torchaudio.functional.resample(
            samples, orig_freq=int(orig_sr), new_freq=int(target_sr)
        ).numpy()

    module.resample_linear = resample_linear


class BandLimitedAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            band_limited(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, BandLimitedAfterLoad())
