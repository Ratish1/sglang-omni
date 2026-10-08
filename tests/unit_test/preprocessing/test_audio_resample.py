# SPDX-License-Identifier: Apache-2.0
"""Band-limited resampling keeps nothing from above the target Nyquist; linear resampling aliases it."""

import asyncio
import base64

import numpy as np
import pytest
from scipy.io import wavfile

from sglang_omni.preprocessing import audio

SOURCE_RATE = 24000
TARGET_RATE = 16000


def tone(frequency: float, sample_rate: int = SOURCE_RATE) -> np.ndarray:
    times = np.arange(sample_rate) / sample_rate
    return (0.5 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_band_limited_resampling_drops_a_tone_above_the_target_nyquist() -> None:
    # 10 kHz is above 16 kHz's 8 kHz Nyquist; without a filter it folds to 6 kHz.
    above = tone(10000)

    assert rms(audio.resample_linear(above, SOURCE_RATE, TARGET_RATE)) > 0.1
    assert rms(audio.resample_band_limited(above, SOURCE_RATE, TARGET_RATE)) < 0.01


def test_band_limited_resampling_keeps_an_in_band_tone() -> None:
    resampled = audio.resample_band_limited(tone(1000), SOURCE_RATE, TARGET_RATE)
    ideal = tone(1000, TARGET_RATE)

    assert resampled.dtype == np.float32
    assert resampled.shape == ideal.shape
    assert rms(resampled[200:-200] - ideal[200:-200]) < 1e-4


def test_band_limited_resampling_passes_empty_and_same_rate_audio_through() -> None:
    empty = np.zeros(0, dtype=np.float32)
    samples = tone(1000, TARGET_RATE)

    assert audio.resample_band_limited(empty, SOURCE_RATE, TARGET_RATE).size == 0
    assert np.array_equal(
        audio.resample_band_limited(samples, TARGET_RATE, TARGET_RATE), samples
    )


@pytest.mark.parametrize("band_limited", [False, True])
def test_every_audio_loader_band_limits_only_when_asked(tmp_path, band_limited) -> None:
    path = tmp_path / "tone.wav"
    wavfile.write(path, SOURCE_RATE, tone(10000))
    data_url = "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()
    media_io = audio.AudioMediaIO(target_sr=TARGET_RATE, band_limited=band_limited)

    loaded = [
        audio.load_audio_path(path, target_sr=TARGET_RATE, band_limited=band_limited),
        media_io.load_bytes(path.read_bytes())[0],
        media_io.load_file(path)[0],
        *asyncio.run(
            audio.ensure_audio_list_async(
                [str(path), data_url], target_sr=TARGET_RATE, band_limited=band_limited
            )
        ),
    ]

    assert [samples.shape for samples in loaded] == [(TARGET_RATE,)] * 5
    assert [rms(samples) < 0.01 for samples in loaded] == [band_limited] * 5
