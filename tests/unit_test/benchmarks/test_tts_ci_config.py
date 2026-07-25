# SPDX-License-Identifier: Apache-2.0

from tests.test_model.tts_ci_config import TTS_CI_PRESETS


def test_moss_tts_ci_uses_reproducible_sampling() -> None:
    assert TTS_CI_PRESETS["moss"].model.seed == 12345
    assert TTS_CI_PRESETS["higgs"].model.seed is None
