from types import SimpleNamespace

from sglang_omni.utils.hf import drop_mrope_keys, use_text_rope


def test_use_text_rope_drops_only_mrope_parameters() -> None:
    original = {
        "rope_type": "default",
        "rope_theta": 1_000_000,
        "interleaved": True,
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
    }
    config = SimpleNamespace(rope_parameters=original, rope_scaling=original)

    use_text_rope(config)

    expected = {"rope_type": "default", "rope_theta": 1_000_000}
    assert config.rope_parameters == expected
    assert config.rope_scaling == expected
    assert original["mrope_section"] == [24, 20, 20]


def test_use_text_rope_leaves_a_text_rope_config_untouched() -> None:
    text = {"rope_type": "default", "rope_theta": 1_000_000}
    config = SimpleNamespace(rope_parameters=text, rope_scaling=None)

    use_text_rope(config)

    assert config.rope_parameters is text
    assert config.rope_scaling is None


def test_drop_mrope_keys_returns_a_new_mapping() -> None:
    original = {"rope_theta": 10_000, "mrope_section": [16, 24, 24]}

    assert drop_mrope_keys(original) == {"rope_theta": 10_000}
    assert original == {"rope_theta": 10_000, "mrope_section": [16, 24, 24]}
