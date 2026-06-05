# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.scheduling.request_lifecycle import attach_sglang_req_compat


def test_attach_sglang_req_compat_sets_explicit_none_values() -> None:
    req = SimpleNamespace()

    attach_sglang_req_compat(
        req,
        tokenizer=None,
        codec_suppress_tokens=None,
        input_embeds_are_projected=False,
    )

    assert req.tokenizer is None
    assert req._codec_suppress_tokens is None
    assert req._input_embeds_are_projected is False


def test_attach_sglang_req_compat_leaves_unset_attrs_untouched() -> None:
    req = SimpleNamespace()

    attach_sglang_req_compat(req, codec_suppress_tokens=None)

    assert req._codec_suppress_tokens is None
    assert not hasattr(req, "tokenizer")
    assert not hasattr(req, "_input_embeds_are_projected")
