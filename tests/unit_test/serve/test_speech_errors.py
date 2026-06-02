# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.serve.speech_errors import bad_request, speech_error_response


def test_speech_error_response_uses_openai_envelope() -> None:
    response = speech_error_response(bad_request("bad speed", param="speed"))

    assert response.status_code == 400
    assert response.body == (
        b'{"error":{"message":"bad speed","type":"BadRequestError",'
        b'"param":"speed","code":400}}'
    )
