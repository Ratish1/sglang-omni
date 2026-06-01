# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible error response validation."""

from __future__ import annotations

import json


def is_openai_error_response(body: str) -> bool:
    if not body.strip() or body.lstrip().startswith("<"):
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    message = error.get("message")
    if not isinstance(message, str) or not message.strip():
        return False
    error_type = error.get("type")
    if error_type is not None and (
        not isinstance(error_type, str) or not error_type.strip()
    ):
        return False
    code = error.get("code")
    if isinstance(code, bool) or (
        code is not None and not isinstance(code, (int, str))
    ):
        return False
    param = error.get("param")
    return param is None or isinstance(param, str)
