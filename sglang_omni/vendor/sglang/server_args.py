"""Small compatibility helpers around SGLang ``ServerArgs``."""

from __future__ import annotations

from typing import Any

from sglang.srt.server_args import get_global_server_args


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Apply an audited post-resolution ServerArgs mutation.

    SGLang 0.5.16 requires real ``ServerArgs`` instances to mutate through
    ``override()`` when strict mutation checking is enabled. The fallback keeps
    lightweight test doubles and third-party ServerArgs-like objects working.
    """
    override = getattr(server_args, "override", None)
    if callable(override):
        override(source, **fields)
        return
    for name, value in fields.items():
        setattr(server_args, name, value)


__all__ = ["get_global_server_args", "override_server_args"]
