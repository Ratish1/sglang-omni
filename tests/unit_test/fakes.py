# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles."""

from __future__ import annotations

from types import SimpleNamespace


class FakeServerArgs(SimpleNamespace):
    """ServerArgs double exposing the 0.5.16 override() mutation entry point."""

    def override(self, source: str, **fields: object) -> None:
        del source
        for name, value in fields.items():
            setattr(self, name, value)
