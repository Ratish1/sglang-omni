# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(autouse=True)
def _strict_mem_check_env_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", raising=False)
    monkeypatch.delenv("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", raising=False)
