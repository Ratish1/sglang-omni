"""Calls one test function of a branch's test file against another tree's source, which comes first on the path; a
pytest fixture argument named tmp_path or monkeypatch is supplied. Prints PASSED, or FAILED with the failing line.

usage: PYTHONPATH=TREE python3 prefix_call.py TEST_FILE TEST_NAME
"""

import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

import pytest

import sglang_omni

print(f"sglang_omni from {sglang_omni.__file__}")
spec = importlib.util.spec_from_file_location("branch_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
test = getattr(module, sys.argv[2])
arguments = test.__code__.co_varnames[: test.__code__.co_argcount]
monkeypatch = pytest.MonkeyPatch()
try:
    with tempfile.TemporaryDirectory() as directory:
        supplied = {"tmp_path": Path(directory), "monkeypatch": monkeypatch}
        test(**{name: supplied[name] for name in arguments})
except Exception:
    print("FAILED at:")
    traceback.print_exc(limit=-1, file=sys.stdout)
else:
    print("PASSED")
finally:
    monkeypatch.undo()
