"""Profiling boots only: loads cosy_call_ledger in every Python process of the
serve when COSY_CALL_LEDGER_DIR is set, after running any sitecustomize the
environment already has. Put this directory on PYTHONPATH for a profiling boot
and never for a census boot.
"""

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _run_shadowed_sitecustomize() -> None:
    search = [
        entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != _HERE
    ]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


_run_shadowed_sitecustomize()

if os.environ.get("COSY_CALL_LEDGER_DIR"):
    import cosy_call_ledger

    cosy_call_ledger.install()
