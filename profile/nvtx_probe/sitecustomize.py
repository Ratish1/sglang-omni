"""Loaded by the interpreter when this directory is first on PYTHONPATH.

Arms the omni NVTX probe (see omni_nvtx_probe.py) when OMNI_NVTX_PROBE=1
and applies OMNI_SWITCH_INTERVAL when set; otherwise a no-op. Stage processes use the spawn context, so each one runs
this file and installs its own hook. Python imports only one sitecustomize,
so this one then hands over to any other sitecustomize further down
sys.path (a venv's own, for example) to keep it in force.
"""

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("OMNI_NVTX_PROBE") == "1":
    import omni_nvtx_probe

    omni_nvtx_probe.arm()

# GIL switch-interval sweep (profile/README.md section 5): seconds, default
# is CPython's 0.005. Applied in every process that imports this file.
_interval = os.environ.get("OMNI_SWITCH_INTERVAL")
if _interval:
    sys.setswitchinterval(float(_interval))

_others = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _others)
if (
    _spec is not None
    and _spec.origin
    and os.path.abspath(_spec.origin) != os.path.abspath(__file__)
):
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
