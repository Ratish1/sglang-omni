"""C1's measurement patch, and the profiling probe when its switches are on (cosy_profile_probe after this
directory on PYTHONPATH), since only the first sitecustomize on the path runs."""

import os

import c1_impl  # noqa: F401

if os.environ.get("OMNI_PIPE_NVTX") == "1" or os.environ.get("OMNI_LOOP_PROBE") == "1":
    import cosy_impl  # noqa: F401
    import loop_impl  # noqa: F401
