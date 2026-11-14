"""Make every Qwen3-Omni thinker request decode exactly its max_tokens, for synthetic decode regimes.

Enabled by OMNI_SYNTH_IGNORE_EOS=1. A watcher thread waits for the request builders module to load, then wraps
build_sglang_thinker_request so the SGLang request it returns carries ignore_eos, as SemiAnalysis-style decode
regimes need. Nothing else in the request changes; the talker keeps its own codec end of speech.
"""

import os
import sys
import threading
import time

MODULE = "sglang_omni.models.qwen3_omni.request_builders"


def install(module) -> None:
    original = module.build_sglang_thinker_request

    def build_sglang_thinker_request(*args, **kwargs):
        data = original(*args, **kwargs)
        data.req.sampling_params.ignore_eos = True
        return data

    module.build_sglang_thinker_request = build_sglang_thinker_request
    print(
        f"synth_eos_probe: thinker requests ignore eos in pid {os.getpid()}",
        file=sys.stderr,
        flush=True,
    )


def watch() -> None:
    while MODULE not in sys.modules or not hasattr(
        sys.modules[MODULE], "build_sglang_thinker_request"
    ):
        time.sleep(0.05)
    install(sys.modules[MODULE])


if os.environ.get("OMNI_SYNTH_IGNORE_EOS") == "1":
    threading.Thread(target=watch, name="synth-eos-probe", daemon=True).start()
else:
    pass

# note (ratish): Python imports one sitecustomize, so another probe's file named here runs too.
if os.environ.get("OMNI_SITECUSTOMIZE_CHAIN"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "chained_sitecustomize", os.environ["OMNI_SITECUSTOMIZE_CHAIN"]
    )
    chained = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chained)
else:
    pass
