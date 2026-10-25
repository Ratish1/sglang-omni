"""Exercise the real GC capture regression, its pre-fix control and GC cleanup."""

import argparse
import contextlib
import gc
import json
from unittest.mock import patch

import pytest

from sglang_omni.models.qwen3_omni.components import audio_layer_graph


@contextlib.contextmanager
def unprotected_capture(*, enable_cudagraph_gc):
    yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unprotected", action="store_true")
    parser.add_argument("--gc-disabled", action="store_true")
    options = parser.parse_args()
    original = gc.isenabled()
    if options.gc_disabled:
        gc.disable()
    before = gc.isenabled()
    try:
        with (
            patch.object(audio_layer_graph, "freeze_gc", unprotected_capture)
            if options.unprotected
            else contextlib.nullcontext()
        ):
            status = pytest.main(
                [
                    "-q",
                    "tests/unit_test/qwen3_omni/test_audio_layer_graph.py::"
                    "test_capture_survives_a_collection_that_frees_dead_graphs",
                ]
            )
        print(
            json.dumps(
                dict(
                    case="gc_capture",
                    unprotected=options.unprotected,
                    pytest_status=int(status),
                    gc_enabled_before=before,
                    gc_enabled_after=gc.isenabled(),
                )
            ),
            flush=True,
        )
    finally:
        if original:
            gc.enable()
        else:
            gc.disable()


if __name__ == "__main__":
    main()
