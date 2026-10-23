"""Log Drama 3 request results and scheduler aborts as JSON lines, for the dropped-streams repro.

Loaded as sitecustomize when this directory is on PYTHONPATH and DRAMA3_PROBE_DIR is set; each process appends to <dir>/<pid>.jsonl. Times are CLOCK_MONOTONIC nanoseconds, shared by every process on the host.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time

PROBE_DIR = os.environ.get("DRAMA3_PROBE_DIR")
lock = threading.Lock()


def write(record):
    record["t"] = time.monotonic_ns()
    record["pid"] = os.getpid()
    record["thread"] = threading.current_thread().name
    with lock, open(f"{PROBE_DIR}/{os.getpid()}.jsonl", "a") as sink:
        sink.write(json.dumps(record, default=str) + "\n")


def finished_reason_of(req):
    if req is None or req.finished_reason is None:
        return None
    else:
        return req.finished_reason.to_json()


def patch_request_builders(module):
    adapter_call = module.Drama3ResultAdapter.__call__

    def call(self, request_data):
        req = request_data.req
        sampled = request_data.frame_rows[request_data.prompt_row_count :]
        write(
            {
                "k": "result",
                "rid": None if req is None else req.rid,
                "inputs": str(request_data.stage_payload.request.inputs)[:24],
                "sampled": len(sampled),
                "last_main": sampled[-1][0] if sampled else None,
                "max_new_tokens": request_data.max_new_tokens,
                "output_ids": None if req is None else len(req.output_ids),
                "finished_reason": finished_reason_of(req),
                "finish_reason": request_data.finish_reason,
            }
        )
        return adapter_call(self, request_data)

    module.Drama3ResultAdapter.__call__ = call


def patch_scheduler(module):
    abort = module.OmniScheduler.abort

    def probe_abort(self, request_id, *args, **kwargs):
        running = [] if self.running_batch is None else self.running_batch.reqs
        write(
            {
                "k": "abort",
                "rid": request_id,
                "running": [
                    (req.rid, len(req.output_ids), finished_reason_of(req))
                    for req in running
                ],
            }
        )
        return abort(self, request_id, *args, **kwargs)

    module.OmniScheduler.abort = probe_abort


PATCHES = {
    "sglang_omni.models.fishaudio_drama3.request_builders": patch_request_builders,
    "sglang_omni.scheduling.omni_scheduler": patch_scheduler,
}


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in getattr(self, "seen", ()):
            return None
        else:
            pass
        self.seen = (*getattr(self, "seen", ()), name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        else:
            pass
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                PATCHES[name](module)
            except Exception as exc:
                print(
                    f"drama3_probe patch {name} failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

        spec.loader.exec_module = exec_and_patch
        return spec


if PROBE_DIR:
    os.makedirs(PROBE_DIR, exist_ok=True)
    sys.meta_path.insert(0, PatchOnImport())
else:
    pass
