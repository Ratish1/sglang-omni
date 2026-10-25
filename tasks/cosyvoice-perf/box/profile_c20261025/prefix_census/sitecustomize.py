# SPDX-License-Identifier: Apache-2.0
"""#3132 review F1 (10-25): the prefix graph steps a server replays. Put this directory first on PYTHONPATH with
PREFIX_CENSUS=<dir>. Every PrefixCudaGraphRunner.run appends one JSON line to <dir>/census.jsonl: each row's new frames,
each row's cached prefix (the conditional twin's committed frames before the step), the step's frames and the tier the
runner picks (null above its largest). The queue appends {"cell": name} before each benchmark cell. No runtime code
changes."""

import bisect
import importlib.abc
import importlib.util
import json
import os
import sys

TARGET = "sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph"


def patch(module) -> None:
    path = os.path.join(os.environ["PREFIX_CENSUS"], "census.jsonl")
    run = module.PrefixCudaGraphRunner.run

    def run_recorded(self, *, new_frames, caches, **inputs):
        frames = sum(new_frames)
        tier = bisect.bisect_left(self.tier_frames, frames)
        record = {
            "new_frames": [int(count) for count in new_frames],
            "prefix_frames": [int(pair[0].committed_frames) for pair in caches],
            "frames": int(frames),
            "tier": self.tier_frames[tier] if tier < len(self.tier_frames) else None,
        }
        with open(path, "a") as census:
            census.write(json.dumps(record) + "\n")
        return run(self, new_frames=new_frames, caches=caches, **inputs)

    module.PrefixCudaGraphRunner.run = run_recorded
    print(f"prefix census patched {module.__name__} into {path}", file=sys.stderr, flush=True)


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name != TARGET or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("PREFIX_CENSUS"):
    sys.meta_path.insert(0, PatchOnImport())
