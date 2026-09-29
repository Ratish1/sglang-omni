# SPDX-License-Identifier: Apache-2.0
"""c1_compile_startup.py with every FX graph cache lookup traced: the key, the
SymInt hints it evaluates, each stored entry's guard and the verdict.

    python c7_fx_guard_trace.py --out startup.json   (c1_compile_startup's args)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from torch._inductor import codecache

original = codecache.FxGraphCache.find_guarded_entry.__func__


def find_guarded_entry(cls, key, local, remote_cache, evaluate_guards, hints):
    graph, content, info = original(
        cls, key, local, remote_cache, evaluate_guards, hints
    )
    candidates = [
        candidate.guards_expr
        for candidate, _, _ in cls.iterate_over_candidates(local, remote_cache, key)
    ]
    verdicts = [
        bool(evaluate_guards(expr, hints)) if expr else True for expr in candidates
    ]
    print(
        f"FXTRACE key={key[:12]} hints={hints} status={info.get('cache_status_detailed')} "
        f"entries={len(candidates)} verdicts={verdicts} guards={candidates[:1]}",
        flush=True,
    )
    return graph, content, info


codecache.FxGraphCache.find_guarded_entry = classmethod(find_guarded_entry)
sys.argv[0] = str(Path(__file__).with_name("c1_compile_startup.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
