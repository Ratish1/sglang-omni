"""Check omni_pipeline_nvtx's targets against a tree without serving it.

Imports every module the probe patches from the tree on PYTHONPATH, runs each patch with the range wrappers replaced
by recorders, and reports: a patch that raises, a name the tree lacks (range_present's "has no"), a label whose
parameters do not bind to the call the wrapped function receives (the probe would then fall back to the function's
own name and the census would lose the range's kind), and a hand-written wrapper whose call into the original does
not bind to the original's signature.

usage: PYTHONPATH=<tree> python3 probe_targets_check.py
"""

import contextlib
import importlib
import inspect
import io
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "omni_pipeline_nvtx")
)
import sitecustomize as probe  # noqa: E402

recorded = []


def recording(kind):
    def record(fn, name):
        recorded.append((kind, fn, name))
        return fn

    return record


probe.ranged = recording("range")
probe.bracketed_async = recording("bracket")
probe.marked_async = recording("mark")


def sample_call(fn):
    """Positional sentinels for every required positional parameter, keywords for every required keyword-only one."""
    args = []
    kwargs = {}
    for parameter in inspect.signature(fn).parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        elif parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(object())
        elif parameter.kind == parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = object()
        else:
            pass
    return args, kwargs


def positional_names(fn):
    return [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


problems = []
# note: the hand-written wrappers call their originals with these arguments; checked before any patch replaces them
for module_name, owner, attribute, call in (
    (
        "sglang_omni.scheduling.omni_scheduler",
        "OmniScheduler",
        "_run_batch",
        ("self", "batch", "pp_proxy_tensors"),
    ),
    (
        "sglang_omni.scheduling.omni_scheduler",
        "OmniScheduler",
        "run_batch_launch",
        ("self", "batch"),
    ),
    (
        "sglang_omni.scheduling.omni_scheduler",
        "OmniScheduler",
        "enqueue_built_request",
        ("self", "payload"),
    ),
    (
        "sglang_omni.models.qwen3_omni.talker_scheduler",
        "QwenTalkerScheduler",
        "note_chunk_wait",
        ("self", "batch"),
    ),
    (
        "sglang_omni.models.qwen3_omni.components.talker",
        "Qwen3OmniTalker",
        "reuse_decode_buffers",
        ("self", "requests"),
    ),
    ("sglang_omni.pipeline.stage.runtime", "Stage", "start", ("self",)),
    ("sglang_omni.pipeline.coordinator", "Coordinator", "start", ("self",)),
    ("sglang_omni.scheduling.message", None, "put_messages", ("outbox", "messages")),
):
    module = importlib.import_module(module_name)
    holder = getattr(module, owner) if owner else module
    original = inspect.unwrap(getattr(holder, attribute))
    try:
        inspect.signature(original).bind(*[object()] * len(call))
    except TypeError as exc:
        problems.append(
            f"{module_name}: {owner}.{attribute} no longer takes {call}: {exc}"
        )
    else:
        pass

for module_name, patch in probe.PATCHES.items():
    if module_name.startswith("torch"):
        continue
    else:
        pass
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        problems.append(f"{module_name}: import failed {exc!r}")
        continue
    before = len(recorded)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            patch(module)
        except Exception as exc:
            problems.append(f"{module_name}: patch raised {exc!r}")
    for line in stderr.getvalue().splitlines():
        problems.append(f"{module_name}: {line}")
    for kind, fn, name in recorded[before:]:
        target = inspect.unwrap(fn)
        label = getattr(name, "__name__", "")
        if label == "<lambda>" and "fixed" not in getattr(name, "__qualname__", ""):
            args, kwargs = sample_call(target)
            try:
                inspect.signature(name).bind(*args, **kwargs)
            except TypeError as exc:
                problems.append(
                    f"{module_name}: label for {target.__qualname__} does not bind: {exc}"
                )
            label_names = positional_names(name)
            target_names = positional_names(target)[: len(label_names)]
            if label_names != target_names:
                problems.append(
                    f"{module_name}: label for {target.__qualname__} names {label_names}, "
                    f"the function takes {positional_names(target)}"
                )
            else:
                pass
        else:
            pass
    print(f"{module_name}: {len(recorded) - before} targets")

print(f"\n{len(problems)} problems")
for problem in problems:
    print(f"  {problem}")
