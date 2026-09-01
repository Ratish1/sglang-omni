# SPDX-License-Identifier: Apache-2.0
"""Per step attribution of an AR stage's scheduler loop from a with_stack trace.

Usage:
    python trace_ingest.py talker_ar_pid<pid>_rank0.trace.json talker.pkl
    python trace_steps.py talker.pkl [--other code2wav.pkl --other thinker.pkl]

Steps are the run_batch calls on the scheduler thread (the thread with the
most python_function events). A cycle is the interval from one run_batch
start to the next. Phases are split at the largest gap between run_batch
starts, which separates the c1 and c16 benches of _traces_run. The batch
size per decode step is read from the DtoH memcpy of the sampled ids
(8 bytes per row). --other pickles (ingested with trace_ingest.py from the
other processes' traces, same host clock) give the GPU time of other
processes inside each step window, which is what stalls the stream syncs.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import pickle
import statistics as st

RUN_BATCH = "omni_scheduler.py(1354): run_batch"
KEYS = [
    "base.py(406): _build_forward_batch",
    "talker_model_runner.py(56): before_decode",
    "talker.py(1056): prepare_decode_buffers",
    "model_worker.py(257): forward_batch_generation",
    "decode_cuda_graph_runner.py(1386): execute",
    "full_cuda_graph_backend.py(144): replay",
    "decode_cuda_graph_runner.py(1240): load_batch",
    "base.py(528): _finalize",
    "base.py(612): _publish_next_tokens",
    "talker_model_runner.py(105): post_decode",
    "talker_model_runner.py(350): _write_feedback_buffers",
    "forward_batch_info.py(703): init_new",
]
RT = (
    "cudaGraphLaunch",
    "cudaEventSynchronize",
    "cudaStreamSynchronize",
    "cudaMemcpyAsync",
    "cudaLaunchKernel",
)


def short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def load(path):
    d = pickle.load(open(path, "rb"))
    return d, {v: k for k, v in d["names"].items()}


def build_steps(d, inv):
    pf = d["cols"]["python_function"]
    name, tid, ts, dur = pf["name"], pf["tid"], pf["ts"], pf["dur"]
    sched = collections.Counter(tid).most_common(1)[0][0]
    idx = sorted((i for i in range(len(name)) if tid[i] == sched), key=lambda i: ts[i])
    rb = [i for i in idx if inv[name[i]].endswith(RUN_BATCH)]
    if not rb:
        raise SystemExit("no run_batch frames on the scheduler thread")
    steps = [
        dict(
            ts=ts[i],
            dur=dur[i],
            b=collections.defaultdict(float),
            rt=collections.defaultdict(lambda: [0.0, 0]),
            evsync=0.0,
        )
        for i in rb
    ]
    k = 0
    for i in idx:
        while k < len(steps) and ts[i] >= steps[k]["ts"] + steps[k]["dur"]:
            k += 1
        if k < len(steps) and ts[i] >= steps[k]["ts"]:
            nm = short(inv[name[i]])
            if nm in KEYS:
                steps[k]["b"][nm] += dur[i]
            if "synchronize of Event" in nm:
                steps[k]["evsync"] += dur[i]
    rt = d["cols"]["cuda_runtime"]
    k = 0
    for i in sorted(
        (i for i in range(len(rt["ts"])) if rt["tid"][i] == sched),
        key=lambda i: rt["ts"][i],
    ):
        t = rt["ts"][i]
        while k < len(steps) and t >= steps[k]["ts"] + steps[k]["dur"]:
            k += 1
        if k < len(steps) and t >= steps[k]["ts"]:
            a = steps[k]["rt"][inv[rt["name"][i]]]
            a[0] += rt["dur"][i]
            a[1] += 1
    bounds = [s["ts"] for s in steps] + [float("inf")]
    kn = d["cols"]["kernel"]
    k = 0
    for i in sorted(range(len(kn["ts"])), key=lambda i: kn["ts"][i]):
        t = kn["ts"][i]
        while k < len(steps) and t >= bounds[k + 1]:
            k += 1
        if k < len(steps) and t >= bounds[k]:
            steps[k]["gpu"] = steps[k].get("gpu", 0.0) + kn["dur"][i]
            steps[k]["nk"] = steps[k].get("nk", 0) + 1
    mc = d["cols"]["gpu_memcpy"]
    k = 0
    for i in sorted(range(len(mc["ts"])), key=lambda i: mc["ts"][i]):
        t = mc["ts"][i]
        while k < len(steps) and t >= bounds[k + 1]:
            k += 1
        if k < len(steps) and t >= bounds[k]:
            nm = inv[mc["name"][i]]
            if nm.startswith("Memcpy DtoH"):
                steps[k]["dtoh"] = steps[k].get("dtoh", 0) + mc["bytes"][i]
            if "Pageable" in nm:
                steps[k]["pageable"] = steps[k].get("pageable", 0) + 1
    for k, s in enumerate(steps):
        s["gpu"] = s.get("gpu", 0.0)
        s["is_decode"] = s["b"]["decode_cuda_graph_runner.py(1386): execute"] > 0
        s["bs"] = s.get("dtoh", 0) // 8
        s["cycle"] = steps[k + 1]["ts"] - s["ts"] if k + 1 < len(steps) else None
        s["ss"] = s["rt"]["cudaStreamSynchronize"][0]
    gaps = [(steps[k + 1]["ts"] - steps[k]["ts"], k) for k in range(len(steps) - 1)]
    split = max(gaps)[1] + 1 if gaps else len(steps)
    return steps, split


def overlap(kern, t0, t1):
    ts, end = kern
    i = bisect.bisect_left(ts, t0 - 50000)
    tot = 0.0
    while i < len(ts) and ts[i] < t1:
        a, b = max(ts[i], t0), min(end[i], t1)
        if b > a:
            tot += b - a
        i += 1
    return tot


def fmt(xs):
    xs = sorted(xs)
    return f"{st.mean(xs) / 1e3:7.3f} mean {xs[len(xs) // 2] / 1e3:7.3f} p50 {xs[int(0.9 * len(xs)) - 1] / 1e3:7.3f} p90"


def report(label, sel, others):
    dec = [
        s
        for s in sel
        if s["is_decode"] and s["cycle"] is not None and s["cycle"] < 200000
    ]
    pre = [s for s in sel if not s["is_decode"]]
    print(
        f"\n=== {label}: {len(sel)} batches, {len(dec)} decode steps, {len(pre)} prefill batches ==="
    )
    if pre:
        print(
            f"prefill run_batch ms: p50 {st.median(s['dur'] for s in pre) / 1e3:.1f} max {max(s['dur'] for s in pre) / 1e3:.1f}"
        )
    if not dec:
        return
    print("bs distribution:", collections.Counter(s["bs"] for s in dec).most_common())
    for s in dec:
        for n, kern in others.items():
            s["ov_" + n] = overlap(kern, s["ts"], s["ts"] + s["dur"])
    rebuild = [
        s for s in dec if s["b"]["talker.py(1056): prepare_decode_buffers"] > 1000
    ]
    stall = [s for s in dec if s["ss"] > 1000]
    steady = [s for s in dec if s not in rebuild and s["ss"] <= 300]
    for name, grp in (
        ("all decode", dec),
        ("steady", steady),
        ("rebuild", rebuild),
        ("sync stall", stall),
    ):
        if not grp:
            continue
        print(f"\n-- {name}: {len(grp)} steps (ms) --")
        print(f"  cycle              {fmt([s['cycle'] for s in grp])}")
        print(f"  run_batch          {fmt([s['dur'] for s in grp])}")
        print(f"  gpu kernels        {fmt([s['gpu'] for s in grp])}")
        print(f"  event wait         {fmt([s['evsync'] for s in grp])}")
        for key in KEYS:
            print(f"  {key[-40:]:40s} {fmt([s['b'][key] for s in grp])}")
        for r in RT:
            print(
                f"  rt {r:22s} {fmt([s['rt'][r][0] for s in grp])}  n/step {st.mean(s['rt'][r][1] for s in grp):.2f}"
            )
        print(
            f"  pageable HtoD per step: {collections.Counter(s.get('pageable', 0) for s in grp).most_common(3)}"
        )
        for n in others:
            xs = [s["ov_" + n] for s in grp]
            print(
                f"  {n} gpu overlap in window {fmt(xs)}  steps with > 0.5 ms: {sum(1 for x in xs if x > 500)}"
            )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("pkl")
    p.add_argument(
        "--other",
        action="append",
        default=[],
        help="pickle of another process's trace on the same host",
    )
    args = p.parse_args(argv)
    d, inv = load(args.pkl)
    steps, split = build_steps(d, inv)
    others = {}
    for path in args.other:
        od, _ = load(path)
        k = od["cols"]["kernel"]
        ev = sorted(zip(k["ts"], k["dur"]))
        others[path.rsplit("/", 1)[-1].split("_")[0]] = (
            [e[0] for e in ev],
            [e[0] + e[1] for e in ev],
        )
    print(f"{len(steps)} batches, phase split at batch {split}")
    report("phase 1", steps[:split], others)
    report("phase 2", steps[split:], others)


if __name__ == "__main__":
    main()
