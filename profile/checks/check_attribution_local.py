"""Build a synthetic nsys-shaped sqlite and check nsys_gap_attribution.py on it.

    python profile/checks/check_attribution_local.py

Timeline (ms): scheduler thread 1, encoder thread 2, builder thread 3.
  iteration 0: recv 0-0.1, next_batch 0.1-0.5 (new_prefill 0.15-0.45, hold mark 0.4), sleep 0.5-1.0;
               builder: build:req 0.2-0.6
  iteration 1: recv 1.0-1.05, next_batch 1.05-1.3, exec:sync:extend bs=2 tok=40 1.3-2.7
               (run:build 1.3-1.4, run:forward 1.4-1.6, run:finalize 1.6-2.7);
               graph 1.5-1.95 (launched at 1.45), kernel 1.98-2.5 (launched at 1.55)
  iteration 2: recv 2.7-2.75, next_batch 2.75-3.0 (prepare_decode 2.8-2.9; memcpy 2.95-2.97 launched at 2.85),
               exec:launch:decode bs=2 3.0-3.2; kernel 3.3-4.0 (launched at 3.05);
               encoder: enc:batch n=3 3.5-4.5 (enc:sync 3.8-4.5), kernel 3.6-3.9 (launched at 3.55 on the encoder thread)
Checked with an explicit 0..5 ms window and with the default exec window (1.3..3.2).
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

MS = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "nsys_gap_attribution.py")


def build(path: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table StringIds (id integer primary key, value text);
        create table ENUM_NVTX_EVENT_TYPE (id integer, name text);
        create table NVTX_EVENTS (start integer, end integer, eventType integer, globalTid integer, text text, textId integer, domainId integer);
        create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer, streamId integer, globalPid integer, shortName integer, correlationId integer);
        create table CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start integer, end integer, deviceId integer, streamId integer, globalPid integer, graphId integer, correlationId integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, deviceId integer, streamId integer, globalPid integer, copyKind integer, correlationId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME (start integer, end integer, correlationId integer, globalTid integer, nameId integer);
        """
    )
    con.executemany(
        "insert into ENUM_NVTX_EVENT_TYPE values (?, ?)",
        [(59, "NvtxPushPopRange"), (60, "NvtxStartEndRange"), (34, "NvtxMark")],
    )
    strings = {}

    def sid(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            con.execute("insert into StringIds values (?, ?)", (strings[text], text))
        return strings[text]

    sched, enc, bld = (7 << 24) | 1, (7 << 24) | 2, (7 << 24) | 3
    ranges = [
        (0.0, 0.1, "sched:recv", sched),
        (0.1, 0.5, "sched:next_batch", sched),
        (0.15, 0.45, "sched:new_prefill", sched),
        (0.5, 1.0, "sched:sleep", sched),
        (0.2, 0.6, "build:req", bld),
        (1.0, 1.05, "sched:recv", sched),
        (1.05, 1.3, "sched:next_batch", sched),
        (1.3, 2.7, "exec:sync:extend bs=2 tok=40", sched),
        (1.3, 1.4, "run:build", sched),
        (1.4, 1.6, "run:forward", sched),
        (1.6, 2.7, "run:finalize", sched),
        (2.7, 2.75, "sched:recv", sched),
        (2.75, 3.0, "sched:next_batch", sched),
        (2.8, 2.9, "sched:prepare_decode", sched),
        (3.0, 3.2, "exec:launch:decode bs=2", sched),
        (3.5, 4.5, "enc:batch n=3", enc),
        (3.8, 4.5, "enc:sync", enc),
    ]
    for i, (s, e, name, tid) in enumerate(ranges):
        inline = i % 2 == 0
        con.execute(
            "insert into NVTX_EVENTS values (?, ?, 59, ?, ?, ?, 0)",
            (
                int(s * MS),
                int(e * MS),
                tid,
                name if inline else None,
                None if inline else sid(name),
            ),
        )
    con.execute(
        "insert into NVTX_EVENTS values (?, NULL, 34, ?, ?, NULL, 0)",
        (int(0.4 * MS), sched, "sched:hold waiting=3"),
    )
    con.execute(
        "insert into CUPTI_ACTIVITY_KIND_GRAPH_TRACE values (?, ?, 0, 7, 7, 1, 1)",
        (int(1.5 * MS), int(1.95 * MS)),
    )
    for s, e, cid in [(1.98, 2.5, 2), (3.3, 4.0, 3), (3.6, 3.9, 4)]:
        con.execute(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 0, 7, 7, 1, ?)",
            (int(s * MS), int(e * MS), cid),
        )
    con.execute(
        "insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, 0, 7, 7, 1, 5)",
        (int(2.95 * MS), int(2.97 * MS)),
    )
    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, ?, 1)",
        [
            (int(1.45 * MS), int(1.46 * MS), 1, sched),
            (int(1.55 * MS), int(1.57 * MS), 2, sched),
            (int(3.05 * MS), int(3.06 * MS), 3, sched),
            (int(3.55 * MS), int(3.56 * MS), 4, enc),
            (int(2.85 * MS), int(2.855 * MS), 5, sched),
        ],
    )
    con.commit()
    con.close()


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def run(db, out, *window):
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            db,
            "--json",
            out,
            *(("--window", *window) if window else ()),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return json.load(open(out))


with tempfile.TemporaryDirectory() as d:
    db = os.path.join(d, "synthetic.sqlite")
    build(db)
    full = run(db, os.path.join(d, "full.json"), "0", str(5 * MS))
    ex = run(db, os.path.join(d, "exec.json"))

assert full["gpu_tables"] == {"KERNEL": 3, "GRAPH_TRACE": 1, "MEMCPY": 1}, full[
    "gpu_tables"
]
assert close(full["gpu_busy_ms"], 1.69) and close(full["gpu_idle_ms"], 3.31), (
    full["gpu_busy_ms"],
    full["gpu_idle_ms"],
)
a = full["big_attribution"]
expected = {
    "sched:recv": 0.2,
    "sched:next_batch": 0.48,
    "sched:new_prefill": 0.3,
    "sched:sleep": 0.5,
    "run:build": 0.1,
    "run:forward": 0.1,
    "run:finalize": 0.2,
    "sched:prepare_decode": 0.1,
    "exec:launch:decode": 0.2,
    "unlabeled": 1.1,
}
assert set(a) == set(expected), set(a) ^ set(expected)
for k, v in expected.items():
    assert close(a[k]["idle_ms"], v), (k, a[k])
assert a["sched:next_batch"]["gaps"] == 3 and a["unlabeled"]["gaps"] == 2, a
assert full["micro_gaps"] == 1 and close(full["micro_idle_ms"], 0.03), (
    full["micro_gaps"],
    full["micro_idle_ms"],
)
assert close(full["micro_by_exec"]["exec:sync:extend"]["idle_ms"], 0.03)
assert close(full["micro_by_inner"]["run:finalize"], 0.03)
assert full["hold_marks"] == 1 and close(full["idle_in_hold_iterations_ms"], 1.5)
assert full["extend_bs_hist"] == {"2": 1} and full["extend_tok_median"] == 40
assert (
    close(full["encoder_overlap_idle_ms"], 0.5)
    and close(full["encoder_overlap_exec_ms"], 0.0)
    and close(full["enc_sync_overlap_exec_ms"], 0.0)
)
assert (
    full["builder_threads_found"] == 1
    and close(full["builder_overlap_idle_ms"], 0.4)
    and close(full["builder_overlap_exec_ms"], 0.0)
)
assert full["iterations"] == 2 and close(full["iterations_without_exec_pct"], 50.0)
assert close(full["host"]["exec:sync:extend"]["total_ms"], 1.4)
assert close(full["host_unlabeled_ms"], 5.0 - (1.0 + 1.7 + 0.5)), full[
    "host_unlabeled_ms"
]

L = full["launches_by_label"]
assert full["runtime_rows_found"] == 5
assert full["decode_steps"] == 1 and full["extends"] == 1, (
    full["decode_steps"],
    full["extends"],
)
assert L["run:forward"]["api_calls"] == 2 and L["run:forward"]["gpu_rows"] == {
    "GRAPH_TRACE": 1,
    "KERNEL": 1,
}, L["run:forward"]
assert close(L["run:forward"]["gpu_ms"], 0.45 + 0.52) and close(
    L["run:forward"]["api_ms"], 0.03
), L["run:forward"]
assert L["exec:launch:decode"]["api_calls"] == 1 and L["exec:launch:decode"][
    "gpu_rows"
] == {"KERNEL": 1}
assert L["sched:prepare_decode"]["api_calls"] == 1 and L["sched:prepare_decode"][
    "gpu_rows"
] == {"MEMCPY": 1}
assert (
    sum(v["api_calls"] for v in L.values()) == 4
), "the encoder-thread launch must be excluded"

assert ex["window_ns"] == [int(1.3 * MS), int(3.2 * MS)], ex["window_ns"]
assert close(ex["gpu_idle_ms"], 0.91), ex["gpu_idle_ms"]
b = ex["big_attribution"]
for k, v in {
    "run:build": 0.1,
    "run:forward": 0.1,
    "run:finalize": 0.2,
    "sched:recv": 0.05,
    "sched:next_batch": 0.13,
    "sched:prepare_decode": 0.1,
    "exec:launch:decode": 0.2,
}.items():
    assert close(b[k]["idle_ms"], v), (k, b[k])
assert "unlabeled" not in b and "sched:sleep" not in b, set(b)
print("ATTRIBUTION LOCAL CHECK PASSED")
