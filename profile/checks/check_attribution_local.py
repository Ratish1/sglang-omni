"""Build a synthetic nsys-shaped sqlite and check nsys_gap_attribution.py on it.

    python profile/checks/check_attribution_local.py

The timeline (ms): scheduler thread 1, encoder thread 2, window 0..5.
  iteration 0: recv 0-0.1, next_batch 0.1-0.5 (new_prefill 0.15-0.45, hold mark 0.4), sleep 0.5-1.0
  iteration 1: recv 1.0-1.05, next_batch 1.05-1.3, exec:sync:extend bs=2 tok=40 1.3-2.7
               (run:build 1.3-1.4, run:forward 1.4-1.6, run:finalize 1.6-2.7); kernel 1.5-2.5
  iteration 2: recv 2.7-2.75, next_batch 2.75-3.0 (prepare_decode 2.8-2.9), exec:launch:decode bs=2 3.0-3.2;
               kernel 3.3-4.0; encoder: enc:batch n=3 3.5-4.5 (enc:sync 3.8-4.5), kernel 3.6-3.9
Expected idle gaps: 0-1.5 (sched:recv), 2.5-3.3 (run:finalize), 4.0-5.0 (unlabeled).
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
        create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer, streamId integer, globalPid integer, shortName integer);
        create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, deviceId integer, streamId integer, globalPid integer, copyKind integer);
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

    sched, enc = (7 << 24) | 1, (7 << 24) | 2
    ranges = [
        (0.0, 0.1, "sched:recv", sched),
        (0.1, 0.5, "sched:next_batch", sched),
        (0.15, 0.45, "sched:new_prefill", sched),
        (0.5, 1.0, "sched:sleep", sched),
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
    for s, e in [(1.5, 2.5), (3.3, 4.0), (3.6, 3.9)]:
        con.execute(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, 0, 7, 7, 1)",
            (int(s * MS), int(e * MS)),
        )
    con.commit()
    con.close()


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


with tempfile.TemporaryDirectory() as d:
    db = os.path.join(d, "synthetic.sqlite")
    out = os.path.join(d, "out.json")
    build(db)
    subprocess.run(
        [sys.executable, SCRIPT, db, "--window", "0", str(5 * MS), "--json", out],
        check=True,
        stdout=subprocess.PIPE,
    )
    r = json.load(open(out))

assert close(r["gpu_busy_ms"], 1.7), r["gpu_busy_ms"]
assert close(r["gpu_idle_ms"], 3.3), r["gpu_idle_ms"]
a = r["attribution"]
assert close(a["sched:recv"]["idle_ms"], 1.5), a
assert close(a["run:finalize"]["idle_ms"], 0.8), a
assert close(a["unlabeled"]["idle_ms"], 1.0), a
assert set(a) == {"sched:recv", "run:finalize", "unlabeled"}, set(a)
assert r["hold_marks"] == 1 and r["hold_iterations"] == 1
assert close(r["idle_in_hold_iterations_ms"], 1.5), r["idle_in_hold_iterations_ms"]
assert r["extend_bs_hist"] == {"2": 1} and r["extend_tok_median"] == 40
assert close(r["idle_overlapping_encoder_ms"], 0.5), r["idle_overlapping_encoder_ms"]
assert close(r["enc_sync_overlapping_exec_ms"], 0.0)
assert r["iterations"] == 2 and close(r["iterations_without_exec_pct"], 50.0), r
assert close(r["host"]["exec:sync:extend"]["total_ms"], 1.4), r["host"]
assert close(r["host_unlabeled_ms"], 5.0 - (1.0 + 1.7 + 0.5)), r["host_unlabeled_ms"]
assert r["scheduler_threads_found"] == 1 and r["encoder_threads_found"] == 1
print("ATTRIBUTION LOCAL CHECK PASSED")
