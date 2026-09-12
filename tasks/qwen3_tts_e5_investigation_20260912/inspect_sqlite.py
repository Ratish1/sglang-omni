"""Read already-exported E5 SQLite files in the container; never collects a profile.

The output distinguishes observed thread stacks from inferred roles. Missing
request/worker/event associations remain missing instead of being guessed.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

MASK_PID = 0xFFFFFFFFFF000000
EXPECTED_EPOCH_NS = {
    "control": 1789229444498424003,
    "early": 1789229795530618708,
}
ROLE_MARKERS = {
    "initial_vocoder": ("_run_initial_worker", "_run_initial_batch", "_commit_initial"),
    "followup_vocoder": ("_run_followup_worker", "_run_followup_batch"),
    "talker": ("_collect_codes", "_write_feedback_buffers", "code_predictor_forward"),
    "reference_encoder": ("_Qwen3TTSRefCodeBatcher", "_synchronize_outcomes"),
    "preprocessing": ("extract_speaker_embedding", "prepare_qwen3_tts_request"),
}


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def describe_ns(values):
    if not values:
        return {
            "count": 0,
            "sum_ms": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    values = sorted(values)

    def pct(q):
        x = (len(values) - 1) * q
        lo, hi = math.floor(x), math.ceil(x)
        return (values[lo] + (values[hi] - values[lo]) * (x - lo)) / 1e6

    return dict(
        count=len(values),
        sum_ms=sum(values) / 1e6,
        mean_ms=statistics.mean(values) / 1e6,
        p50_ms=pct(0.5),
        p95_ms=pct(0.95),
        p99_ms=pct(0.99),
        max_ms=values[-1] / 1e6,
    )


def main():
    if not sys.platform.startswith("linux"):
        raise SystemExit("Run this analysis in the Linux container, not on the Mac.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--arm", choices=EXPECTED_EPOCH_NS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    tables = {
        r[0]: r[1]
        for r in db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    }
    (out / "schema.sql").write_text(
        "\n\n".join(v + ";" for v in tables.values() if v) + "\n"
    )

    def columns(table):
        return {
            r[1]
            for r in db.execute('PRAGMA table_info("' + table.replace('"', '""') + '")')
        }

    def require(table, required):
        missing = set(required) - columns(table)
        if missing:
            raise RuntimeError(
                f"{table}: missing columns {sorted(missing)}; schema.sql saved. Do not guess replacements."
            )

    require("StringIds", ["id", "value"])
    require(
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        ["start", "end", "globalTid", "correlationId", "nameId"],
    )
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    dump_json(
        out / "python_related_strings.json",
        {
            k: v
            for k, v in strings.items()
            if any(
                s in v.lower()
                for s in ("qwen3", "python", "request_builders", "vocoder", "gil")
            )
        },
    )
    metadata = {}
    for name in tables:
        if name.startswith(
            ("META_DATA", "TARGET_INFO_SESSION", "NVTX_PAYLOAD")
        ) or name in ("ThreadNames", "TARGET_INFO_CUDA_STREAM", "ENUM_CUPTI_SYNC_TYPE"):
            safe = '"' + name.replace('"', '""') + '"'
            metadata[name] = [dict(r) for r in db.execute(f"SELECT * FROM {safe}")]
    dump_json(out / "metadata.json", metadata)
    require("TARGET_INFO_SESSION_START_TIME", ["utcEpochNs"])
    epochs = {
        r[0]
        for r in db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME")
    }
    expected_epoch = EXPECTED_EPOCH_NS[args.arm]
    if epochs != {expected_epoch}:
        raise RuntimeError(
            f"Capture epoch {epochs} differs from verified E5 metadata {expected_epoch}. Stop and inspect metadata.json."
        )

    # Equal 18-second interiors of the two approximately 20.2-second captures.
    # SQLite activity timestamps are capture-relative, not UTC or systemClockNs.
    lo, hi = 1_000_000_000, 19_000_000_000
    bounds = db.execute(
        "SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    if bounds[0] is None or not (
        -1_000_000_000 <= bounds[0] < lo and hi <= bounds[1] <= 30_000_000_000
    ):
        raise RuntimeError(
            f"Unexpected timestamp range {tuple(bounds)}; do not silently shift the time origin."
        )
    dump_json(
        out / "window.json",
        dict(
            arm=args.arm,
            epoch_ns=expected_epoch,
            relative_start_ns=lo,
            relative_end_ns=hi,
            utc_start_ns=expected_epoch + lo,
            utc_end_ns=expected_epoch + hi,
            observed_runtime_bounds_ns=list(bounds),
        ),
    )

    api_stats = defaultdict(lambda: array("d"))
    selected_names = (
        "EventSynchronize",
        "StreamSynchronize",
        "DeviceSynchronize",
        "StreamWaitEvent",
        "EventRecord",
        "GraphLaunch",
        "Memcpy",
    )
    with gzip.open(out / "selected_cuda_api.jsonl.gz", "wt") as f:
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            cc = columns(table)
            condition = "AND callchainId IS NULL" if "callchainId" in cc else ""
            query = f'SELECT * FROM "{table}" WHERE start >= ? AND end <= ? {condition} ORDER BY start'
            for r in db.execute(query, (lo, hi)):
                tid = r["globalTid"]
                if tid is None:
                    continue
                pid, native_tid = (tid >> 24) & 0xFFFFFF, tid & 0xFFFFFF
                name = strings.get(r["nameId"], str(r["nameId"]))
                api_stats[(table, pid, native_tid, name)].append(r["end"] - r["start"])
                if any(s in name for s in selected_names):
                    f.write(
                        json.dumps(
                            dict(r)
                            | dict(table=table, pid=pid, tid=native_tid, name=name)
                        )
                        + "\n"
                    )
    stat_fields = list(describe_ns([]))
    write_csv(
        out / "cuda_api_stats.csv",
        (
            dict(table=k[0], pid=k[1], tid=k[2], name=k[3], **describe_ns(v))
            for k, v in sorted(api_stats.items())
        ),
        ["table", "pid", "tid", "name"] + stat_fields,
    )
    del api_stats

    # Preserve existing event-record and synchronization identities. A record
    # timestamp alone does not prove when all of its preceding GPU work finished.
    for table in (
        "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    ):
        if table in tables:
            time_col = "timestamp" if "timestamp" in columns(table) else "start"
            with gzip.open(out / f"{table}.jsonl.gz", "wt") as f:
                for r in db.execute(
                    f'SELECT * FROM "{table}" WHERE "{time_col}" >= ? AND "{time_col}" <= ?',
                    (0, 21_000_000_000),
                ):
                    f.write(json.dumps(dict(r)) + "\n")

    nvtx_examples, marker_hits, seen_examples = [], Counter(), Counter()
    gil = defaultdict(lambda: array("d"))
    nvtx_labels = Counter()
    if "NVTX_EVENTS" in tables:
        nvtx_cols = columns("NVTX_EVENTS")
        query = "SELECT * FROM NVTX_EVENTS WHERE start < ? AND (end IS NULL OR end > ?) ORDER BY start"
        for r in db.execute(query, (hi, lo)):
            r = dict(r)
            if r.get("globalTid") is None:
                continue
            tid = r["globalTid"]
            pid, native_tid = (tid >> 24) & 0xFFFFFF, tid & 0xFFFFFF
            text = r.get("text") or strings.get(r.get("textId"), "") or ""
            if text in ("Waiting for GIL", "Holding GIL"):
                if r.get("end") is not None:
                    gil[(pid, native_tid, text)].append(
                        min(hi, r["end"]) - max(lo, r["start"])
                    )
                continue
            if not lo <= r["start"] < hi:
                continue
            js = r.get("jsonText") or strings.get(r.get("jsonTextId"), "") or ""
            # Retain raw payloads when Python backtraces are encoded; no invented
            # decoding of binaryData or conversion of string IDs into frames.
            joined = text + "\n" + js
            hit_roles = [
                role
                for role, markers in ROLE_MARKERS.items()
                if any(m in joined for m in markers)
            ]
            for role in hit_roles:
                marker_hits[(pid, native_tid, role)] += 1
            label = text if len(text) < 120 else text[:117] + "..."
            nvtx_labels[(pid, native_tid, label)] += 1
            group = (pid, native_tid, tuple(hit_roles) or ("unclassified",))
            if seen_examples[group] < 5 and (hit_roles or js or r.get("binaryData")):
                for col in ("textId", "jsonTextId"):
                    if col in r:
                        r[col + "_resolved"] = strings.get(r[col])
                nvtx_examples.append(
                    r | dict(pid=pid, tid=native_tid, marker_roles=hit_roles)
                )
                seen_examples[group] += 1
        dump_json(out / "python_payload_examples.json", nvtx_examples)
    write_csv(
        out / "thread_identity_candidates.csv",
        (
            dict(pid=k[0], tid=k[1], role_marker=k[2], matching_samples=v)
            for k, v in sorted(marker_hits.items())
        ),
        ["pid", "tid", "role_marker", "matching_samples"],
    )
    write_csv(
        out / "gil_stats.csv",
        (
            dict(pid=k[0], tid=k[1], name=k[2], **describe_ns(v))
            for k, v in sorted(gil.items())
        ),
        ["pid", "tid", "name"] + stat_fields,
    )
    write_csv(
        out / "nvtx_labels.csv",
        (
            dict(pid=k[0], tid=k[1], label=k[2], count=v)
            for k, v in nvtx_labels.most_common(500)
        ),
        ["pid", "tid", "label", "count"],
    )
    dump_json(
        out / "identity_status.json",
        dict(
            status=(
                "candidate markers require stack inspection"
                if marker_hits
                else "UNRESOLVED: no readable role markers; inspect Python payload/schema in the container"
            ),
            rules="Do not infer roles from TID ordering, kernel counts, or cudaEventSynchronize counts.",
            missing_request_markers="No initial enqueue/start/plan/chunk-selection instrumentation was added in E5.",
        ),
    )

    kernel_stats = defaultdict(
        lambda: {k: array("d") for k in ("kernel", "launch_to_start", "post_api_queue")}
    )
    unmatched = 0
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        require(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            ["start", "end", "globalPid", "correlationId", "streamId", "demangledName"],
        )
        # Match process + correlation, not correlation alone. Nested runtime
        # callchain records are excluded, following NVIDIA's own report query.
        runtime_condition = (
            "WHERE callchainId IS NULL"
            if "callchainId" in columns("CUPTI_ACTIVITY_KIND_RUNTIME")
            else ""
        )
        db.execute(
            f"CREATE TEMP TABLE launches AS SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME {runtime_condition}"
        )
        db.execute(
            "CREATE INDEX temp.launch_correlation ON launches(correlationId, (globalTid & 0xFFFFFFFFFF000000))"
        )
        query = """SELECT k.*, r.start AS api_start, r.end AS api_end,
            r.globalTid AS api_tid, r.nameId AS api_name_id
            FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN launches r
            ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & 0xFFFFFFFFFF000000)
            WHERE k.start >= ? AND k.end <= ? ORDER BY k.start"""
        with gzip.open(out / "kernels.jsonl.gz", "wt") as f:
            for row in db.execute(query, (lo, hi)):
                r = dict(row)
                if r["api_tid"] is None:
                    unmatched += 1
                    continue
                pid, tid = (r["api_tid"] >> 24) & 0xFFFFFF, r["api_tid"] & 0xFFFFFF
                name = strings.get(r["demangledName"], str(r["demangledName"]))
                api = strings.get(r["api_name_id"], str(r["api_name_id"]))
                key = (pid, tid, r["streamId"], api)
                v = kernel_stats[key]
                v["kernel"].append(r["end"] - r["start"])
                v["launch_to_start"].append(r["start"] - r["api_start"])
                if r["start"] >= r["api_end"]:
                    v["post_api_queue"].append(r["start"] - r["api_end"])
                f.write(
                    json.dumps(
                        {
                            k: r[k]
                            for k in (
                                "start",
                                "end",
                                "streamId",
                                "contextId",
                                "deviceId",
                                "correlationId",
                                "api_start",
                                "api_end",
                            )
                        }
                        | dict(pid=pid, tid=tid, name=name, api=api)
                    )
                    + "\n"
                )
    stats_rows = []
    for key, v in sorted(kernel_stats.items()):
        for metric, values in v.items():
            stats_rows.append(
                dict(
                    pid=key[0],
                    tid=key[1],
                    stream=key[2],
                    api=key[3],
                    metric=metric,
                    **describe_ns(values),
                )
            )
    write_csv(
        out / "kernel_stats.csv",
        stats_rows,
        ["pid", "tid", "stream", "api", "metric"] + stat_fields,
    )
    dump_json(
        out / "kernel_join_status.json",
        dict(
            unmatched_runtime_kernel_rows=unmatched,
            meaning="Unmatched driver/graph work is not attributed to an arbitrary thread. Kernel queue sums overlap and are not wall time.",
        ),
    )

    # All first-audio paths must start AND finish inside the same 18-second window.
    points = {
        "admit": ("coordinator", "request_admission"),
        "pre_dispatch": ("preprocessing", "stage_dispatch"),
        "pre_complete": ("preprocessing", "stage_complete"),
        "prefill_start": ("tts_engine", "scheduler_prefill_start"),
        "prefill_end": ("tts_engine", "scheduler_prefill_end"),
        "code_received": ("vocoder", "stage_stream_chunk_received"),
        "audio_sent": ("vocoder", "stage_first_stream_chunk_sent"),
        "audio_received": ("coordinator", "stage_stream_chunk_received"),
    }
    by_request = defaultdict(dict)
    event_files = sorted(args.events.glob("*.jsonl"))
    if not event_files:
        raise RuntimeError(f"No event files at {args.events}")
    with gzip.open(out / "stage_events.jsonl.gz", "wt") as f:
        for path in event_files:
            for line in path.open():
                r = json.loads(line)
                relative = r["timestamp_ns"] - expected_epoch
                if not lo <= relative < hi:
                    continue
                r["relative_ns"] = relative
                f.write(json.dumps(r) + "\n")
                for key, pair in points.items():
                    if (r["stage"], r["event_name"]) != pair:
                        continue
                    if (
                        key in ("code_received", "audio_received")
                        and r.get("metadata", {}).get("chunk_id") != 0
                    ):
                        continue
                    old = by_request[r["request_id"]].get(key)
                    by_request[r["request_id"]][key] = (
                        relative if old is None else min(old, relative)
                    )
    paths = []
    for rid, p in by_request.items():
        if len(p) == len(points):
            if (
                not p["admit"]
                <= p["code_received"]
                <= p["audio_sent"]
                <= p["audio_received"]
            ):
                raise RuntimeError(f"Unexpected stage ordering for {rid}")
            paths.append(
                dict(
                    request_id=rid,
                    **{k + "_ns": v for k, v in p.items()},
                    preprocessing_ms=(p["pre_complete"] - p["pre_dispatch"]) / 1e6,
                    prefill_to_audio_ms=(p["audio_sent"] - p["prefill_start"]) / 1e6,
                    code_to_audio_ms=(p["audio_sent"] - p["code_received"]) / 1e6,
                    admit_to_audio_ms=(p["audio_received"] - p["admit"]) / 1e6,
                )
            )
    paths.sort(key=lambda r: r["code_to_audio_ms"], reverse=True)
    write_csv(
        out / "first_audio_paths.csv",
        paths,
        ["request_id"]
        + [k + "_ns" for k in points]
        + [
            "preprocessing_ms",
            "prefill_to_audio_ms",
            "code_to_audio_ms",
            "admit_to_audio_ms",
        ],
    )
    dump_json(
        out / "first_audio_summary.json",
        dict(
            complete_paths=len(paths),
            incomplete_observed_requests=len(by_request) - len(paths),
            preprocessing=describe_ns([r["preprocessing_ms"] * 1e6 for r in paths]),
            prefill_to_audio=describe_ns(
                [r["prefill_to_audio_ms"] * 1e6 for r in paths]
            ),
            code_to_audio=describe_ns([r["code_to_audio_ms"] * 1e6 for r in paths]),
            admit_to_audio=describe_ns([r["admit_to_audio_ms"] * 1e6 for r in paths]),
            slowest_paths=paths[:10],
            attribution="UNRESOLVED: stage endpoints do not expose initial enqueue/start or the chunks selected by a plan.",
        ),
    )
    manifest = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            manifest.append(
                dict(name=path.name, bytes=path.stat().st_size, sha256=digest)
            )
    dump_json(out / "files.json", manifest)
    print(
        f"{args.arm}: wrote {out}; {len(paths)} complete first-audio paths. Read identity_status.json before comparing threads."
    )


if __name__ == "__main__":
    main()
