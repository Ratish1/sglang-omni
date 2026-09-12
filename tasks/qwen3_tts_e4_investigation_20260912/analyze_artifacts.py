"""Reconstruct E4 streaming timelines offline; no model or framework imports."""

import ast
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/qwen3-tts-e4-investigation-20260912/results_e4_dc55923e2"
OUT = Path(__file__).parent


def describe(values):
    values = sorted(values)
    if not values:
        return None

    def percentile(q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(
        n=len(values),
        mean=statistics.mean(values),
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
        max=values[-1],
    )


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace(",", "."))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def first(events, stage, name):
    return next(
        (
            row["timestamp_ns"]
            for row in events
            if row["stage"] == stage and row["event_name"] == name
        ),
        None,
    )


POINTS = {
    "admit": ("coordinator", "request_admission"),
    "pre_input": ("preprocessing", "stage_input_received"),
    "pre_complete": ("preprocessing", "stage_complete"),
    "tts_input": ("tts_engine", "stage_input_received"),
    "build_start": ("tts_engine", "scheduler_request_build_start"),
    "build_end": ("tts_engine", "scheduler_request_build_end"),
    "queue": ("tts_engine", "scheduler_queue_enter"),
    "prefill_start": ("tts_engine", "scheduler_prefill_start"),
    "prefill_end": ("tts_engine", "scheduler_prefill_end"),
    "first_emit": ("tts_engine", "scheduler_first_emit"),
    "code_sent": ("tts_engine", "stage_first_stream_chunk_sent"),
    "code_received": ("vocoder", "stage_stream_chunk_received"),
    "audio_sent": ("vocoder", "stage_first_stream_chunk_sent"),
    "audio_received": ("coordinator", "coordinator_stream_received"),
    "model_end": ("tts_engine", "model_path_end"),
    "tts_complete": ("tts_engine", "stage_complete"),
    "voc_final_input": ("vocoder", "stage_input_received"),
    "voc_complete": ("vocoder", "stage_complete"),
    "terminal": ("coordinator", "terminal_response"),
}
CHAIN = [
    "admit",
    "pre_input",
    "pre_complete",
    "tts_input",
    "build_start",
    "build_end",
    "queue",
    "prefill_start",
    "first_emit",
    "code_sent",
    "code_received",
    "audio_sent",
    "audio_received",
]
PAIRS = list(zip(CHAIN, CHAIN[1:])) + [
    ("admit", "audio_received"),
    ("pre_input", "audio_sent"),
    ("prefill_start", "prefill_end"),
    ("prefill_end", "model_end"),
    ("tts_input", "tts_complete"),
    ("voc_final_input", "voc_complete"),
    ("audio_sent", "voc_final_input"),
    ("model_end", "terminal"),
]


def aggregate_timelines(rows):
    result = {}
    for left, right in PAIRS:
        result[f"{left}_to_{right}_ms"] = describe(
            [
                (r["points_ns"][right] - r["points_ns"][left]) / 1e6
                for r in rows
                if r["points_ns"][left] is not None
                and r["points_ns"][right] is not None
            ]
        )
    for field in ("codes_received_before_first_audio", "code_gap_ms", "audio_gap_ms"):
        values = [r[field] for r in rows if r[field] is not None]
        if field.endswith("gap_ms"):
            values = [v for row in values for v in row]
        result[field] = describe(values)
    return result


audit = {"files": [], "arms": {}, "passes": {}, "comparisons": {}}
for path in sorted(SOURCE.rglob("*")):
    if path.is_file():
        raw = path.read_bytes()
        audit["files"].append(
            dict(
                path=str(path.relative_to(SOURCE)),
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )

all_runs = {}
for arm in sorted(SOURCE.glob("e4*")):
    if not arm.is_dir():
        continue
    events = []
    by_file = {}
    for path in sorted((arm / "events_pass2").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        by_file[path.name] = dict(
            rows=len(rows),
            events=dict(Counter(f'{r["stage"]}/{r["event_name"]}' for r in rows)),
        )
        events.extend(rows)
    events.sort(key=lambda row: row["timestamp_ns"])
    grouped = defaultdict(list)
    for row in events:
        grouped[row["request_id"]].append(row)
    timelines = []
    for request_id, rows in grouped.items():
        points = {key: first(rows, *point) for key, point in POINTS.items()}
        code_times = [
            r["timestamp_ns"]
            for r in rows
            if r["stage"] == "tts_engine"
            and r["event_name"] == "stage_stream_chunk_sent"
        ]
        audio_times = [
            r["timestamp_ns"]
            for r in rows
            if r["stage"] == "vocoder" and r["event_name"] == "stage_stream_chunk_sent"
        ]
        n_codes = (
            sum(
                r["stage"] == "vocoder"
                and r["event_name"] == "stage_stream_chunk_received"
                and r["timestamp_ns"] <= points["audio_sent"]
                for r in rows
            )
            if points["audio_sent"]
            else None
        )
        timelines.append(
            dict(
                request_id=request_id,
                points_ns=points,
                codes_received_before_first_audio=n_codes,
                code_gap_ms=[(b - a) / 1e6 for a, b in zip(code_times, code_times[1:])],
                audio_gap_ms=[
                    (b - a) / 1e6 for a, b in zip(audio_times, audio_times[1:])
                ],
            )
        )
    timelines.sort(key=lambda row: row["points_ns"]["admit"] or math.inf)
    assert all(r["points_ns"]["admit"] is not None for r in timelines)
    warmup, measured = timelines[0], timelines[1:]
    common = measured[:180]
    assert len(common) == 180 and all(r["points_ns"]["terminal"] for r in common)
    first_audio_complete = [
        r for r in measured if all(r["points_ns"][p] is not None for p in CHAIN)
    ]
    prof_start, prof_end = (
        events[0]["timestamp_ns"] / 1e9,
        events[-1]["timestamp_ns"] / 1e9,
    )
    lines = (arm / "serve.log").read_text().splitlines()
    records, codec, refs, notable = [], [], [], []
    for number, line in enumerate(lines, 1):
        tm = re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}", line)
        t = timestamp(tm[0]) if tm else None
        kind = next(
            (
                name
                for needle, name in (
                    ("Coordinator submitted", "submit"),
                    ("Prefill batch", "prefill"),
                    ("Decode batch", "decode"),
                )
                if needle in line
            ),
            None,
        )
        if kind:
            fields = {
                key: int(value) for key, value in re.findall(r"#([\w-]+): (\d+)", line)
            }
            records.append(dict(t=t, kind=kind, line=number, **fields))
        if "incremental Codec state: {" in line:
            codec.append(
                dict(
                    t=t,
                    line=number,
                    state=ast.literal_eval(line.split("incremental Codec state: ")[1]),
                )
            )
        if "reference encode stats:" in line:
            refs.append(
                dict(
                    t=t,
                    line=number,
                    state=ast.literal_eval(line.split("reference encode stats: ")[1]),
                )
            )
        if (
            re.search(
                r"ERROR|WARNING|Traceback|Exception|failed|KV pool holds|KV Cache|Capture target|runtime configuration:|placement/topology|mem_fraction_static|tokenizer.*load",
                line,
                re.I,
            )
            and "incremental Codec state:" not in line
        ):
            notable.append(dict(line=number, text=line))
    audit["arms"][arm.name] = dict(
        head=(arm / "head.txt").read_text().strip(),
        diff=(arm / "change.diff").read_text(),
        event_files=by_file,
        event_start_utc=datetime.fromtimestamp(prof_start, timezone.utc).isoformat(),
        event_end_utc=datetime.fromtimestamp(prof_end, timezone.utc).isoformat(),
        admissions=len(timelines),
        terminal=sum(r["points_ns"]["terminal"] is not None for r in timelines),
        full_first_audio_paths=len(first_audio_complete),
        all_after_warmup=aggregate_timelines(measured),
        full_first_audio_cohort=aggregate_timelines(first_audio_complete),
        first180_after_warmup=aggregate_timelines(common),
        timelines=timelines,
        codec_snapshots=codec,
        reference_snapshots=refs,
        notable_log_lines=notable,
    )
    for run in sorted(arm.glob("pass*")):
        name = str(run.relative_to(SOURCE))
        loaded = json.loads((run / "speed_results.json").read_text())
        rows = loaded["per_request"]
        clocks = [json.loads(line) for line in (run / "client_timestamps.jsonl").open()]
        assert len(clocks) == len(rows) + 1 == 1089
        assert (
            abs(timestamp(clocks[0]["start"]) - prof_start) < 0.1 or run.name == "pass1"
        )
        clocks = {row["request_id"]: row for row in clocks[1:]}
        assert len(clocks) == len(rows) == len({row["id"] for row in rows})
        generated = json.loads((run / "generated.json").read_text())
        with (run / "results.csv").open() as handle:
            csv_rows = list(csv.DictReader(handle))
        assert (
            [r["id"] for r in rows]
            == [r["sample_id"] for r in generated]
            == [r["id"] for r in csv_rows]
        )
        for row, item in zip(rows, csv_rows):
            assert (
                row["is_success"] and not row["error"] and clocks[row["id"]]["success"]
            )
            assert (
                row["audio_chunk_count"]
                == len(row["inter_chunk_s"]) + 1
                == len(row["chunk_audio_duration_s"])
            )
            assert (
                abs(sum(row["chunk_audio_duration_s"]) - row["audio_duration_s"])
                < 0.00011
            )
            assert all(
                float(item[k]) == row[k]
                for k in ("latency_s", "audio_ttfp_s", "audio_duration_s")
            )
            row["start"] = timestamp(clocks[row["id"]]["start"])
            row["end"] = timestamp(clocks[row["id"]]["end"])
        start, end = min(r["start"] for r in rows), max(r["end"] for r in rows)
        window = [r for r in records if start <= r["t"] <= end]
        prefills = [r for r in window if r["kind"] == "prefill"]
        submits = [r for r in window if r["kind"] == "submit"]
        decodes = [r for r in window if r["kind"] == "decode"]
        assert len(submits) == sum(r["new-seq"] for r in prefills) == len(rows)
        ordered = sorted(rows, key=lambda r: r["start"])
        by_end = sorted(rows, key=lambda r: r["end"])
        profile_cohort = (
            [r for r in rows if r["end"] <= prof_end] if run.name == "pass2" else []
        )
        after_profile = (
            [r for r in rows if r["start"] > prof_end] if run.name == "pass2" else []
        )
        audit["passes"][name] = dict(
            summary=loaded["summary"],
            config=loaded["config"],
            start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
            end_utc=datetime.fromtimestamp(end, timezone.utc).isoformat(),
            qps_from_timestamps=len(rows) / (end - start),
            prefill_batches=len(prefills),
            prefill_batch_sizes=describe([r["new-seq"] for r in prefills]),
            input_tokens=sum(r["new-token"] for r in prefills),
            cached_tokens=sum(r["cached-token"] for r in prefills),
            decode_running=describe([r["running-req"] for r in decodes]),
            queue_max=max(r["queue-req"] for r in prefills + decodes),
            ttfc_quartiles_by_start=[
                describe([r["audio_ttfp_s"] for r in ordered[i : i + 272]])
                for i in range(0, 1088, 272)
            ],
            profile_cohort_ttfc_s=describe([r["audio_ttfp_s"] for r in profile_cohort]),
            after_profile_ttfc_s=describe([r["audio_ttfp_s"] for r in after_profile]),
            profile_cohort_latency_s=describe([r["latency_s"] for r in profile_cohort]),
            after_profile_latency_s=describe([r["latency_s"] for r in after_profile]),
            chunk_arrivals_s=[
                describe(
                    [
                        r["audio_ttfp_s"] + sum(r["inter_chunk_s"][:k])
                        for r in rows
                        if len(r["inter_chunk_s"]) >= k
                    ]
                )
                for k in range(8)
            ],
            max_underrun_s=describe([r["max_playback_underrun_s"] for r in rows]),
            underrun_requests=sum(r["max_playback_underrun_s"] > 0 for r in rows),
            first_payload_sizes=sorted({r["first_audio_payload_bytes"] for r in rows}),
            total_audio_s=sum(r["audio_duration_s"] for r in rows),
            last_completion={
                **{
                    k: by_end[-1][k]
                    for k in (
                        "id",
                        "text",
                        "latency_s",
                        "audio_duration_s",
                        "audio_ttfp_s",
                        "audio_chunk_count",
                    )
                },
                "after_penultimate_s": by_end[-1]["end"] - by_end[-2]["end"],
            },
            longest_audio=[
                {k: r[k] for k in ("id", "text", "audio_duration_s", "latency_s")}
                for r in sorted(
                    rows, key=lambda r: r["audio_duration_s"], reverse=True
                )[:5]
            ],
        )
        all_runs[name] = {r["id"]: r for r in rows}

control_path = SOURCE / "control_sliceA_pass2/speed_results.json"
control = json.loads(control_path.read_text())
old_path = (
    ROOT
    / "artifacts/qwen3-tts-streaming-investigation-20260912/qwen3-tts-runbook22-898dc3234/streaming_pair2/A_sliceA_valid/pass2/speed_results.json"
)
audit["control_is_byte_identical_to_runbook22"] = (
    control_path.read_bytes() == old_path.read_bytes()
)
control_rows = {r["id"]: r for r in control["per_request"]}
for name, rows in all_runs.items():
    assert rows.keys() == control_rows.keys()
    audit["comparisons"][name] = dict(
        ttfc_delta_s=describe(
            [rows[k]["audio_ttfp_s"] - control_rows[k]["audio_ttfp_s"] for k in rows]
        ),
        latency_delta_s=describe(
            [rows[k]["latency_s"] - control_rows[k]["latency_s"] for k in rows]
        ),
        matching_audio_duration=sum(
            rows[k]["audio_duration_s"] == control_rows[k]["audio_duration_s"]
            for k in rows
        ),
        matching_chunk_schedule=sum(
            rows[k]["chunk_audio_duration_s"]
            == control_rows[k]["chunk_audio_duration_s"]
            for k in rows
        ),
    )

(OUT / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
for name, arm in audit["arms"].items():
    print(
        name,
        "admit/terminal/full-first",
        arm["admissions"],
        arm["terminal"],
        arm["full_first_audio_paths"],
    )
    for key, d in arm["first180_after_warmup"].items():
        print(
            " ",
            key,
            (
                None
                if d is None
                else {
                    k: round(v, 3)
                    for k, v in d.items()
                    if k in ("n", "mean", "p50", "p95")
                }
            ),
        )
for name, run in audit["passes"].items():
    print(
        name,
        "qps",
        round(run["qps_from_timestamps"], 3),
        "tokens",
        run["input_tokens"],
        run["cached_tokens"],
        "prefillbatches",
        run["prefill_batches"],
        "audio",
        run["total_audio_s"],
        "last",
        run["last_completion"],
    )
