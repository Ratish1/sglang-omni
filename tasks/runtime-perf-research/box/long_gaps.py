"""What the talker and the thinker did inside a streamed speech cell's long chunk gaps, from hop probe records.

usage: python3 long_gaps.py --cell c16_events --min-ms 600 <boot> [<boot> ...]

Boots run with baseline.py --probe, OMNI_HOP_PROBE_TALKER_BATCHES=1 and OMNI_HOP_PROBE_THINKER_STEPS=1. A gap is the time
between two code2wav outputs of one stream. Gaps at or over --min-ms are read against gaps under 300 ms: talker decode
steps inside the gap and how many carried the stream, the longest talker period inside it, whether the stream's thinker
was still sending text, and the thinker's busy share, longest step and prefill tokens inside it.
"""

import argparse
import bisect
import collections
import statistics

from chunk_gap_parts import load_records


def gap_reads(boot: str, cell: str, low_ms: float, high_ms: float) -> list[dict]:
    events, queue, _, _ = load_records(boot, cell)
    talker_steps = sorted((e for e in events if e["k"] == "tb"), key=lambda e: e["ns"])
    step_ns = [e["ns"] for e in talker_steps]
    thinker_steps = sorted(
        (e for e in events if e["k"] == "ts"), key=lambda e: e["ns0"]
    )
    thinker_starts = [e["ns0"] for e in thinker_steps]
    outputs = collections.defaultdict(list)
    last_thinker_token = {}
    for item in queue:
        if item[2] != "put":
            continue
        elif item[1] == "code2wav.out" and item[5] == "stream":
            outputs[item[4]].append(item[0])
        elif item[1] == "thinker.out" and item[6] == "talker_ar":
            last_thinker_token[item[4]] = item[0]
        else:
            pass
    reads = []
    for rid, times in outputs.items():
        for out_a, out_b in zip(times, times[1:]):
            gap_ms = (out_b - out_a) / 1e6
            if not low_ms <= gap_ms < high_ms:
                continue
            else:
                pass
            first, last = bisect.bisect_left(step_ns, out_a), bisect.bisect_right(
                step_ns, out_b
            )
            window = talker_steps[first:last]
            periods = [step_ns[i + 1] - step_ns[i] for i in range(first, last - 1)]
            reach = bisect.bisect_right(thinker_starts, out_b)
            inside = [
                step
                for step in thinker_steps[max(0, reach - 400) : reach]
                if step["ns"] > out_a
            ]
            busy_ns = sum(
                min(step["ns"], out_b) - max(step["ns0"], out_a) for step in inside
            )
            reads.append(
                {
                    "gap_ms": gap_ms,
                    "talker_steps": len(window),
                    "with_stream": sum(rid in step["rids"] for step in window),
                    "longest_talker_period_ms": (
                        max(periods) / 1e6 if periods else gap_ms
                    ),
                    "thinker_sending": last_thinker_token.get(rid, 0) >= out_b,
                    "thinker_busy": busy_ns / (out_b - out_a),
                    "thinker_longest_ms": max(
                        ((step["ns"] - step["ns0"]) / 1e6 for step in inside),
                        default=0.0,
                    ),
                    "prefill_tokens": sum(
                        step.get("ext", 0)
                        for step in inside
                        if step["mode"] != "DECODE"
                    ),
                }
            )
    return reads


def summary(label: str, reads: list[dict]) -> None:
    if not reads:
        print(f"  {label}: none")
        return
    else:
        pass

    def mean(key: str) -> float:
        return statistics.fmean(float(read[key]) for read in reads)

    print(
        f"  {label}: {len(reads)} gaps, mean {mean('gap_ms'):.0f} ms; talker steps {mean('talker_steps'):.1f}, with the stream "
        f"{mean('with_stream'):.1f}; longest talker period {mean('longest_talker_period_ms'):.0f} ms; thinker still sending "
        f"{100 * mean('thinker_sending'):.0f} %; thinker busy {100 * mean('thinker_busy'):.0f} %, longest step "
        f"{mean('thinker_longest_ms'):.0f} ms, prefill tokens {mean('prefill_tokens'):.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--min-ms", type=float, default=600.0)
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    for boot in args.boots:
        print(f"== {boot} ({args.cell})")
        summary(
            f"gaps >= {args.min_ms:.0f} ms",
            gap_reads(boot, args.cell, args.min_ms, 1e9),
        )
        summary("gaps < 300 ms", gap_reads(boot, args.cell, 0.0, 300.0))


if __name__ == "__main__":
    main()
