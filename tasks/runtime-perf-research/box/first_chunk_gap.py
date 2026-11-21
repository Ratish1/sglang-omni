#!/usr/bin/env python3
"""Each arm's chunk gaps by position and the first gap against the thinker's text, pooled over the arm's boots.

usage: python3 first_chunk_gap.py --cell c64_events <run dir> <arm>=<boot>,<boot> [<arm>=<boot>,...]

Per arm:
  gaps         client side (speed_results.json): every gap's mean, p50, p90, p99; the first gap and later gaps apart;
               the share of the slowest 1 % that are first gaps; requests with any playback underrun
  first gap    server side (hop probe queue records, boots run with --probe): from the request's first thinker token to
               code2wav's first and second outputs, the share of the reply's thinker tokens sent by the first output,
               the thinker's last token after it, and the thinker's mean token gap per request
"""

import argparse
import bisect
import glob
import json
import os
import statistics


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def client_gaps(run: str, boots: list[str], cell: str) -> None:
    gaps, firsts, later, underruns = [], [], [], 0
    for boot in boots:
        with open(
            os.path.join(run, boot, cell, "bench", "speed_results.json")
        ) as source:
            data = json.load(source)
        for request in data["per_request"]:
            request_gaps = request["inter_chunk_s"] or []
            gaps.extend((gap, index) for index, gap in enumerate(request_gaps))
            if request_gaps:
                firsts.append(request_gaps[0])
                later.extend(request_gaps[1:])
            else:
                pass
            underruns += (request["max_playback_underrun_s"] or 0) > 0
    every = [gap for gap, _ in gaps]
    p99 = quantile(every, 0.99)
    slowest = [index for gap, index in gaps if gap >= p99]
    print(
        f"  gaps {len(every)}: mean {1e3 * statistics.fmean(every):.1f} p50 {1e3 * quantile(every, 0.5):.1f} "
        f"p90 {1e3 * quantile(every, 0.9):.1f} p99 {1e3 * p99:.1f} ms; first gap mean {1e3 * statistics.fmean(firsts):.1f} "
        f"p99 {1e3 * quantile(firsts, 0.99):.1f}; later mean {1e3 * statistics.fmean(later):.1f} "
        f"p99 {1e3 * quantile(later, 0.99):.1f}; slowest 1 % first gaps {100 * slowest.count(0) / len(slowest):.0f} %; "
        f"requests with underrun {underruns}"
    )


def server_first_gap(run: str, boots: list[str], cell: str) -> None:
    to_first, to_second, sent_share, last_after, token_gaps = [], [], [], [], []
    for boot in boots:
        with open(os.path.join(run, boot, cell, "window.json")) as source:
            window = json.load(source)
        start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
        tokens, outputs = {}, {}
        for path in glob.glob(os.path.join(run, boot, "probe", "*.jsonl")):
            with open(path) as source:
                for line in source:
                    if not line.startswith("["):
                        continue
                    else:
                        pass
                    item = json.loads(line)
                    if item[2] != "put" or item[5] != "stream":
                        continue
                    elif item[1] == "thinker.out" and item[6] == "talker_ar":
                        tokens.setdefault(item[4], []).append(item[0])
                    elif item[1] == "code2wav.out":
                        outputs.setdefault(item[4], []).append(item[0])
                    else:
                        pass
        for rid, output_times in outputs.items():
            output_times.sort()
            if (
                len(output_times) < 2
                or not start_ns <= output_times[0] <= end_ns
                or rid not in tokens
            ):
                continue
            else:
                pass
            token_times = sorted(tokens[rid])
            to_first.append((output_times[0] - token_times[0]) / 1e6)
            to_second.append((output_times[1] - token_times[0]) / 1e6)
            sent_share.append(
                bisect.bisect_right(token_times, output_times[0]) / len(token_times)
            )
            last_after.append((token_times[-1] - output_times[0]) / 1e6)
            if len(token_times) > 1:
                token_gaps.append(
                    (token_times[-1] - token_times[0]) / 1e6 / (len(token_times) - 1)
                )
            else:
                pass
    if not to_first:
        print("  first gap: no probe records")
        return
    else:
        pass
    print(
        f"  first gap, {len(to_first)} requests, from the first thinker token: chunk 1 {statistics.fmean(to_first):.1f} ms, "
        f"chunk 2 {statistics.fmean(to_second):.1f} ms; reply sent by chunk 1 {100 * statistics.fmean(sent_share):.0f} %; "
        f"thinker's last token after chunk 1 {statistics.fmean(last_after):.1f} ms (p90 {quantile(last_after, 0.9):.1f}); "
        f"thinker token gap {statistics.fmean(token_gaps):.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("run")
    parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    for arm in args.arms:
        name, boots = arm.split("=", 1)
        print(f"{name}:")
        client_gaps(args.run, boots.split(","), args.cell)
        server_first_gap(args.run, boots.split(","), args.cell)


if __name__ == "__main__":
    main()
