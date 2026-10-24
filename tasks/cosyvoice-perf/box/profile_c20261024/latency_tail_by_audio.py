"""Whether a pair's latency tail moves with the audio its arms drew (10-24, #3129's b16 latency p99): per order and arm,
latency p99, the slowest 1 % of requests with their mean audio seconds and completion tokens, RTF p50 and p99 over
every request, and over the requests whose completion token count is equal in both arms (same tokens, so the same
work) the paired latency change at the median and among the slowest 1 % of the first arm. Order a puts the first arm
on gpu0, order b on gpu1.
usage: python3 latency_tail_by_audio.py <pair dir> <cell> [<cell> ...]"""

import json
import statistics
import sys
from pathlib import Path

ORDERS = (("a", ("gpu0", "gpu1")), ("b", ("gpu1", "gpu0")))


def records(path):
    with open(path) as handle:
        return {record["id"]: record for record in json.load(handle)["per_request"] if record["is_success"]}


def quantile(values, share):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def describe(name, by_id):
    latencies = [record["latency_s"] for record in by_id.values()]
    cutoff = quantile(latencies, 0.99)
    tail = [record for record in by_id.values() if record["latency_s"] >= cutoff]
    rtfs = [record["rtf"] for record in by_id.values()]
    print(
        f"  {name}: requests {len(by_id)}, latency p99 {cutoff:.3f} s, slowest {len(tail)} mean latency "
        f"{statistics.mean(r['latency_s'] for r in tail):.3f} s, audio {statistics.mean(r['audio_duration_s'] for r in tail):.2f} s, "
        f"tokens {statistics.mean(r['completion_tokens'] for r in tail):.0f}; RTF p50 {quantile(rtfs, 0.5):.4f} p99 {quantile(rtfs, 0.99):.4f}"
    )


def main():
    pair = Path(sys.argv[1])
    for cell in sys.argv[2:]:
        print(f"# {cell}")
        for order, (first_gpu, second_gpu) in ORDERS:
            first = records(pair / order / first_gpu / cell / "speed_results.json")
            second = records(pair / order / second_gpu / cell / "speed_results.json")
            print(f"order {order}")
            describe(f"first arm ({first_gpu})", first)
            describe(f"second arm ({second_gpu})", second)
            same = [key for key in first if key in second and first[key]["completion_tokens"] == second[key]["completion_tokens"]]
            if not same:
                print("  no request with equal tokens in both arms")
                continue
            changes = [second[key]["latency_s"] / first[key]["latency_s"] - 1 for key in same]
            cutoff = quantile([first[key]["latency_s"] for key in same], 0.99)
            slow = [key for key in same if first[key]["latency_s"] >= cutoff]
            slow_changes = [second[key]["latency_s"] / first[key]["latency_s"] - 1 for key in slow]
            print(
                f"  equal tokens: {len(same)} requests, paired latency change median {100 * statistics.median(changes):+.1f} %, "
                f"mean {100 * statistics.mean(changes):+.1f} %; the first arm's slowest {len(slow)} of them "
                f"{100 * statistics.mean(slow_changes):+.1f} % mean"
            )


if __name__ == "__main__":
    main()
