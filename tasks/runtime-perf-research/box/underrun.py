import glob
import json
import statistics
import sys


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


for label, pattern in zip(sys.argv[1::2], sys.argv[2::2]):
    stalls = []
    stalled = 0
    total = 0
    chunk_counts = []
    for path in sorted(glob.glob(pattern)):
        for request in json.load(open(path))["per_request"]:
            if not request["is_success"] or request["inter_chunk_s"] is None:
                continue
            gaps = request["inter_chunk_s"]
            chunks = len(gaps) + 1
            chunk_counts.append(chunks)
            chunk_duration = request["audio_duration_s"] / chunks
            arrival = 0.0
            worst = 0.0
            for index, gap in enumerate(gaps, start=1):
                arrival += gap
                worst = max(worst, arrival - index * chunk_duration)
            stalls.append(worst)
            stalled += worst > 0
            total += 1
    print(f"{label}: requests {total}, chunks mean {statistics.mean(chunk_counts):.1f}, requests with a stall {stalled} ({stalled / total * 100:.1f} %), worst stall mean {statistics.mean(stalls) * 1000:.0f} ms, p95 {quantile(stalls, 0.95) * 1000:.0f} ms, p99 {quantile(stalls, 0.99) * 1000:.0f} ms")
