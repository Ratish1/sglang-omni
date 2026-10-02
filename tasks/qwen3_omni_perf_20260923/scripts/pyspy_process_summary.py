"""Summarize a py-spy --subprocesses --format raw record per process: samples, the share of samples
not idle (a wait, poll, sleep or lock leaf), and the top frames by inclusive presence and by self time
among the busy samples. usage: python pyspy_process_summary.py FILE.raw [--top 20]"""

import argparse
import re
from collections import Counter, defaultdict

IDLE_LEAF = re.compile(
    r"\b(wait|_wait|poll|select|epoll|sleep|recv|recv_multipart|acquire|get|_poll|"
    r"wait_for|_recv|accept|read|readinto|join)\b \("
)

PROCESS_TAG = re.compile(r"^process \d+:")

parser = argparse.ArgumentParser()
parser.add_argument("raw")
parser.add_argument("--top", type=int, default=20)
args = parser.parse_args()

per_process: dict[str, dict] = defaultdict(
    lambda: {
        "samples": 0,
        "busy": 0,
        "inclusive": Counter(),
        "self": Counter(),
        "cmd": "",
    }
)
# A frame string that carries a newline (a command line, a thread name) splits its record across
# lines; those pieces end in no sample count and are counted here, not parsed.
broken_lines = 0
with open(args.raw, errors="replace") as raw:
    for line in raw:
        line = line.rstrip("\n")
        if not line:
            continue
        stack, _, count_text = line.rpartition(" ")
        if not count_text.isdigit():
            broken_lines += 1
            continue
        count = int(count_text)
        frames = stack.split(";")
        process_frames = [frame for frame in frames if PROCESS_TAG.match(frame)]
        code_frames = [frame for frame in frames if not PROCESS_TAG.match(frame)]
        key = process_frames[-1].split(":", 1)[0] if process_frames else "?"
        entry = per_process[key]
        entry["cmd"] = process_frames[-1][:120] if process_frames else ""
        entry["samples"] += count
        if not code_frames or IDLE_LEAF.search(code_frames[-1]):
            continue
        entry["busy"] += count
        entry["self"][code_frames[-1]] += count
        for frame in set(code_frames):
            entry["inclusive"][frame] += count

total = sum(entry["samples"] for entry in per_process.values())
for key, entry in sorted(per_process.items(), key=lambda item: -item[1]["busy"]):
    print(
        f"== {key} samples {entry['samples']} busy {entry['busy']} ({entry['busy'] / max(entry['samples'], 1):.0%}) {entry['cmd']}"
    )
    print("  inclusive (busy samples):")
    for frame, count in entry["inclusive"].most_common(args.top):
        print(f"    {count:6d} {count / max(entry['busy'], 1):6.1%}  {frame[:150]}")
    print("  self:")
    for frame, count in entry["self"].most_common(10):
        print(f"    {count:6d} {count / max(entry['busy'], 1):6.1%}  {frame[:150]}")
print(f"total samples {total}, broken record lines skipped {broken_lines}")
