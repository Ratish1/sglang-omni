"""Python self time by function in a torch profiler trace recorded with stacks (mapping mode): each
python_function event's duration minus its direct children's, summed per function, per thread.
usage: python trace_python_self.py TRACE.json.gz [--top 30] [--thread SUBSTRING]"""

import argparse
import gzip
import json
from collections import Counter, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("trace")
parser.add_argument("--top", type=int, default=30)
parser.add_argument("--thread", default="")
args = parser.parse_args()

with gzip.open(args.trace) as trace:
    events = json.load(trace)["traceEvents"]
thread_names = {
    (e["pid"], e["tid"]): e["args"]["name"]
    for e in events
    if e.get("ph") == "M" and e.get("name") == "thread_name"
}
by_thread = defaultdict(list)
for event in events:
    if event.get("ph") == "X" and event.get("cat") in ("python_function", "cpu_op"):
        by_thread[(event["pid"], event["tid"])].append(event)
for key, thread_events in by_thread.items():
    name = str(thread_names.get(key, key))
    if args.thread not in name:
        continue
    thread_events.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
    self_time = Counter()
    stack: list[tuple[dict, float]] = []
    for event in thread_events:
        end = event["ts"] + event.get("dur", 0)
        while stack and stack[-1][0]["ts"] + stack[-1][0].get("dur", 0) <= event["ts"]:
            stack.pop()
        stack.append((event, 0.0))
        self_time[(event["cat"], event["name"])] += event.get("dur", 0)
    print(f"== thread {name} events {len(thread_events)}")
    children = Counter()
    stack = []
    for event in thread_events:
        while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) <= event["ts"]:
            stack.pop()
        if stack:
            children[(stack[-1]["cat"], stack[-1]["name"])] += event.get("dur", 0)
        stack.append(event)
    exclusive = Counter({key: self_time[key] - children[key] for key in self_time})
    for (cat, function), us in exclusive.most_common(args.top):
        print(f"  {us / 1e3:10.1f} ms  {cat:15s} {function[:140]}")
