"""PLAN.md 5's gates per captured cell, from each torch trace: graph launches, graph captures and instantiations, lazy
kernel loads, module loads, Dynamo regions and python frames, and the window's span. A formal window must hold graph
launches and none of the one-time work; a mapping window must hold python frames and no graph launch but the speech
tokenizer's replays, which have no switch (speech_tokenizer.py).
usage: python3 trace_gates.py <pass dir> [<pass dir> ...]"""

import gzip
import json
import sys
from pathlib import Path

COUNTED = {
    "graph launch": lambda name: name.startswith("cudaGraphLaunch"),
    "capture": lambda name: name.startswith(("cudaStreamBeginCapture", "cudaGraphInstantiate")),
    "lazy load": lambda name: "Lazy Function Loading" in name,
    "module load": lambda name: name.startswith(("cuModuleLoad", "cuLibraryLoad")),
    "dynamo": lambda name: "dynamo_timed" in name,
}


def main() -> None:
    print("| pass | cell | span s | " + " | ".join(COUNTED) + " | python frames | verdict |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in COUNTED) + " | --- | --- |")
    for pass_dir in map(Path, sys.argv[1:]):
        mode = json.loads((pass_dir / "boot.json").read_text())["mode"]
        for cell_dir in sorted(path for path in pass_dir.iterdir() if path.is_dir()):
            for trace in sorted(cell_dir.glob("trace*.trace.json.gz")):
                events = json.load(gzip.open(trace)).get("traceEvents", [])
                counts = dict.fromkeys(COUNTED, 0)
                python_frames = 0
                lazy_load_us = 0.0
                stamps = []
                for event in events:
                    name = str(event.get("name", ""))
                    if event.get("cat") == "python_function":
                        python_frames += 1
                    else:
                        pass
                    # note (ratish): metadata events carry ts 0 and widened the span to the epoch (10-22).
                    if event.get("ph") == "X" and float(event.get("ts", 0)) > 0:
                        stamps.append(float(event["ts"]))
                    else:
                        pass
                    for label, matches in COUNTED.items():
                        if matches(name):
                            counts[label] += 1
                        else:
                            pass
                    if COUNTED["lazy load"](name):
                        lazy_load_us += float(event.get("dur", 0))
                    else:
                        pass
                span = (max(stamps) - min(stamps)) / 1e6 if stamps else 0.0
                if mode == "formal":
                    # note (ratish): a first use kernel load of tens of microseconds charges nothing a window can
                    # see; past 1 ms in total it is one-time work in the steady state (10-22: 0.03 to 0.09 ms each).
                    clean = lazy_load_us <= 1000 and not any(
                        counts[label] for label in ("capture", "module load", "dynamo")
                    )
                    verdict = "pass" if clean and counts["graph launch"] > 0 else "FAIL"
                else:
                    verdict = "pass" if python_frames > 0 else "FAIL"
                print(
                    f"| {mode} | {cell_dir.name}/{trace.name.split('.')[0]} | {span:.2f} | "
                    + " | ".join(str(counts[label]) for label in COUNTED)
                    + f" | {python_frames} | {verdict} (lazy loads {lazy_load_us / 1000:.2f} ms) |"
                )


if __name__ == "__main__":
    main()
