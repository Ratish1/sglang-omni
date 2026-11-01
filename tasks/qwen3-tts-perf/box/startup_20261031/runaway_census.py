import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = 0
requests = 0
rows = []
prompt_runs = {}
for speed in sorted(root.rglob("speed_results.json")):
    bench = speed.parent
    per_request = json.loads(speed.read_text())["per_request"]
    wer_path = bench / "wer_results.csv"
    wer = {}
    if wer_path.exists():
        with wer_path.open() as handle:
            wer = {row["id"]: row for row in csv.DictReader(handle)}
    else:
        pass
    runs += 1
    requests += len(per_request)
    for item in per_request:
        prompt_runs.setdefault(item["id"], []).append(str(bench.relative_to(root).parent))
        scored = wer.get(item["id"], {})
        value = float(scored["wer"]) if scored.get("wer") else None
        if item.get("finish_reason") != "stop" or (value is not None and value > 0.5) or not item["is_success"]:
            rows.append((str(bench.relative_to(root).parent), item["id"], item.get("finish_reason"), item["completion_tokens"],
                         item["audio_duration_s"], value, scored.get("whisper_text", "")[:60]))
        else:
            pass
print(f"runs {runs} requests {requests} flagged {len(rows)}")
for row in rows:
    print(" | ".join(str(field) for field in row), f"| runs of this prompt {len(prompt_runs[row[1]])}")
