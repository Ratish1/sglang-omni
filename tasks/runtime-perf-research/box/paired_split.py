import glob
import json
import statistics
import sys


def per_prompt(pattern):
    values = {}
    for path in sorted(glob.glob(pattern)):
        data = json.load(open(path))
        if data["summary"].get("max_token_hits", 0) > 0:
            continue
        for request in data["per_request"]:
            if request["is_success"]:
                values.setdefault(request["id"], []).append(request)
    return values


run = sys.argv[1]
cell = sys.argv[2]
for order in ("r1", "r2"):
    main = per_prompt(f"{run}_{order}_a/{cell}/bench/rep[12]/speed_results.json")
    branch = per_prompt(f"{run}_{order}_b/{cell}/bench/rep[12]/speed_results.json")
    shared = [prompt for prompt in main if prompt in branch]
    first_audio = []
    after_first = []
    total = []
    duration = []
    for prompt in shared:
        main_first = statistics.mean(r["audio_ttfp_s"] for r in main[prompt])
        branch_first = statistics.mean(r["audio_ttfp_s"] for r in branch[prompt])
        main_total = statistics.mean(r["latency_s"] for r in main[prompt])
        branch_total = statistics.mean(r["latency_s"] for r in branch[prompt])
        first_audio.append((branch_first - main_first) * 1000)
        after_first.append(((branch_total - branch_first) - (main_total - main_first)) * 1000)
        total.append((branch_total - main_total) * 1000)
        duration.append(statistics.mean(r["audio_duration_s"] for r in branch[prompt]) - statistics.mean(r["audio_duration_s"] for r in main[prompt]))
    print(f"{order}: prompts {len(shared)}; branch minus main per prompt, mean / median ms: first audio {statistics.mean(first_audio):+.2f} / {statistics.median(first_audio):+.2f}, after first audio {statistics.mean(after_first):+.2f} / {statistics.median(after_first):+.2f}, latency {statistics.mean(total):+.2f} / {statistics.median(total):+.2f}; audio duration {statistics.mean(duration):+.3f} s")
