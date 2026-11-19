"""A PR body table for text Video-MME: per load, each arm's boot values from one pass and the change of the arm means,
then every pass's accuracy per arm.

usage: python3 video_body_table.py <run dir> "<title>" <pass, e.g. rep1> <loads, e.g. 1,16> main=boot,boot pr=boot,boot
"""

import json
import statistics
import sys
from pathlib import Path

ROWS = [
    ("QPS", "throughput_qps", 1, "{:.3f}"),
    ("latency mean (s)", "latency_mean_s", 1, "{:.2f}"),
    ("latency p95 (s)", "latency_p95_s", 1, "{:.2f}"),
]


def result(boot: Path, load: int, measured: str) -> dict:
    if measured == "events":
        path = boot / f"c{load}_events" / "bench" / "videomme_results.json"
    else:
        path = boot / f"c{load}_clean" / "bench" / measured / "videomme_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def every_pass(boot: Path, load: int) -> list[dict]:
    paths = sorted((boot / f"c{load}_clean" / "bench").glob("rep*/videomme_results.json"))
    paths.append(boot / f"c{load}_events" / "bench" / "videomme_results.json")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    run, title, measured = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    loads = [int(item) for item in sys.argv[4].split(",")]
    arms = [(spec.split("=")[0], spec.split("=")[1].split(",")) for spec in sys.argv[5:]]
    (base_name, base_boots), (pr_name, pr_boots) = arms
    print(f"| {title} | {base_name} | {pr_name} | change |")
    print("|---|---|---|---|")
    rows = [(f"{label} c{load}", key, scale, form, load) for load in loads for label, key, scale, form in ROWS]
    rows += [(f"accuracy c{load}", "accuracy", 1, "{:.2f}", load) for load in loads]
    for label, key, scale, form, load in rows:
        source = "summary" if key == "accuracy" else "speed"
        base = [result(run / boot, load, measured)[source][key] * scale for boot in base_boots]
        pr = [result(run / boot, load, measured)[source][key] * scale for boot in pr_boots]
        change = (statistics.fmean(pr) / statistics.fmean(base) - 1) * 100
        print(
            f"| {label} | {' / '.join(form.format(value) for value in base)} | "
            f"{' / '.join(form.format(value) for value in pr)} | {change:+.1f} % |"
        )
    for load in loads:
        for name, boots in arms:
            scores = [entry["summary"]["accuracy"] for boot in boots for entry in every_pass(run / boot, load)]
            print(f"accuracy c{load} {name} every pass: {' '.join(f'{score:.2f}' for score in scores)}")


if __name__ == "__main__":
    main()
