"""The component suite's reports and the numerics ledger as one table: match against HF FP32 per component.

Match is 100 * (1 - relative L2) for tensors and 100 * (1 - total variation) for distributions. For the suite each
row shows ours and HF BF16 (the floor), mean and worst case, and the count of cases under 99 %. For the ledger each row
shows the error a sub-component adds alone (local) and the chain's error at that point (accumulated).

usage: python3 match_report.py SUITE_REPORT_DIR [LEDGER.json]
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BAR = 99.0


def percent(relative: float) -> float:
    return 100.0 * (1.0 - relative)


def group_of(case: str) -> str:
    parts = case.split("/")
    if parts[-1] in ("decode_inputs", "talker_hidden", "code0_logits", "served_audio"):
        return parts[-1]
    if parts[0] in ("graph", "eager"):
        return "batched_" + parts[0] if parts[1].startswith("batch") else "single_" + parts[0]
    if parts[0] in ("whole", "walk"):
        return parts[0]
    return re.sub(r"\d+", "*", parts[0])


def suite_rows(report_dir: Path) -> list[tuple[str, str, int, float, float, float, float, int]]:
    rows = []
    for path in sorted(report_dir.glob("*.json")):
        data = json.loads(path.read_text())
        records = data.get("records") if isinstance(data, dict) else data
        if not records or not isinstance(records, list) or "omni" not in records[0]:
            continue
        groups = defaultdict(list)
        for record in records:
            groups[group_of(record["case"])].append(record)
        for group, entries in sorted(groups.items()):
            ours = [percent(entry["omni"]["relative_l2"]) for entry in entries]
            floor = [percent(entry["floor"]["relative_l2"]) for entry in entries]
            rows.append(
                (
                    path.stem,
                    group,
                    len(entries),
                    statistics.fmean(ours),
                    min(ours),
                    statistics.fmean(floor),
                    min(floor),
                    sum(value < BAR for value in ours),
                )
            )
    return rows


def main() -> None:
    report_dir = Path(sys.argv[1])
    print(f"{'component':<18}{'group':<22}{'cases':>6}{'ours mean':>11}{'ours worst':>11}{'BF16 mean':>11}{'BF16 worst':>11}{'<99%':>6}")
    for stem, group, count, mean, worst, floor_mean, floor_worst, under in suite_rows(report_dir):
        print(f"{stem:<18}{group:<22}{count:>6}{mean:>11.2f}{worst:>11.2f}{floor_mean:>11.2f}{floor_worst:>11.2f}{under:>6}")
    for name in ("preprocessing.json", "talker.json"):
        path = report_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if name == "preprocessing.json":
            identical = [r for r in data if r["prompt_equal"] and r["input_ids_equal"] and r["mask_equal"] and r["features_max_abs"] == 0.0]
            errors = [r["features_relative_l2"] for r in data]
            print(
                f"\npreprocessing: {len(identical)} of {len(data)} identical to the reference; mel match mean "
                f"{percent(statistics.fmean(errors)):.2f}, worst {percent(max(errors)):.2f}; prompt ids equal "
                f"{sum(r['input_ids_equal'] for r in data)} of {len(data)}"
            )
        else:
            requests = data["requests"]
            print(
                f"\ntalker: first code top-1 agreement {statistics.fmean(r['code0_top1_agreement'] for r in requests) * 100:.2f}; "
                f"residual argmax agreement ours {100 * (1 - sum(r['residual_departures'] for r in requests) / sum(r['residual_codes'] for r in requests)):.2f}, "
                f"HF BF16 {100 * (1 - sum(r['floor_residual_departures'] for r in requests) / sum(r['residual_codes'] for r in requests)):.2f}; "
                f"window walk against whole decode match {percent(statistics.fmean(r['walk_vs_whole_relative_l2'] for r in requests)):.2f}"
            )
    if len(sys.argv) > 2:
        ledger = json.loads(Path(sys.argv[2]).read_text())
        print("\nHF BF16 against HF FP32, each sub-component alone (local); fp32_card is the truth's own platform error")
        print(f"{'sub-component':<62}{'match':>8}{'row mean':>10}{'row p99':>9}{'row worst':>10}{'fp32_card':>11}")
        for key, row in sorted(ledger["local"].items()):
            print(
                f"{key:<62}{percent(row['whole']):>8.2f}{percent(row['row_mean']):>10.2f}{percent(row['row_p99']):>9.2f}"
                f"{percent(row['row_max']):>10.2f}{row['fp32_card']:>11.1e}"
            )
        print("\nHF BF16 chain against HF FP32, accumulated")
        print(f"{'point':<46}{'match':>8}{'row mean':>10}{'row median':>11}{'row p99':>9}{'row worst':>10}")
        for key, row in sorted(ledger["accumulated"].items()):
            if re.search(r"layer_(\d+)", key) and not re.search(r"layer_(00|05|11|12|19|23|24|35|47)", key):
                continue
            print(
                f"{key:<46}{percent(row['whole']):>8.2f}{percent(row['row_mean']):>10.2f}"
                f"{percent(row.get('row_median', row['row_mean'])):>11.2f}{percent(row['row_p99']):>9.2f}{percent(row['row_max']):>10.2f}"
            )
        print("\nOutput distributions, HF BF16 against HF FP32")
        for key, row in sorted(ledger["distributions"].items()):
            print(f"{key:<30}" + "  ".join(f"{field} {value:.4f}" for field, value in row.items()))


if __name__ == "__main__":
    main()
