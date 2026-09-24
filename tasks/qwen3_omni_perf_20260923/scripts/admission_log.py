"""Summarize the talker admission log that memdiag writes with MEMDIAG_ADMISSION=1.

Pairs each request's build line (answer tokens, prompt rows) with its finish line (frames),
then prints the frame and frames-per-answer-token distributions, the admission verdicts,
the budget terms of rejected attempts, and the retractions.

usage: python admission_log.py <serve.log> [<serve.log> ...]
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter

FIELD = re.compile(r"(\w+)=(\S+)")


def fields(line: str) -> dict[str, str]:
    return dict(FIELD.findall(line.split("MEMDIAG admission", 1)[1]))


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    else:
        pass
    ordered = sorted(values)
    picks = {
        name: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        for name, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
    }
    body = " ".join(f"{name}={value:.2f}" for name, value in picks.items())
    return f"n={len(values)} mean={statistics.fmean(values):.2f} {body} max={ordered[-1]:.2f}"


def summarize(path: str) -> None:
    builds: dict[str, dict[str, str]] = {}
    finishes: dict[str, dict[str, str]] = {}
    verdicts: Counter[str] = Counter()
    rejected: list[dict[str, str]] = []
    retractions: list[dict[str, str]] = []
    with open(path, errors="replace") as log_file:
        for line in log_file:
            if "MEMDIAG admission" not in line:
                continue
            else:
                pass
            values = fields(line)
            if "admission build" in line:
                builds[values["rid"]] = values
            elif "admission finish" in line:
                finishes[values["rid"]] = values
            elif "admission try" in line:
                verdicts[values["result"]] += 1
                if values["result"] == "NO_TOKEN":
                    rejected.append(values)
                else:
                    pass
            elif "admission retract" in line:
                retractions.append(values)
            else:
                pass

    paired = [rid for rid in finishes if rid in builds]
    frames = [float(finishes[rid]["frames"]) for rid in paired]
    answers = [float(builds[rid]["answer_tokens"]) for rid in paired]
    prompts = [float(builds[rid]["prompt_rows"]) for rid in paired]
    per_token = [f / a for f, a in zip(frames, answers) if a > 0]
    reasons = Counter(finishes[rid]["reason"] for rid in paired)
    print(f"== {path}")
    print(f"requests built={len(builds)} finished={len(finishes)} paired={len(paired)}")
    print(f"finish reasons {dict(reasons)}")
    print(f"frames            {quantiles(frames)}")
    print(f"answer tokens     {quantiles(answers)}")
    print(f"prompt rows       {quantiles(prompts)}")
    print(f"frames per token  {quantiles(per_token)}")
    above = sum(1 for value in frames if value > 256)
    print(f"frames above 256: {above} of {len(frames)}")
    print(f"admission verdicts {dict(verdicts)}")
    for values in rejected[:5]:
        print(f"  rejected {values}")
    print(f"retractions {len(retractions)}")
    for values in retractions[:5]:
        print(f"  retract {values}")


def main() -> None:
    for path in sys.argv[1:]:
        summarize(path)


if __name__ == "__main__":
    main()
