"""Whether the colocated thinker's KV pool holds back its admissions, from a boot's serve.log (check 5b, 11-01).

SGLang's scheduler logs every prefill batch with its token usage (of that engine's pool) and its queue. On colocated
H100 BF16 the thinker's pool is about 5,200 tokens and the talker's about 49,000, so a usage of 0.5 or more on a
prefill or decode line is the thinker's. Prints: the thinker's lines, how many carry a waiting queue, the usage
quantiles, lines with a queue at a usage of 0.9 or more (admission held by the pool), and SGLang's retraction lines.

usage: python3 thinker_admission.py SERVE_LOG
"""

from __future__ import annotations

import re
import sys

PREFILL = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+).*?token usage: ([0-9.]+), #running-req: (\d+), #queue-req: (\d+)"
)
DECODE = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([0-9.]+).*?#queue-req: (\d+)"
)


def main() -> None:
    lines = open(sys.argv[1], errors="replace").read().splitlines()
    thinker = []
    for line in lines:
        prefill = PREFILL.search(line)
        decode = DECODE.search(line)
        if prefill and float(prefill.group(3)) >= 0.5:
            thinker.append(("prefill", float(prefill.group(3)), int(prefill.group(5)), int(prefill.group(4))))
        elif decode and float(decode.group(3)) >= 0.5:
            thinker.append(("decode", float(decode.group(3)), int(decode.group(4)), int(decode.group(1))))
        else:
            pass
    retracts = [line for line in lines if "etract" in line and "KV" in line]
    print(f"lines with token usage >= 0.5 (the thinker's pool): {len(thinker)}")
    if thinker:
        usages = sorted(u for _, u, _, _ in thinker)
        print(
            "usage p50 {:.2f} p90 {:.2f} max {:.2f}".format(
                usages[len(usages) // 2], usages[int(len(usages) * 0.9)], usages[-1]
            )
        )
        held = [row for row in thinker if row[1] >= 0.9 and row[2] > 0]
        print(f"lines at usage >= 0.9 with a waiting queue: {len(held)}; queue max {max((r[2] for r in held), default=0)}; running at those lines max {max((r[3] for r in held), default=0)}")
    else:
        pass
    print(f"retraction lines: {len(retracts)}")
    for line in retracts[:5]:
        print("  " + line[:200])


if __name__ == "__main__":
    main()
