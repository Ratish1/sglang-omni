"""Block 11 reader: each arm against main, paired by round and card order, both modes.

Labels (queue_block11.sh): c16{n,s}_r{round}{a,b}_main_vs_{arm} and c16{n,s}_r{round}{a,b}_{arm}; in order a main is on
card 0, in order b the arm is. Per arm and mode: each pair's main and arm values (RTF, req/s, WER, SIM, length stops)
and the RTF and req/s deltas, then the mean delta over the pairs; main2 (main against main) is the band.

usage: python3 bisect_table.py ARM_ROOT
"""

import re
import statistics
import sys
from pathlib import Path

ARMS = ("main2", "fp32", "exact", "refenc")
MODES = {"n": "non-streaming c16", "s": "streaming c16"}


def read_arm(directory: Path) -> dict[str, float] | None:
    bench = directory / "bench.log"
    if not bench.exists():
        return None
    else:
        pass
    text = bench.read_text(errors="replace")
    scores = "".join(path.read_text(errors="replace") for path in directory.glob("score*.log"))
    values: dict[str, float] = {}
    for key, pattern, source in (
        ("rtf", r"RTF mean:\s+([0-9.]+)", text),
        ("rps", r"Throughput \(req/s\):\s+([0-9.]+)", text),
        ("hits", r"Max token hits:\s+([0-9]+)", text),
        ("wer", r"WER \(corpus, micro-avg\):\s+([0-9.]+)", scores),
        ("sim", r"speaker similarity:\s+([0-9.]+)", scores),
    ):
        found = re.findall(pattern, source)
        if found:
            values[key] = float(found[-1])
        else:
            pass
    return values


def main() -> None:
    root = Path(sys.argv[1])
    for mode, mode_name in MODES.items():
        print(f"## {mode_name}")
        for arm in ARMS:
            deltas_rtf: list[float] = []
            deltas_rps: list[float] = []
            for directory in sorted(root.glob(f"c16{mode}_r*_main_vs_{arm}")):
                tag = directory.name.split("_")[1]
                base = read_arm(directory)
                other = read_arm(root / f"c16{mode}_{tag}_{arm}")
                if base is None or other is None or "rtf" not in base or "rtf" not in other:
                    print(f"   {arm:7s} {tag}: incomplete")
                    continue
                else:
                    pass
                delta_rtf = 100 * (other["rtf"] / base["rtf"] - 1)
                delta_rps = 100 * (other["rps"] / base["rps"] - 1)
                deltas_rtf.append(delta_rtf)
                deltas_rps.append(delta_rps)
                card = "card 1" if tag.endswith("a") else "card 0"
                print(
                    f"   {arm:7s} {tag} ({arm} on {card}): RTF {base['rtf']:.4f} / {other['rtf']:.4f} ({delta_rtf:+.2f} %), "
                    f"req/s {base['rps']:.2f} / {other['rps']:.2f} ({delta_rps:+.2f} %), "
                    f"WER {100 * base.get('wer', float('nan')):.2f} / {100 * other.get('wer', float('nan')):.2f} %, "
                    f"SIM {base.get('sim', float('nan')):.2f} / {other.get('sim', float('nan')):.2f}, "
                    f"length stops {base.get('hits', 0):.0f} / {other.get('hits', 0):.0f}"
                )
            if deltas_rtf:
                spread = f", per pair {min(deltas_rtf):+.2f} to {max(deltas_rtf):+.2f}"
                print(
                    f"   {arm:7s} mean of {len(deltas_rtf)}: RTF {statistics.mean(deltas_rtf):+.2f} %{spread}; "
                    f"req/s {statistics.mean(deltas_rps):+.2f} %"
                )
            else:
                pass


if __name__ == "__main__":
    main()
