"""FlashInfer's tails (p_fi) from the pair's own records, no rerun (LIFECYCLE_20261016.md 11.5), with the functions of
a2_tails_c20261016.py: paired per request first audio and latency against the 10-16 control (p_a3's unseeded c16, a
same code spread), the latency tail split at c16 and c64, c16 first audio split into the AR's share and the rest, and
the vocoder's steps per cell.

usage: python3 fi_tails_c20261017.py <copy of /data/c20261017> <copy of /data/c20261016 second box>
       (artifacts/cosyvoice-c20261017 artifacts/cosyvoice-c20261016b)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from a2_tails_c20261016 import first_audio_split, paired, tail, vocoder_steps  # noqa: E402

root, control = Path(sys.argv[1]), Path(sys.argv[2])
print("1. paired first audio and latency")
for key in ("audio_ttfp_s", "latency_s"):
    paired(control, "p_a3", "c16", key)
    for cell in ("c1u", "c16", "c64"):
        paired(root, "p_fi", cell, key)
print("2. latency tail split")
for cell in ("c16", "c64"):
    tail(root, "p_fi", cell)
print("3. c16 first audio split")
first_audio_split(control, "p_a3")
first_audio_split(root, "p_fi")
print("4. vocoder steps")
vocoder_steps(root, "p_fi")
