"""SeedTTS quality of #2861's rows with band-limited resampling: main against the branch, both arms resampling as SGLang.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/resample_quality_check.py --arms upk,phu
    --out /data/pr2800/<run>
run221's seeded c1 gave the branch more outputs above 50 % WER (7 against 2) and a lower SIM (6.33 to 6.36 against
6.42). Every SeedTTS EN reference clip is 24 kHz, and omni's resample_linear (np.interp) aliases it on the way to the
thinker, which the rows carry to the talker and main's zero rows never did. Both arms' servers here load
torchaudio_resample/sitecustomize.py, which resamples as SGLang's load_audio does; the ASR scorer runs unhooked, as in
run221. One round, the arms on cards 0 and 1 at the same time: SeedTTS EN streamed voice clone, seeded c1 on the full
corpus, WER and SIM scored (pair_check.py). Writes <out>/summary.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

CARD_ENV = pair_check.card_env
SCORE = pair_check.score


def card_env(card: int, tree_path: Path) -> dict[str, str]:
    env = CARD_ENV(card, tree_path)
    env["PYTHONPATH"] = f"{BOX / 'torchaudio_resample'}:{tree_path}"
    return env


def score(*args: object, **kwargs: object) -> None:
    pair_check.card_env = CARD_ENV
    SCORE(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True, help="main,branch")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main_arm, branch_arm = args.arms.split(",")
    pair_check.card_env = card_env
    pair_check.score = score
    sys.argv = [
        "pair_check.py", "--out", args.out, "--rounds", f"{main_arm}@0,{branch_arm}@1",
        "--passes", "c1_seeded_full", "--score-passes", "c1_seeded_full",
    ]
    pair_check.main()


if __name__ == "__main__":
    main()
