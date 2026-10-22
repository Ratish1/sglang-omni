"""run220: the talker's prompt rows kept per KV slot in a pinned host store (#2861 on our branch, dbcfec3d3, pht)
against main (272845a9c, upk). run219 measured phs, whose prefix cap cost Video-AMME's repeated videos their cached
prefixes (+2 s a request); pht drops the cap.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/talker_rows_queue.py
Trees are pair_check.py's, fetched by full commit; a tree hash unlike the Mac's or a failed suite stops the queue.
1. pht's suites on both cards at once: the store, the shared thinker runner and its MiniCPM-o subclass, the engine
   builder (card 0); Qwen3-Omni (card 1).
2. prompt_rows_probe.py on run218's phs records with HF fed the server's audio loader (card 1), against run219's
   librosa read of the same records.
3. prompt_rows_probe.py on pht with the server's audio loader: 40 MMSU requests, then the first 8 again, whose rows
   come from the store through their cached prompts.
4. videoamme_talker_tp2.py: the CI's FP8 thinker TP2 Video-AMME talker, upk then pht.
5. mmsu_talker_pair.py: the full mmsu-ci-2000 with the talker, upk on card 0 and pht on card 1 at once.
6. pair_check.py, four rounds on both cards with the arms swapped: SeedTTS c16 and c64 unseeded, c16 and c64 seeded,
   seeded c1 on the full corpus every round; WER and SIM scored on the c16 and seeded c1 passes.
Writes /data/pr2800/run220 and talker_rows.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402
from pair_check import tree  # noqa: E402

DATA = Path("/data/pr2800")
OUT = DATA / "run220"
LOG = DATA / "talker_rows.log"
EXPECTED_TREES = {"upk": "1620e868", "pht": "50962505"}
SUITES = {
    "0": [
        "tests/unit_test/scheduling/test_prompt_hidden.py",
        "tests/unit_test/model_runner",
        "tests/unit_test/minicpm_o/test_thinker_model_runner.py",
        "tests/unit_test/cpu/test_engine_builder.py",
    ],
    "1": ["tests/unit_test/qwen3_omni"],
}
ROUNDS = "upk@0,pht@1;pht@0,upk@1;upk@0,pht@1;pht@0,upk@1"
PASSES = "c16,c64,c16_seeded,c64_seeded,c1_seeded_full"
SCORED = "c16,c1_seeded_full"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a") as handle:
        handle.write(line + "\n")


def step(
    name: str, command: list[str], env: dict[str, str], keep: tuple[str, ...]
) -> None:
    rc = subprocess.run(
        command,
        cwd=BOX,
        env=env,
        stdout=open(OUT / f"{name}.log", "w"),
        stderr=subprocess.STDOUT,
    ).returncode
    log(f"{name} rc {rc}")
    for line in (OUT / f"{name}.log").read_text().splitlines():
        if any(text in line for text in (*keep, "Traceback")):
            log(f"  {line[:300]}")
        else:
            pass


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    box_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BOX, capture_output=True, text=True
    ).stdout.strip()
    log(f"pins {cpus} node {node}; box branch {box_head[:9]}")
    paths = {name: tree(name) for name in EXPECTED_TREES}
    for name, path in paths.items():
        tree_hash = subprocess.run(
            ["git", "rev-parse", "HEAD:sglang_omni"],
            cwd=path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "sglang_omni"], cwd=path
        ).returncode
        log(f"{name} sglang_omni {tree_hash} expected {EXPECTED_TREES[name]} dirty {dirty}")
        if not tree_hash.startswith(EXPECTED_TREES[name]) or dirty:
            log("stopped: a tree is not the Mac's")
            return
        else:
            pass
    suites = {
        card: subprocess.Popen(
            [
                "python3",
                str(BOX / "suite_run.py"),
                "--tree",
                str(paths["pht"]),
                "--card",
                card,
                "--out",
                str(OUT / f"units_card{card}.log"),
                *tests,
            ],
            env=env,
            stdout=open(OUT / f"units_card{card}.out", "w"),
            stderr=subprocess.STDOUT,
        )
        for card, tests in SUITES.items()
    }
    failed = False
    for card, process in suites.items():
        rc = process.wait()
        log(f"units card {card} rc {rc}")
        for line in (OUT / f"units_card{card}.log").read_text().splitlines()[-3:]:
            log(f"  {line}")
        failed = failed or rc != 0
    if failed:
        log("stopped: a suite failed")
        return
    else:
        pass
    step(
        "probe_phs_omni",
        [
            "python3",
            str(BOX / "prompt_rows_probe.py"),
            "--arm",
            "phs",
            "--records",
            str(DATA / "run218" / "probe" / "dump"),
            "--omni-audio",
            "--out",
            str(OUT / "probe_phs_omni"),
        ],
        env,
        ("summary", "talker prompt prefills"),
    )
    step(
        "probe",
        [
            "python3",
            str(BOX / "prompt_rows_probe.py"),
            "--arm",
            "pht",
            "--omni-audio",
            "--repeats",
            "8",
            "--out",
            str(OUT / "probe"),
        ],
        env,
        ("summary", "talker prompt prefills", "repeats with", "client rc"),
    )
    step(
        "videoamme",
        [
            "python3",
            str(BOX / "videoamme_talker_tp2.py"),
            "--arms",
            "upk,pht",
            "--out",
            str(OUT / "videoamme"),
        ],
        env,
        ("client rc", "score rc", "accuracy"),
    )
    step(
        "mmsu",
        [
            "python3",
            str(BOX / "mmsu_talker_pair.py"),
            "--arms",
            "upk@0,pht@1",
            "--out",
            str(OUT / "mmsu"),
        ],
        env,
        ("client rc", "score rc", "accuracy"),
    )
    step(
        "seedtts",
        [
            "python3",
            str(BOX / "pair_check.py"),
            "--out",
            str(OUT / "seedtts"),
            "--rounds",
            ROUNDS,
            "--passes",
            PASSES,
            "--c1-every-round",
            "--score-passes",
            SCORED,
        ],
        env,
        ("failed",),
    )
    log("done")


if __name__ == "__main__":
    main()
