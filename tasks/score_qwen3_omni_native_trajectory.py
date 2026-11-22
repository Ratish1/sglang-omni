# SPDX-License-Identifier: Apache-2.0
"""Score captured SeedTTS speech with the existing pinned ASR procedure."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from qwen3_omni_corpus_support import (
    SEEDTTS_SAMPLE_COUNT,
    SERVER_TIMEOUT_SECONDS,
    server_command,
)
from score_qwen3_omni_rounded_add_screen import score_seedtts

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH
from sglang_omni.utils import find_available_port


def main(root: Path, checkpoint: Path) -> None:
    arms = (root / "first", root / "second")
    waveforms = sorted(
        waveform for arm in arms for waveform in (arm / "seedtts/audio").glob("*.wav")
    )
    assert len(waveforms) == len(arms) * SEEDTTS_SAMPLE_COUNT
    with (root / "waveform-hashes.json").open("x") as receipt:
        json.dump(
            {
                str(waveform.relative_to(root)): sha256(
                    waveform.read_bytes()
                ).hexdigest()
                for waveform in waveforms
            },
            receipt,
        )
    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        root / "asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        for arm in arms:
            score_seedtts(arm, port)
    finally:
        stop_server(server)
    summary = {}
    for arm in arms:
        result = json.loads((arm / "seedtts/wer_results.json").read_text())
        selected_samples = {path.name for path in (arm / "captures").iterdir()}
        summary[arm.name] = {
            "summary": result["summary"],
            "selected_samples": [
                row for row in result["per_sample"] if row["id"] in selected_samples
            ],
        }
    with (root / "trajectory-wer.json").open("x") as receipt:
        json.dump(summary, receipt, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint)
else:
    pass
