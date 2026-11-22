# SPDX-License-Identifier: Apache-2.0
"""Share the pinned corpus commands across stock and instrumented checkouts."""

import subprocess
import sys
import time
from pathlib import Path

from benchmarks.eval.asr_profiling import start_request_profile, stop_request_profile
from sglang_omni.profiler.views import assess_completeness

MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)
MMSU_SAMPLE_COUNT = 40
CLIENT_CONCURRENCY = 16
REQUEST_SEED = 42
SEEDTTS_SAMPLE_COUNT = 1088
SERVER_TIMEOUT_SECONDS = 600
PROFILE_FINALIZATION_TIMEOUT_SECONDS = 30
PROFILE_EVENT_BUFFER_CAPACITY = 131_072
PROFILE_PROCESS_COUNT = 7


def server_command() -> list[str]:
    return [
        sys.executable,
        "-P",
        "-m",
        "sglang_omni.cli",
        "serve",
        "--host",
        "127.0.0.1",
    ]


def seedtts_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--voice-clone",
        "--max-concurrency",
        str(CLIENT_CONCURRENCY),
        "--seed",
        str(REQUEST_SEED),
        "--disable-tqdm",
    ]


def run_seedtts_profile(
    command: list[str], server_port: int, run_directory: Path
) -> None:
    base_url = f"http://127.0.0.1:{server_port}"
    event_directory = run_directory / "events"
    run_id = run_directory.name
    start_request_profile(
        base_url,
        run_id,
        str(event_directory),
        event_buffer_capacity=PROFILE_EVENT_BUFFER_CAPACITY,
    )
    try:
        subprocess.run([*command, "--record-response-ids"], check=True)
    finally:
        stop_request_profile(base_url, run_id)
    deadline_seconds = time.monotonic() + PROFILE_FINALIZATION_TIMEOUT_SECONDS
    while True:
        status = assess_completeness(event_directory)
        if status["complete"] and status["found_status_files"] == PROFILE_PROCESS_COUNT:
            return
        elif status["state"] == "incomplete" or time.monotonic() >= deadline_seconds:
            raise RuntimeError(f"Incomplete SeedTTS request events: {status}")
        else:
            time.sleep(0.25)
