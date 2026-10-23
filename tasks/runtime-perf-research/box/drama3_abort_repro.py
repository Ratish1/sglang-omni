"""Repeat the Drama 3 HTTP test's dropped-streams round against one server and print each round's frame counts.

usage (cwd and PYTHONPATH: the sglang-omni tree under test):
  python3 drama3_abort_repro.py <rounds> <out dir> [settle seconds]

Boots the test's end-of-speech-masked server once, then per round drops three streamed requests after their first event, waits the settle time, and sends one full round (MAX_RUNNING_REQUESTS requests of 20 frames). Every request must return 20 frames.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

from sglang_omni.models.fishaudio_drama3.config import DEFAULT_CODEC_PATH
from sglang_omni.models.fishaudio_drama3.stages import resolve_codec_directory
from sglang_omni.utils.connection import find_available_port
from tests.unit_test.fishaudio_drama3.http_server import MODEL_NAME
from tests.unit_test.fishaudio_drama3.test_http_e2e import (
    MAX_RUNNING_REQUESTS,
    REPO_ROOT,
    REQUEST_TIMEOUT_S,
    SAMPLES_PER_FRAME,
    STARTUP_TIMEOUT_S,
    full_round,
    write_checkpoint,
)
from tests.utils import start_server_from_cmd, stop_server


def drop_streams(base_url):
    for _ in range(3):
        with httpx.stream(
            "POST",
            f"{base_url}/v1/audio/speech",
            json={
                "model": MODEL_NAME,
                "input": "hello",
                "stream": True,
                "stream_format": "sse",
                "response_format": "pcm",
                "max_new_tokens": 2048,
            },
            timeout=REQUEST_TIMEOUT_S,
        ) as response:
            assert response.status_code == 200
            next(line for line in response.iter_lines() if line.startswith("data: "))


rounds = int(sys.argv[1])
out_dir = Path(sys.argv[2])
settle_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
out_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = out_dir / "checkpoint"
checkpoint_dir.mkdir(exist_ok=True)
write_checkpoint(checkpoint_dir)
resolve_codec_directory(DEFAULT_CODEC_PATH)
port = find_available_port()
server = start_server_from_cmd(
    [
        sys.executable,
        "-c",
        "from tests.unit_test.fishaudio_drama3.http_server import main; main()",
        "--checkpoint",
        str(checkpoint_dir),
        "--port",
        str(port),
    ],
    out_dir / "server.log",
    port,
    timeout=STARTUP_TIMEOUT_S,
    env={
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
        ),
        "HF_HUB_OFFLINE": "1",
    },
    strip_proxy=True,
)
base_url = f"http://127.0.0.1:{port}"
expected = 20 * SAMPLES_PER_FRAME
try:
    full = full_round(base_url)
    print(
        json.dumps(
            {"round": "clean", "frames": [count // SAMPLES_PER_FRAME for count in full]}
        ),
        flush=True,
    )
    for index in range(rounds):
        start = time.monotonic()
        drop_streams(base_url)
        dropped_s = time.monotonic() - start
        time.sleep(settle_s)
        counts = full_round(base_url)
        short = [
            (position, count // SAMPLES_PER_FRAME)
            for position, count in enumerate(counts)
            if count != expected
        ]
        print(
            json.dumps(
                {
                    "round": index,
                    "drop_s": round(dropped_s, 3),
                    "short": short,
                    "requests": MAX_RUNNING_REQUESTS,
                }
            ),
            flush=True,
        )
finally:
    stop_server(server)
