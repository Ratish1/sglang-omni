# SPDX-License-Identifier: Apache-2.0
"""E2E smoke for dots TTS through ``sgl-omni serve`` and /v1/audio/speech."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/home/kps_spark/workspace/models/dots.tts-base",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--text", default="Hello.")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    return parser.parse_args()


def _find_available_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_tail(path: Path, *, max_bytes: int = 12000) -> str:
    if not path.exists():
        return "<server log not created>"
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def _get_json(url: str, *, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, *, timeout: float) -> tuple[bytes, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("content-type", "")


def _write_server_bootstrap(path: Path) -> None:
    bootstrap = """
def main():
    from sglang_omni.cli import app

    app()


if __name__ == "__main__":
    main()
"""
    path.write_text(bootstrap.lstrip(), encoding="utf-8")


def _write_pipeline_config(path: Path, *, model_path: str) -> None:
    config = {
        "config_cls": "DotsTTSPipelineConfig",
        "model_path": model_path,
        "stages": [
            {
                "name": "preprocessing",
                "process": "pipeline",
                "factory": (
                    "sglang_omni.models.dots_tts.stages."
                    "create_preprocessing_executor"
                ),
                "next": "latent_engine",
            },
            {
                "name": "latent_engine",
                "process": "pipeline",
                "factory": (
                    "sglang_omni.models.dots_tts.stages."
                    "create_sglang_latent_engine_executor"
                ),
                "factory_args": {
                    "device": "cuda",
                    "precision": "bfloat16",
                    "server_args_overrides": {
                        "max_running_requests": 1,
                        "mem_fraction_static": 0.55,
                        "disable_cuda_graph": True,
                    },
                },
                "gpu": 0,
                "next": "vocoder",
                "stream_to": ["vocoder"],
            },
            {
                "name": "vocoder",
                "process": "pipeline",
                "factory": (
                    "sglang_omni.models.dots_tts.stages.create_vocoder_executor"
                ),
                "factory_args": {
                    "device": "cuda",
                    "precision": "bfloat16",
                },
                "gpu": 0,
                "terminal": True,
                "can_accept_stream_before_payload": True,
            },
        ],
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _wait_healthy(
    base_url: str,
    proc: subprocess.Popen,
    log_file: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode}\n{_read_tail(log_file)}"
            )
        try:
            payload = _get_json(f"{base_url}/health", timeout=5.0)
            if payload.get("status") == "healthy" or payload.get("running") is True:
                return
            last_error = str(payload)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(
        f"server did not become healthy within {timeout_s}s: {last_error}\n"
        f"{_read_tail(log_file)}"
    )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dots TTS server e2e smoke")

    port = args.port or _find_available_port()
    base_url = f"http://{args.host}:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

    with tempfile.TemporaryDirectory(prefix="dots-tts-server-smoke-") as tmp_dir:
        bootstrap = Path(tmp_dir) / "serve_bootstrap.py"
        config_path = Path(tmp_dir) / "dots_tts_pipeline.json"
        _write_server_bootstrap(bootstrap)
        _write_pipeline_config(config_path, model_path=args.model_path)
        cmd = [
            sys.executable,
            str(bootstrap),
            "serve",
            "--config",
            str(config_path),
            "--host",
            args.host,
            "--port",
            str(port),
            "--model-name",
            "dots-tts-smoke",
            "--log-level",
            "info",
        ]
        log_file = Path(tmp_dir) / "server.log"
        with log_file.open("wb") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_healthy(base_url, proc, log_file, args.timeout_s)
            audio, content_type = _post_json(
                f"{base_url}/v1/audio/speech",
                {
                    "model": "dots-tts-smoke",
                    "input": args.text,
                    "voice": "default",
                    "response_format": "wav",
                    "max_new_tokens": args.max_new_tokens,
                },
                timeout=args.timeout_s,
            )
            if not audio.startswith(b"RIFF") or b"WAVE" not in audio[:16]:
                raise RuntimeError(
                    f"speech response is not WAV: content_type={content_type!r} "
                    f"head={audio[:16]!r}"
                )
            if len(audio) <= 44:
                raise RuntimeError(f"speech response is too small: {len(audio)} bytes")
            print(
                "dots TTS server e2e smoke passed "
                f"port={port} bytes={len(audio)} content_type={content_type!r}"
            )
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15.0)


if __name__ == "__main__":
    raise SystemExit(main())
