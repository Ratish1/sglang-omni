# SPDX-License-Identifier: Apache-2.0
"""Real-model H100 probes for MiniMax Music 3 scheduler boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

from sglang_omni.models.minimax_music3.checkpoint import resolve_checkpoint
from sglang_omni.models.minimax_music3.prompt import build_prompt

_CAPTION = "A minimal acoustic test song at 100 BPM"
_PAIR_ERRORS = (
    "pairs; this batch has",
    "CFG rows are not adjacent pairs",
)
_REPLAY_ERROR = "retract/replay are not supported"
_NATURAL_RETRACTION = "KV cache pool is full. Retract requests."
_INJECTED_RETRACTION = "Testing retraction."


def _render(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    digest = hashlib.sha256()
    body = bytearray()
    response_bytes = 0
    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/audio/speech",
        json=payload,
        timeout=timeout_s,
    ) as response:
        for chunk in response.iter_bytes():
            response_bytes += len(chunk)
            digest.update(chunk)
            if response.status_code != 200 and len(body) < 16_384:
                body.extend(chunk[: 16_384 - len(body)])
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "response_bytes": response_bytes,
            "sha256": digest.hexdigest(),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error": body.decode("utf-8", errors="replace"),
        }


def _log_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _log_tail(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as file:
        file.seek(offset)
        return file.read()


def _admin_headers(api_key: str | None) -> dict[str, str]:
    if api_key is None:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _admin_post(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        headers=_admin_headers(api_key),
        timeout=60,
    )
    return {"status_code": response.status_code, "body": response.json()}


def _running_rows(base_url: str, api_key: str | None) -> int:
    response = _admin_post(
        base_url,
        "/model_info",
        {"stages": ["minimax_music3_ar"]},
        api_key=api_key,
    )
    if response["status_code"] != 200:
        return 0
    for result in response["body"].get("results", []):
        if result.get("stage") == "minimax_music3_ar":
            return int(result["data"]["running_batch_size"])
    return 0


def _admission_prompt(model_path: str, budget: int) -> tuple[str, int]:
    tokenizer_path = resolve_checkpoint(model_path).tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for repetitions in range(1, budget * 4):
        lyrics = "[Verse]\n" + "la " * repetitions
        prompt_tokens = len(tokenizer(build_prompt(_CAPTION, lyrics))["input_ids"])
        if budget // 2 < prompt_tokens < budget:
            return lyrics, prompt_tokens
    raise RuntimeError(
        f"could not construct a prompt between {budget // 2} and {budget} tokens"
    )


def _run_admission(args: argparse.Namespace) -> int:
    server_log = Path(args.server_log)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = _log_offset(server_log)
    lyrics, prompt_tokens = _admission_prompt(args.model_path, args.max_prefill_tokens)
    result = _render(
        args.base_url,
        {
            "model": "MiniMaxAI/MiniMax-Music3",
            "input": lyrics,
            "instructions": _CAPTION,
            "seed": 42,
            "max_new_tokens": 1,
            "response_format": "wav",
        },
        timeout_s=args.timeout_s,
    )
    time.sleep(1)
    log_tail = _log_tail(server_log, offset)
    pair_error = next((value for value in _PAIR_ERRORS if value in log_tail), None)
    if result["status_code"] == 200:
        outcome = "handled"
    elif pair_error is not None:
        outcome = "reproduced"
    else:
        outcome = "inconclusive"
    report = {
        "probe": "pair_admission",
        "outcome": outcome,
        "max_prefill_tokens": args.max_prefill_tokens,
        "physical_row_prompt_tokens": prompt_tokens,
        "cfg_pair_prompt_tokens": 2 * prompt_tokens,
        "pair_error": pair_error,
        "request": result,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if outcome != "inconclusive" else 2


def _run_pressure(args: argparse.Namespace) -> int:
    server_log = Path(args.server_log)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = _log_offset(server_log)
    payload = {
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Intro]\n(instrumental)",
        "instructions": (
            "An instrumental ambient piece, no vocals: warm analog pads, slow "
            "evolving texture, distant piano, 70 BPM"
        ),
        "seed": 3,
        "max_new_tokens": args.max_new_tokens,
        "response_format": "wav",
    }
    with ThreadPoolExecutor(max_workers=args.requests) as pool:
        results = list(
            pool.map(
                lambda _: _render(
                    args.base_url,
                    payload,
                    timeout_s=args.timeout_s,
                ),
                range(args.requests),
            )
        )
    time.sleep(1)
    log_tail = _log_tail(server_log, offset)
    natural_retractions = log_tail.count(_NATURAL_RETRACTION)
    injected_retractions = log_tail.count(_INJECTED_RETRACTION)
    failure_marker = next(
        (value for value in (*_PAIR_ERRORS, _REPLAY_ERROR) if value in log_tail),
        None,
    )
    failed = sum(result["status_code"] != 200 for result in results)
    if natural_retractions and failure_marker is not None:
        outcome = "reproduced"
    elif natural_retractions and failed == 0:
        outcome = "handled"
    elif natural_retractions:
        outcome = "inconclusive"
    elif injected_retractions and failure_marker is not None:
        outcome = "fault_injection_reproduced"
    else:
        outcome = "not_triggered"
    report = {
        "probe": "kv_pressure",
        "outcome": outcome,
        "requests": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "natural_retraction_log_count": natural_retractions,
        "injected_retraction_log_count": injected_retractions,
        "failure_marker": failure_marker,
        "successful_requests": len(results) - failed,
        "failed_requests": failed,
        "requests_detail": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if outcome not in {"inconclusive", "not_triggered"} else 2


def _run_pause_retract(args: argparse.Namespace) -> int:
    server_log = Path(args.server_log)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = _log_offset(server_log)
    payload = {
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Intro]\n(instrumental)",
        "instructions": (
            "An instrumental ambient piece, no vocals: warm analog pads, slow "
            "evolving texture, distant piano, 70 BPM"
        ),
        "seed": 3,
        "max_new_tokens": args.max_new_tokens,
        "response_format": "wav",
    }
    running_rows = 0
    pause = None
    continued = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(
            _render,
            args.base_url,
            payload,
            timeout_s=args.timeout_s,
        )
        deadline = time.monotonic() + args.wait_running_s
        while time.monotonic() < deadline and not request.done():
            running_rows = _running_rows(args.base_url, args.admin_api_key)
            if running_rows >= 2:
                break
            time.sleep(0.25)
        if running_rows >= 2:
            pause = _admin_post(
                args.base_url,
                "/pause_generation",
                {"mode": "retract", "stages": ["minimax_music3_ar"]},
                api_key=args.admin_api_key,
            )
            continued = _admin_post(
                args.base_url,
                "/continue_generation",
                {"torch_empty_cache": True, "stages": ["minimax_music3_ar"]},
                api_key=args.admin_api_key,
            )
        request_result = request.result()
    time.sleep(1)
    log_tail = _log_tail(server_log, offset)
    failure_marker = next(
        (value for value in (*_PAIR_ERRORS, _REPLAY_ERROR) if value in log_tail),
        None,
    )
    if running_rows < 2:
        outcome = "not_triggered"
    elif pause is None or pause["status_code"] != 200:
        outcome = "inconclusive"
    elif continued is None or continued["status_code"] != 200:
        outcome = "inconclusive"
    elif request_result["status_code"] != 200 and failure_marker is not None:
        outcome = "reproduced"
    elif request_result["status_code"] == 200:
        outcome = "handled"
    else:
        outcome = "inconclusive"
    report = {
        "probe": "pause_retract",
        "outcome": outcome,
        "running_rows_before_pause": running_rows,
        "failure_marker": failure_marker,
        "pause": pause,
        "continue": continued,
        "request": request_result,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if outcome not in {"inconclusive", "not_triggered"} else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="probe", required=True)
    admission = subparsers.add_parser(
        "pair-admission", help="exercise one-row-versus-one-CFG-pair prefill budget"
    )
    admission.add_argument("--base-url", default="http://localhost:8000")
    admission.add_argument("--model-path", required=True)
    admission.add_argument("--server-log", required=True)
    admission.add_argument("--output", required=True)
    admission.add_argument("--max-prefill-tokens", type=int, default=256)
    admission.add_argument("--timeout-s", type=float, default=900)
    admission.set_defaults(run=_run_admission)

    pressure = subparsers.add_parser(
        "kv-pressure", help="drive real or fault-injected decode retraction"
    )
    pressure.add_argument("--base-url", default="http://localhost:8000")
    pressure.add_argument("--server-log", required=True)
    pressure.add_argument("--output", required=True)
    pressure.add_argument("--requests", type=int, default=16)
    pressure.add_argument("--max-new-tokens", type=int, default=250)
    pressure.add_argument("--timeout-s", type=float, default=1800)
    pressure.set_defaults(run=_run_pressure)

    pause = subparsers.add_parser(
        "pause-retract", help="retract and resume a live request through the admin API"
    )
    pause.add_argument("--base-url", default="http://localhost:8000")
    pause.add_argument("--server-log", required=True)
    pause.add_argument("--output", required=True)
    pause.add_argument("--admin-api-key")
    pause.add_argument("--max-new-tokens", type=int, default=9000)
    pause.add_argument("--wait-running-s", type=float, default=60)
    pause.add_argument("--timeout-s", type=float, default=1800)
    pause.set_defaults(run=_run_pause_retract)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
