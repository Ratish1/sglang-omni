# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Qwen3-Omni with speech output.

Each stage runs in its own process with dedicated GPU placement.
Supports text + audio responses via the OpenAI chat completions API.

Usage::

    python examples/run_qwen3_omni_speech_server.py

    # Custom GPU placement:
    python examples/run_qwen3_omni_speech_server.py \
        --gpu-thinker 0 --gpu-talker 1 --gpu-code2wav 1

    # Thinker TP=2 across two cards, talker disaggregated on one of them:
    python examples/run_qwen3_omni_speech_server.py \
        --thinker-tp-size 2 --gpu-thinker-tp 0,1 \
        --gpu-talker 1 --gpu-code2wav 1

    # Thinker TP=4 across four cards, talker + code2wav on a fifth card:
    python examples/run_qwen3_omni_speech_server.py \
        --thinker-tp-size 4 --gpu-thinker-tp 0,1,2,3 \
        --gpu-talker 4 --gpu-code2wav 4

    # Talker TP=2 across two cards, code2wav stays single-rank:
    python examples/run_qwen3_omni_speech_server.py \
        --gpu-thinker 0 \
        --talker-tp-size 2 --gpu-talker-tp 1,2 \
        --gpu-code2wav 2

    # Then test:
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 64,
            "stream": true,
            "modalities": ["text", "audio"]
        }'
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from typing import Any

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )

    # GPU placement
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--gpu-code-predictor", type=int, default=None)
    parser.add_argument("--gpu-code2wav", type=int, default=0)
    parser.add_argument("--gpu-image-encoder", type=int, default=0)
    parser.add_argument("--gpu-audio-encoder", type=int, default=0)

    # Thinker tensor parallelism (disaggregated path; not used by colocation).
    parser.add_argument(
        "--thinker-tp-size",
        type=int,
        default=1,
        help=(
            "Tensor-parallel size for the thinker stage. Accepts any integer "
            ">= 1; common values are 1, 2, 4, 8. When > 1, also pass "
            "--gpu-thinker-tp with exactly that many GPU ids."
        ),
    )
    parser.add_argument(
        "--gpu-thinker-tp",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for thinker when --thinker-tp-size > 1, "
            "e.g. '0,1'. Length must equal --thinker-tp-size. Overrides "
            "--gpu-thinker when set."
        ),
    )
    parser.add_argument(
        "--talker-tp-size",
        type=int,
        default=1,
        help=(
            "Tensor-parallel size for the talker_ar stage. When > 1, also "
            "pass --gpu-talker-tp with exactly that many GPU ids. code2wav "
            "remains a separate single-rank stage."
        ),
    )
    parser.add_argument(
        "--gpu-talker-tp",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for talker_ar when --talker-tp-size > 1, "
            "e.g. '1,2'. Length must equal --talker-tp-size. Overrides "
            "--gpu-talker when set."
        ),
    )

    # Pipeline
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )
    parser.add_argument(
        "--thinker-max-seq-len",
        type=int,
        default=8192,
        help=(
            "Context length for the thinker stage. The same value is routed "
            "to preprocessing and Talker context guards."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for both Qwen AR stages "
            "(thinker and talker). If omitted, SGLang chooses automatically."
        ),
    )
    parser.add_argument(
        "--thinker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static only for the thinker stage. "
            "Overrides --mem-fraction-static for thinker."
        ),
    )
    parser.add_argument(
        "--talker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static only for the talker stage. "
            "Overrides --mem-fraction-static for talker."
        ),
    )
    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="qwen3-omni")

    return parser.parse_args()


def _validate_fraction(flag_name: str, value: float | None) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")


def _apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue

        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args
        return

    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _set_stage_gpu(config: Any, stage_name: str, gpu_id: int | list[int]) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = gpu_id
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _set_stage_tp_size(config: Any, stage_name: str, tp_size: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.tp_size = tp_size
            stage.parallelism.tp = tp_size
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _parse_tp_gpu_list(flag_name: str, spec: str, tp_size: int) -> list[int]:
    try:
        gpu_ids = [int(piece.strip()) for piece in spec.split(",") if piece.strip()]
    except ValueError as exc:
        raise ValueError(
            f"{flag_name} must be a comma-separated list of integers, got {spec!r}"
        ) from exc
    for gpu in gpu_ids:
        if gpu < 0:
            raise ValueError(f"{flag_name} GPU ids must be >= 0, got {gpu_ids}")
    if len(gpu_ids) != tp_size:
        raise ValueError(
            f"{flag_name} has {len(gpu_ids)} entries but requested tp_size="
            f"{tp_size} requires exactly {tp_size}"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"{flag_name} must list distinct GPU ids, got {gpu_ids}")
    return gpu_ids


def _parse_thinker_tp_gpu_list(spec: str, tp_size: int) -> list[int]:
    return _parse_tp_gpu_list("--gpu-thinker-tp", spec, tp_size)


def _parse_talker_tp_gpu_list(spec: str, tp_size: int) -> list[int]:
    return _parse_tp_gpu_list("--gpu-talker-tp", spec, tp_size)


def _launch_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.serve import launch_server

    for flag_name, value in (
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--thinker-mem-fraction-static", args.thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", args.talker_mem_fraction_static),
    ):
        _validate_fraction(flag_name, value)

    if args.talker_tp_size < 1:
        raise ValueError(f"--talker-tp-size must be >= 1, got {args.talker_tp_size}")
    if args.gpu_code_predictor is not None and (
        args.talker_tp_size > 1 or args.gpu_code_predictor != args.gpu_talker
    ):
        raise ValueError(
            "Qwen3 speech pipeline does not expose a separate code_predictor "
            "stage. For TP=1, use the same GPU for --gpu-code-predictor and "
            "--gpu-talker; for talker TP, omit --gpu-code-predictor."
        )

    config = Qwen3OmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    _set_stage_gpu(config, "image_encoder", args.gpu_image_encoder)
    _set_stage_gpu(config, "audio_encoder", args.gpu_audio_encoder)

    if args.thinker_tp_size < 1:
        raise ValueError(f"--thinker-tp-size must be >= 1, got {args.thinker_tp_size}")

    if args.thinker_tp_size > 1:
        if args.gpu_thinker_tp is None:
            raise ValueError(
                "--thinker-tp-size > 1 requires --gpu-thinker-tp "
                "(comma-separated GPU ids, one per TP rank)."
            )
        thinker_gpu_ids = _parse_tp_gpu_list(
            "--gpu-thinker-tp", args.gpu_thinker_tp, args.thinker_tp_size
        )
        _set_stage_tp_size(config, "thinker", args.thinker_tp_size)
        _set_stage_gpu(config, "thinker", thinker_gpu_ids)
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"disable_custom_all_reduce": True},
        )
    else:
        if args.gpu_thinker_tp is not None:
            raise ValueError(
                "--gpu-thinker-tp only applies when --thinker-tp-size > 1; "
                "for TP=1, use --gpu-thinker."
            )
        _set_stage_gpu(config, "thinker", args.gpu_thinker)

    if args.talker_tp_size > 1:
        if args.gpu_talker_tp is None:
            raise ValueError(
                "--talker-tp-size > 1 requires --gpu-talker-tp "
                "(comma-separated GPU ids, one per TP rank)."
            )
        talker_gpu_ids = _parse_tp_gpu_list(
            "--gpu-talker-tp", args.gpu_talker_tp, args.talker_tp_size
        )
        _set_stage_tp_size(config, "talker_ar", args.talker_tp_size)
        _set_stage_gpu(config, "talker_ar", talker_gpu_ids)
    else:
        if args.gpu_talker_tp is not None:
            raise ValueError(
                "--gpu-talker-tp only applies when --talker-tp-size > 1; "
                "for TP=1, use --gpu-talker."
            )
        _set_stage_gpu(config, "talker_ar", args.gpu_talker)
    _set_stage_gpu(config, "code2wav", args.gpu_code2wav)

    thinker_mem_fraction = (
        args.thinker_mem_fraction_static
        if args.thinker_mem_fraction_static is not None
        else args.mem_fraction_static
    )
    talker_mem_fraction = (
        args.talker_mem_fraction_static
        if args.talker_mem_fraction_static is not None
        else args.mem_fraction_static
    )

    if thinker_mem_fraction is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"mem_fraction_static": thinker_mem_fraction},
        )
    if talker_mem_fraction is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"mem_fraction_static": talker_mem_fraction},
        )

    if args.thinker_max_seq_len is not None:
        thinker_seq_len_updates: dict[str, object] = {
            "thinker_max_seq_len": int(args.thinker_max_seq_len)
        }
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=thinker_seq_len_updates,
        )
        _apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=thinker_seq_len_updates,
        )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    _launch_speech_server(args)


if __name__ == "__main__":
    main()
