"""Exercise warmup with real HTTP, coordinator admission and CPU stage processes."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import ClassVar, Literal

import httpx
import psutil

from sglang_omni.config.schema import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.serve.launcher import run_server
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage

PROBE_MODULE = "tools.review_pr2686_lifecycle"


class ProbePipelineConfig(PipelineConfig):
    server_warmup_request_factory: ClassVar[str] = f"{PROBE_MODULE}.warmup_request"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "compute": EngineStageConfig
    }

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, int]:
        return {"max_running_requests": 1, "max_queued_requests": 1}


class NoWarmupPipelineConfig(ProbePipelineConfig):
    server_warmup_request_factory: ClassVar[None] = None


def warmup_request(pipeline_config: PipelineConfig) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=pipeline_config.name)],
        max_tokens=8,
    )


def create_probe_scheduler(
    *,
    directory: str,
    label: str,
    case: Literal[
        "success", "failure", "capacity", "timeout", "signal", "worker-death"
    ],
) -> SimpleScheduler[StagePayload, StagePayload]:
    root = Path(directory)

    async def compute(payload: StagePayload) -> StagePayload:
        (root / f"{label}-{os.getpid()}.json").write_text(
            json.dumps({"pid": os.getpid(), "request_id": payload.request_id})
        )
        if label == "compute":
            if case == "failure":
                raise ValueError("probe compute failure")
            elif case == "worker-death":
                os._exit(9)
            elif case in {"timeout", "signal"}:
                while not (root / "release").exists():
                    await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.25)
            payload.data = {"text": "completed", "finish_reason": "stop"}
        else:
            pass
        return payload

    return SimpleScheduler(compute)


async def run_probe(args: argparse.Namespace) -> None:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config_cls = (
        NoWarmupPipelineConfig if args.no_warmup_factory else ProbePipelineConfig
    )
    config = config_cls(
        model_path="review-no-weights",
        name=f"probe-{args.case}",
        stages=[
            StageConfig(
                name="input",
                process="input",
                factory_path=f"{PROBE_MODULE}.create_probe_scheduler",
                factory=FactoryArgs(directory=str(root), label="input", case=args.case),
                next="compute",
            ),
            EngineStageConfig(
                name="compute",
                process="compute",
                factory_path=f"{PROBE_MODULE}.create_probe_scheduler",
                factory=FactoryArgs(
                    directory=str(root), label="compute", case=args.case
                ),
                terminal=True,
            ),
        ],
        processes={
            "input": ProcessConfig(num_replicas=3 if args.case == "capacity" else 1)
        },
    )
    task = asyncio.create_task(
        run_server(
            config,
            host="127.0.0.1",
            port=args.port,
            skip_server_warmup=args.skip_server_warmup,
        )
    )
    observations: list[dict[str, str | int | float]] = []
    start = time.monotonic()
    blocked_after_timeout = False
    worker_requests_before_ready: int | None = None
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            while not task.done():
                if time.monotonic() - start > 90:
                    raise TimeoutError("probe server startup exceeded 90 seconds")
                else:
                    pass
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{args.port}/health", timeout=1
                    )
                    observations.append(
                        {
                            "elapsed_s": time.monotonic() - start,
                            "status_code": response.status_code,
                            "status": response.json()["status"],
                        }
                    )
                    if response.status_code == 200:
                        worker_requests_before_ready = len(list(root.glob("*-*.json")))
                        ordinary = await client.post(
                            f"http://127.0.0.1:{args.port}/v1/chat/completions",
                            json=warmup_request(config).model_dump(exclude_none=True),
                            timeout=5,
                        )
                        observations.append(
                            {"ordinary_status_code": ordinary.status_code}
                        )
                        break
                    else:
                        pass
                except httpx.HTTPError:
                    pass
                if args.case == "timeout" and list(root.glob("compute-*.json")):
                    await asyncio.sleep(2)
                    blocked_after_timeout = not task.done()
                    (root / "release").touch()
                    await asyncio.wait({task}, timeout=5)
                elif args.case == "signal" and list(root.glob("compute-*.json")):
                    os.kill(os.getpid(), 15)
                    await asyncio.sleep(0.5)
                    blocked_after_timeout = not task.done()
                    (root / "release").touch()
                    await asyncio.wait({task}, timeout=5)
                else:
                    pass
                await asyncio.sleep(0.01)
        finally:
            if not task.done():
                task.cancel()
            else:
                pass
            outcome = await asyncio.gather(task, return_exceptions=True)
    workers = [json.loads(path.read_text()) for path in root.glob("*-*.json")]
    receipt = {
        "case": args.case,
        "observations": observations,
        "outcome": [type(value).__name__ + ": " + str(value) for value in outcome],
        "blocked_after_timeout_or_signal": blocked_after_timeout,
        "worker_requests": workers,
        "worker_requests_before_ready": worker_requests_before_ready,
        "worker_pids_alive_after_stop": [
            worker["pid"] for worker in workers if psutil.pid_exists(worker["pid"])
        ],
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["success", "failure", "capacity", "timeout", "signal", "worker-death"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--skip-server-warmup", action="store_true")
    parser.add_argument("--no-warmup-factory", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_probe(args))


if __name__ == "__main__":
    main()
