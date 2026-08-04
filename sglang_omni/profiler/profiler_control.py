# SPDX-License-Identifier: Apache-2.0
"""Acknowledged coordinator-to-stage profiler control."""

from __future__ import annotations

import asyncio
import logging
import math
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from sglang_omni.pipeline.control_plane import PullSocket, PushSocket
from sglang_omni.proto import (
    ProfilerResultMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
)

logger = logging.getLogger(__name__)

_DEFAULT_OPERATION_TIMEOUT_S = 120.0


@dataclass
class ProfilerControlClient:
    """Broadcast profiler operations and wait for one aggregate per stage."""

    stage_endpoints: dict[str, str]

    _socks: dict[str, PushSocket] | None = None
    _result_socket: PullSocket | None = None
    _reply_endpoint: str | None = None
    _reply_dir: tempfile.TemporaryDirectory[str] | None = None
    _operation_lock: asyncio.Lock | None = None

    async def start(self) -> None:
        if self._socks is not None:
            return
        self._operation_lock = asyncio.Lock()
        self._reply_dir = tempfile.TemporaryDirectory(prefix="sglang-omni-profiler-")
        self._reply_endpoint = f"ipc://{self._reply_dir.name}/results.sock"
        self._result_socket = PullSocket(self._reply_endpoint, bind=True)
        await self._result_socket.start()

        self._socks = {}
        for stage_name, endpoint in self.stage_endpoints.items():
            sock = PushSocket(endpoint)
            await sock.connect()
            self._socks[stage_name] = sock
        logger.info(
            "ProfilerControlClient connected to %d stages; replies=%s",
            len(self._socks),
            self._reply_endpoint,
        )

    async def close(self) -> None:
        if self._socks:
            for sock in self._socks.values():
                sock.close()
        self._socks = None
        if self._result_socket is not None:
            self._result_socket.close()
        self._result_socket = None
        self._reply_endpoint = None
        if self._reply_dir is not None:
            self._reply_dir.cleanup()
        self._reply_dir = None
        self._operation_lock = None

    async def broadcast_start(
        self,
        run_id: str,
        trace_path_template: str,
        config: dict[str, Any] | None = None,
        stages: list[str] | None = None,
        event_dir: str | None = None,
        enable_torch: bool = True,
        enable_nvtx: bool = False,
        torch_owner: str = "scheduler",
        timeout_s: float = _DEFAULT_OPERATION_TIMEOUT_S,
    ) -> dict[str, Any]:
        await self.start()
        _validate_timeout(timeout_s)
        targets = self._resolve_targets(stages)
        op_id = uuid.uuid4().hex
        assert self._reply_endpoint is not None
        msg = ProfilerStartMessage(
            op_id=op_id,
            run_id=run_id,
            trace_path_template=trace_path_template,
            reply_endpoint=self._reply_endpoint,
            event_dir=event_dir,
            enable_torch=enable_torch,
            enable_nvtx=enable_nvtx,
            torch_owner=torch_owner,
            torch_config=config,
            timeout_s=timeout_s,
        )
        return await self._broadcast_and_collect(
            msg,
            action="start",
            run_id=run_id,
            targets=targets,
            timeout_s=timeout_s,
        )

    async def broadcast_stop(
        self,
        run_id: str | None = None,
        stages: list[str] | None = None,
        *,
        timeout_s: float = _DEFAULT_OPERATION_TIMEOUT_S,
    ) -> dict[str, Any]:
        await self.start()
        _validate_timeout(timeout_s)
        targets = self._resolve_targets(stages)
        op_id = uuid.uuid4().hex
        assert self._reply_endpoint is not None
        msg = ProfilerStopMessage(
            op_id=op_id,
            run_id=run_id,
            reply_endpoint=self._reply_endpoint,
            timeout_s=timeout_s,
        )
        return await self._broadcast_and_collect(
            msg,
            action="stop",
            run_id=run_id,
            targets=targets,
            timeout_s=timeout_s,
        )

    def _resolve_targets(self, stages: list[str] | None) -> list[str]:
        assert self._socks is not None
        targets = list(dict.fromkeys(stages or self._socks.keys()))
        unknown = [stage for stage in targets if stage not in self._socks]
        if unknown:
            raise ValueError(f"unknown profiler stages: {unknown}")
        if not targets:
            raise ValueError("at least one profiler stage is required")
        return targets

    async def _broadcast_and_collect(
        self,
        msg: ProfilerStartMessage | ProfilerStopMessage,
        *,
        action: str,
        run_id: str | None,
        targets: list[str],
        timeout_s: float,
    ) -> dict[str, Any]:
        assert self._socks is not None
        assert self._operation_lock is not None
        async with self._operation_lock:
            for stage in targets:
                await self._socks[stage].send(msg)
            logger.info(
                "Broadcast profiler_%s op_id=%s run_id=%s stages=%s",
                action,
                msg.op_id,
                run_id,
                targets,
            )
            results, missing = await self._collect_results(
                op_id=msg.op_id,
                targets=targets,
                timeout_s=timeout_s,
            )
        success = not missing and all(result.success for result in results)
        return {
            "op_id": msg.op_id,
            "run_id": run_id,
            "action": action,
            "success": success,
            "stages": [result.to_dict() for result in results],
            "missing_stages": sorted(missing),
        }

    async def _collect_results(
        self,
        *,
        op_id: str,
        targets: list[str],
        timeout_s: float,
    ) -> tuple[list[ProfilerResultMessage], set[str]]:
        assert self._result_socket is not None
        pending = set(targets)
        results: dict[str, ProfilerResultMessage] = {}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(float(timeout_s), 0.1)
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(
                    self._result_socket.recv(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            if not isinstance(raw, ProfilerResultMessage):
                logger.warning(
                    "Ignoring non-profiler result on profiler reply socket: %s",
                    type(raw).__name__,
                )
                continue
            if raw.op_id != op_id:
                logger.warning(
                    "Ignoring stale profiler result op_id=%s while waiting for %s",
                    raw.op_id,
                    op_id,
                )
                continue
            if raw.stage not in pending:
                logger.warning(
                    "Ignoring duplicate/unexpected profiler result stage=%s op_id=%s",
                    raw.stage,
                    op_id,
                )
                continue
            results[raw.stage] = raw
            pending.remove(raw.stage)
        return [results[stage] for stage in targets if stage in results], pending


def _validate_timeout(timeout_s: float) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("profiler timeout_s must be finite and positive")
