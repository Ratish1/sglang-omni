# SPDX-License-Identifier: Apache-2.0
"""Order one stage's device work after another stage's step on a shared card.

The talker process records a CUDA interprocess event at the end of each step's device work
and after each code2wav window; the thinker process waits on both events before every
forward, so a thinker forward starts only when the talker is idle or in its host phase.
Handles travel through a file under SGLANG_OMNI_CARD_TURN_DIR; the feature is off when the
variable is unset.
"""

from __future__ import annotations

import logging
import os
import pickle

import torch

from sglang_omni.utils.ipc_weights import atomic_write

logger = logging.getLogger(__name__)

CARD_TURN_DIR_ENV = "SGLANG_OMNI_CARD_TURN_DIR"
HANDLE_FILE_NAME = "card_turn.events"


def card_turn_dir() -> str | None:
    return os.environ.get(CARD_TURN_DIR_ENV) or None


class CardTurnPublisher:
    """The talker side: two interprocess events, exported once, recorded per step."""

    def __init__(self, dir_path: str, device: torch.device) -> None:
        self.device = device
        with torch.cuda.device(device):
            self.step_event = torch.cuda.Event(interprocess=True)
            self.window_event = torch.cuda.Event(interprocess=True)
            handles = {
                "device_index": torch.cuda.current_device(),
                "step": self.step_event.ipc_handle(),
                "window": self.window_event.ipc_handle(),
            }
        atomic_write(os.path.join(dir_path, HANDLE_FILE_NAME), pickle.dumps(handles))
        logger.info(f"card turn events published to {dir_path} on {device}")

    def record_step(self, stream: torch.cuda.Stream) -> None:
        self.step_event.record(stream)

    def record_window(self, stream: torch.cuda.Stream) -> None:
        self.window_event.record(stream)


class CardTurnWaiter:
    """The thinker side: attaches to the handle file once it exists, then waits per forward."""

    def __init__(self, dir_path: str, device: torch.device) -> None:
        self.handle_path = os.path.join(dir_path, HANDLE_FILE_NAME)
        self.device = device
        self.step_event: torch.cuda.Event | None = None
        self.window_event: torch.cuda.Event | None = None

    def attach(self) -> bool:
        if self.step_event is not None:
            return True
        elif not os.path.exists(self.handle_path):
            return False
        else:
            with open(self.handle_path, "rb") as handle_file:
                handles = pickle.load(handle_file)
            self.step_event = torch.cuda.Event.from_ipc_handle(
                self.device, handles["step"]
            )
            self.window_event = torch.cuda.Event.from_ipc_handle(
                self.device, handles["window"]
            )
            logger.info(
                f"card turn events attached from {self.handle_path} on {self.device}"
            )
            return True

    def wait(self, stream: torch.cuda.Stream) -> None:
        if not self.attach():
            return
        else:
            pass
        assert self.step_event is not None and self.window_event is not None
        stream.wait_event(self.step_event)
        stream.wait_event(self.window_event)


PROCESS_PUBLISHER: CardTurnPublisher | None = None


def process_publisher(device: torch.device) -> CardTurnPublisher | None:
    """One publisher per talker process, shared by the talker runner and code2wav."""
    global PROCESS_PUBLISHER
    dir_path = card_turn_dir()
    if dir_path is None:
        return None
    elif PROCESS_PUBLISHER is None:
        PROCESS_PUBLISHER = CardTurnPublisher(dir_path, device)
    else:
        pass
    return PROCESS_PUBLISHER
