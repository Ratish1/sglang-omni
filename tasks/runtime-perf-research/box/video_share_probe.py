"""Whether a video request answers when the preprocessing stage has fewer CPUs than videos running (one CPU, two videos).

The first video holds its thread until the second has answered or 10 s pass. Prints the second video's outcome and
the team its preprocessing ran on: a result, an error, or nothing at all (the request would wait forever).

usage: PYTHONPATH=<tree> python3 video_share_probe.py
"""

import inspect
import queue
import threading

import torch

import sglang_omni.models.qwen3_omni.stages as qwen_stages
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

ANSWER_TIMEOUT_S = 10.0


def video_message(request_id: str) -> IncomingMessage:
    inputs = {"messages": [{"role": "user", "content": "describe"}], "videos": ["clip.mp4"]}
    return IncomingMessage(
        request_id=request_id,
        type="new_request",
        data=StagePayload(request_id=request_id, request=OmniRequest(inputs=inputs, params={}), data={}),
    )


def main() -> None:
    second_done = threading.Event()
    teams: dict[str, int] = {}

    class Preprocessor:
        async def process(self, payload, request_inputs, video_share):
            teams[payload.request_id] = torch.get_num_threads()
            if payload.request_id == "first":
                second_done.wait(timeout=3 * ANSWER_TIMEOUT_S)
            else:
                pass
            return payload.request_id

    parameters = inspect.signature(qwen_stages.PreprocessingScheduler.__init__).parameters
    sizing = (
        {"video_cpus": 1}
        if "video_cpus" in parameters
        else {"stage_threads": 8, "stage_cpus": 1}
    )
    scheduler = qwen_stages.PreprocessingScheduler(
        Preprocessor(), video_threads=2, video_decode_threads=1, **sizing
    )
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()
    scheduler.inbox.put(video_message("first"))
    scheduler.inbox.put(video_message("second"))
    try:
        answer = scheduler.outbox.get(timeout=ANSWER_TIMEOUT_S)
        outcome = f"{answer.type} for {answer.request_id}"
    except queue.Empty:
        outcome = f"no answer in {ANSWER_TIMEOUT_S:.0f} s"
    second_done.set()
    print(f"sizing {sizing}: second video {outcome}, teams {teams}", flush=True)
    scheduler.stop()
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
