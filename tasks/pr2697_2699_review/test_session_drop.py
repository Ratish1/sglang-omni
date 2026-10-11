import asyncio

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, SubmitMessage
from sglang_omni.proto.session import TimedChunk, find_session_operation
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class SessionWorkerControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self, coordinator: Coordinator) -> None:
        super().__init__()
        self.coordinator = coordinator
        self.append_entered = asyncio.Event()
        self.append_request_id = ""
        self.closed_owners: list[str] = []

    async def submit_to_stage(
        self, stage: str, endpoint: str, message: SubmitMessage
    ) -> None:
        await super().submit_to_stage(stage, endpoint, message)
        operation = find_session_operation(message.data.request.metadata)
        assert operation is not None
        if operation.operation == "append":
            self.append_request_id = message.request_id
            self.append_entered.set()
        else:
            if operation.operation == "close":
                self.closed_owners.append(stage)
            else:
                pass
            await self.coordinator.handle_completion(
                CompleteMessage(message.request_id, stage, True, result={})
            )


def test_admin_dropped_append_ends_session_output_and_closes_owners() -> None:
    async def run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="decode",
            terminal_stages=["decode"],
        )
        control_plane = SessionWorkerControlPlane(coordinator)
        coordinator.control_plane = control_plane
        coordinator.running = True
        coordinator.register_stage("decode", "inproc://decode")
        identity = await coordinator.open_session(
            OmniRequest(inputs=None), stages=["decode"], session_id="admin-session"
        )
        session = coordinator.sessions[identity.id]
        outputs = coordinator.session_outputs(identity)
        waiting = asyncio.create_task(anext(outputs))
        try:
            await coordinator.append_session(
                identity, TimedChunk("audio", 0, 20, 0, b"pcm")
            )
            await control_plane.append_entered.wait()
            await coordinator.handle_completion(
                CompleteMessage(
                    control_plane.append_request_id,
                    "decode",
                    False,
                    error="The request was dropped by an admin action.",
                )
            )
            for _ in range(30):
                await asyncio.sleep(0)
            print(
                {
                    "pump_done": session.pump.done(),
                    "pump_cancelled": session.pump.cancelled(),
                    "session_closing": session.is_closing,
                    "session_closed": session.is_closed,
                    "output_done": waiting.done(),
                    "closed_owners": control_plane.closed_owners,
                    "session_reserved": identity.id in coordinator.sessions,
                }
            )
            assert waiting.done()
            assert control_plane.closed_owners == ["decode"]
            assert identity.id not in coordinator.sessions
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)

    asyncio.run(run())
