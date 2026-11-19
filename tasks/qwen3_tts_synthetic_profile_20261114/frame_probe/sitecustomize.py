"""Measurement only: one request event each time the Qwen3-TTS vocoder ingests a stream's codec frames.

With QWEN3_TTS_FRAME_PROBE=1, Qwen3TTSStreamingVocoderScheduler.ingest records qwen3_tts_frames_ingested with the
stream's generated frame count after the chunk, while the request event recorder is active.
"""

import importlib.abc
import importlib.util
import os
import sys

VOCODER = "sglang_omni.models.qwen3_tts.streaming_vocoder"


def patch_vocoder(module) -> None:
    from sglang_omni.profiler.event_recorder import emit, get_recorder

    cls = module.Qwen3TTSStreamingVocoderScheduler
    original = cls.ingest

    def ingest(self, request_id, state, codes):
        original(self, request_id, state, codes)
        if get_recorder().is_active():
            emit(
                request_id=request_id,
                stage=None,
                event_name="qwen3_tts_frames_ingested",
                metadata={"generated_frames": state.total_frames - state.ref_frames},
            )
        else:
            pass

    cls.ingest = ingest


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name != VOCODER or name in self.seen:
            return None
        else:
            pass
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch_vocoder(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("QWEN3_TTS_FRAME_PROBE") == "1":
    sys.meta_path.insert(0, PatchOnImport())
else:
    pass
