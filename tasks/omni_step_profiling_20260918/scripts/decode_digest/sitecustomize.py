"""Per-decode digests of the Qwen3-TTS vocoder, for comparing two servers decode by decode.

Put this directory on a server's PYTHONPATH and set OMNI_DECODE_DIGEST=<dir>. Each process appends one JSON line per
vocoder decode to <dir>/<pid>.jsonl: the graph runner's mode (a replay) or the eager entry, the shape of the codes, the
row valid widths of a tail replay, and SHA-1 digests of the input codes and of the returned waveform rows, in call
order. Calls made while a CUDA graph is captured are skipped. Every record copies to the host and so syncs the device;
such a run's timings are not used. With OMNI_DECODE_CENSUS=1 a record holds only the kind, mode, thread and shape,
read from host metadata, so the run's batching is the served one.
"""

import hashlib
import json
import os
import threading

DIGEST_DIR = os.environ.get("OMNI_DECODE_DIGEST")
CENSUS = os.environ.get("OMNI_DECODE_CENSUS") == "1"

if DIGEST_DIR:
    import torch

    from sglang_omni.models.qwen3_tts import incremental_codec, incremental_codec_cuda_graph

    LOCK = threading.Lock()
    COUNTER = [0]
    PATH = os.path.join(DIGEST_DIR, f"{os.getpid()}.jsonl")

    def digest(tensor: torch.Tensor) -> str:
        tensor = tensor.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.int16)
        else:
            pass
        return hashlib.sha1(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]

    def record(kind: str, codes: torch.Tensor, waveform: torch.Tensor, extra: dict) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        else:
            pass
        entry = {
            "kind": kind,
            "thread": threading.current_thread().name,
            "codes_shape": list(codes.shape),
            **extra,
        }
        if not CENSUS:
            entry["codes"] = digest(codes)
            entry["out"] = digest(waveform)
        else:
            pass
        with LOCK:
            COUNTER[0] += 1
            entry["n"] = COUNTER[0]
            with open(PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    Runner = incremental_codec_cuda_graph.Qwen3TTSIncrementalCodecCudaGraphRunner
    original_decode_slots = Runner.decode_slots

    def decode_slots(self, codes, slots, valid_frames=None):
        waveform = original_decode_slots(self, codes, slots, valid_frames)
        if waveform is not None:
            record(
                "replay",
                codes,
                waveform,
                {
                    "mode": self.mode,
                    "valid": None if valid_frames is None or CENSUS else [int(v) for v in valid_frames],
                },
            )
        else:
            pass
        return waveform

    Runner.decode_slots = decode_slots

    Decoder = incremental_codec.Qwen3TTSIncrementalDecoder
    original_decode = Decoder.decode

    def decode(self, codes, state, *, compiled=False):
        waveform = original_decode(self, codes, state, compiled=compiled)
        record("eager", codes, waveform, {"compiled": bool(compiled)})
        return waveform

    Decoder.decode = decode
else:
    pass
