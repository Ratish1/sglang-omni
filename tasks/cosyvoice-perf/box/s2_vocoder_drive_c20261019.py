"""S2 of DESIGN_VOCODER_STANDARD_20261019.md, the vocoder's checks on the checkpoint, one tree at a time (the tree on
PYTHONPATH): the serving factory's vocoder driven as the serving loop drives it, single threaded, from a fixed set of
streams whose chunks all wait in the inbox in a fixed order (first chunks of 28 tokens, then 25 a chunk, stream_done,
the payload), so main and S2 form the same steps.

  --check-waits: every step runs under torch.cuda.set_sync_debug_mode("warn"), and each warning prints the stack that
  raised it, once per call site; the summary counts the sites (S2's contract: none inside a launch).
  --out FILE: each request's audio, concatenated in arrival order, saved as a .npz for --compare.
  --compare A B: whether two .npz files hold the same audio for every request, bit for bit.

The prefix pool is sized small (--prefix-gb) so some rows run the plain whole history hop as well.

usage (in the container, one card): CUDA_VISIBLE_DEVICES=<card> python3 s2_vocoder_drive_c20261019.py
  [--check-waits] [--out FILE] [--streams N] [--prefix-gb G] | --compare A B
"""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import traceback
import warnings

import numpy as np
import torch

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PROMPT_TOKENS = 100
FIRST_CHUNK_TOKENS = 28
FOLLOWUP_CHUNK_TOKENS = 25


def compare(first: str, second: str) -> None:
    a, b = np.load(first), np.load(second)
    names = sorted(set(a.files) | set(b.files))
    same = [name for name in names if name in a.files and name in b.files and np.array_equal(a[name], b[name])]
    print(f"{len(same)} of {len(names)} requests bit identical")
    for name in names:
        if name not in same:
            left = a[name] if name in a.files else np.zeros(0)
            right = b[name] if name in b.files else np.zeros(0)
            print(f"  {name}: {left.shape} against {right.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-waits", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--streams", type=int, default=12)
    parser.add_argument("--prefix-gb", type=float, default=1.0)
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
    from sglang_omni.models.fun_cosyvoice3.stages import create_vocoder_executor
    from sglang_omni.pipeline.stage.stream_queue import StreamItem
    from sglang_omni.proto import OmniRequest, StagePayload
    from sglang_omni.scheduling.message import IncomingMessage

    import sglang_omni

    print(f"sglang_omni from {sglang_omni.__file__}", flush=True)
    scheduler = create_vocoder_executor(
        MODEL,
        device="cuda",
        gpu_id=0,
        max_batch_size=16,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        enable_hift_cuda_graph=True,
        flow_prefix_cache_gb=args.prefix_gb,
    )
    # one clock reading for every call, so main and S2 rank every selection alike
    scheduler.clock = lambda: 1000.0
    flow = scheduler.vocoder.flow
    generator = torch.Generator().manual_seed(2910)
    lengths = [40 + (index * 37) % 260 for index in range(args.streams)]
    for index, length in enumerate(lengths):
        request_id = f"req-{index:02d}"
        tokens = torch.randint(0, 6561, (length,), dtype=torch.long, generator=generator)
        state = FunCosyVoice3State(
            text="check",
            stream=True,
            audio_codes=tokens,
            flow_prompt_speech_token=torch.randint(
                0, 6561, (1, PROMPT_TOKENS), dtype=torch.int32, generator=generator
            ),
            flow_prompt_speech_feat=torch.randn(
                1, PROMPT_TOKENS * flow.token_mel_ratio, flow.output_size, generator=generator
            ),
            flow_embedding=torch.randn(1, 192, generator=generator),
            finish_reason="stop",
        )
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="check", params={"stream": True}),
            data=state.to_dict(),
        )
        scheduler.inbox.put(IncomingMessage(request_id=request_id, type="new_request", data=payload))
        start, chunk = 0, FIRST_CHUNK_TOKENS
        while start < length:
            codes = tokens[start : start + chunk]
            scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_chunk",
                    data=StreamItem(
                        chunk_id=start,
                        data=codes,
                        from_stage="tts_engine",
                        metadata={"modality": "audio_codes", "stream": True},
                    ),
                )
            )
            start += chunk
            chunk = FOLLOWUP_CHUNK_TOKENS
        scheduler.inbox.put(IncomingMessage(request_id=request_id, type="stream_done"))

    sites: collections.Counter = collections.Counter()

    def show(message, category, filename, lineno, file=None, line=None) -> None:
        stack = "".join(
            frame for frame in traceback.format_stack()[:-1] if "sglang_omni" in frame or "cosyvoice" in frame
        )
        if stack not in sites:
            print(f"--- synchronizing call: {message}\n{stack}", flush=True)
        sites[stack] += 1

    if args.check_waits:
        warnings.showwarning = show
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
    steps = 0
    while True:
        try:
            msg = scheduler.get_batch_message()
        except queue.Empty:
            if not scheduler.has_ready_work():
                break
            scheduler.run_ready_step()
            steps += 1
            continue
        scheduler.handle_message(msg, None)
    torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    audio: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    results = errors = 0
    while True:
        try:
            out = scheduler.outbox.get_nowait()
        except queue.Empty:
            break
        if out.type == "stream":
            data = out.data
            audio[out.request_id].append(
                np.frombuffer(data["audio_waveform"], dtype=np.float32).reshape(data["audio_waveform_shape"])
            )
        elif out.type == "result":
            results += 1
        else:
            errors += 1
            print(f"{out.request_id} {out.type}: {out.data}")
    print(
        f"steps {steps}, streams {len(lengths)}, results {results}, errors {errors}, "
        f"synchronizing call sites {len(sites)} ({sum(sites.values())} calls)"
    )
    if args.out:
        np.savez(args.out, **{name: np.concatenate(parts) for name, parts in audio.items()})
        print(f"saved {len(audio)} requests to {args.out}")
    sys.exit(0 if errors == 0 and results == len(lengths) else 1)


if __name__ == "__main__":
    main()
