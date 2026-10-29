"""Which threads' CUDA calls an nsys export holds (STATE 10-29 item 8, the reference batcher's thread gap).

In block 8's serve exports the reference batcher's thread (`qwen3-tts-ref-code`) carries NVTX ranges but no CUDA
runtime row, on either card, while the preprocessing pool's threads are recorded. In one process under nsys:
  pre       a thread started before CUDA is initialized
  main      the main thread
  post      a thread started after CUDA is initialized
  batcher   the real Qwen3TTSRefCodeBatcher (graphs captured, its thread started after the capture) encoding clips
Each of the first three runs, inside an NVTX range named for it, eager matmuls, a side-stream graph replay and an event
synchronize on the card; the batcher's encode and handoff are wrapped in ranges of their own. The reader prints, per
thread holding a probe range, its name, its ranges, and its CUDA runtime rows inside them by call name.

usage: nsys profile -o OUT ... python3 nsys_thread_probe.py run WAV_DIR --model M --card 1
       python3 nsys_thread_probe.py read OUT.sqlite
"""

import argparse
import sqlite3
import threading
from collections import Counter, defaultdict
from pathlib import Path

ITERS = 20
CLIPS = 40
PUSH_POP = 59


def work(tag: str, device: str, go: threading.Event) -> None:
    import torch

    go.wait()
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device)
    static = torch.zeros(1024, device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(2):
            static.add_(1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static.add_(1)
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_push(f"probe.{tag}")
    matrix = torch.randn(256, 256, device=device)
    for _ in range(ITERS):
        matrix = matrix @ matrix
        matrix = matrix / matrix.norm()
        with torch.cuda.stream(stream):
            graph.replay()
        event = torch.cuda.Event()
        event.record(stream)
        event.synchronize()
    torch.cuda.nvtx.range_pop()


def run(args: argparse.Namespace) -> None:
    go_pre = threading.Event()
    pre = threading.Thread(target=work, args=("pre", f"cuda:{args.card}", go_pre), name="probe-pre")
    pre.start()

    import librosa
    import torch

    from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    device = f"cuda:{args.card}"
    torch.cuda.set_device(device)
    go_main = threading.Event()
    go_main.set()
    work("main", device, go_main)
    go_post = threading.Event()
    post = threading.Thread(target=work, args=("post", device, go_post), name="probe-post")
    post.start()
    go_pre.set()
    go_post.set()
    pre.join()
    post.join()

    encode_waveform = Qwen3TTSRefCodeBatcher.encode_waveform
    synchronize_outcomes = Qwen3TTSRefCodeBatcher.synchronize_outcomes

    def ranged_encode(self, waveform, sample_rate):
        torch.cuda.nvtx.range_push("probe.batcher.encode")
        try:
            return encode_waveform(self, waveform, sample_rate)
        finally:
            torch.cuda.nvtx.range_pop()

    def ranged_sync(self, outcomes):
        torch.cuda.nvtx.range_push("probe.batcher.sync")
        try:
            return synchronize_outcomes(self, outcomes)
        finally:
            torch.cuda.nvtx.range_pop()

    Qwen3TTSRefCodeBatcher.encode_waveform = ranged_encode
    Qwen3TTSRefCodeBatcher.synchronize_outcomes = ranged_sync
    clips = [librosa.load(str(path), sr=None, mono=True) for path in sorted(args.wavs.glob("*.wav"))[:CLIPS]]
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    batcher = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    for waveform, sample_rate in clips:
        batcher.encode(waveform, sample_rate)
    batcher.close()
    torch.cuda.synchronize(device)
    print(f"probe done: threads pre, main, post, batcher; {len(clips)} clips on {device}", flush=True)


def read(path: Path) -> None:
    db = sqlite3.connect(str(path))
    strings = dict(db.execute("select id, value from StringIds"))
    names = {tid: strings.get(name_id, "") for name_id, tid in db.execute("select nameId, globalTid from ThreadNames")}
    ranges: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType = ?", (PUSH_POP,)
    ):
        label = text or strings.get(text_id, "")
        if label.startswith("probe."):
            ranges[tid].append((start, end, label))
        else:
            pass
    calls: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for start, tid, name_id in db.execute("select start, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        calls[tid].append((start, strings.get(name_id, "")))
    print(f"== {path.name}: threads with runtime rows {len(calls)}, rows {sum(len(rows) for rows in calls.values())}")
    for tid, spans in sorted(ranges.items()):
        inside = Counter(
            name.split("_v")[0] for start, name in calls.get(tid, []) if any(lo <= start <= hi for lo, hi, _ in spans)
        )
        labels = Counter(label for _, _, label in spans)
        print(
            f"   tid {tid} name {names.get(tid, '?')!r}: ranges {dict(labels)}; runtime rows on the thread "
            f"{len(calls.get(tid, []))}, inside its ranges {sum(inside.values())} {dict(inside.most_common(6))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("wavs", type=Path)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--card", type=int, default=1)
    read_parser = sub.add_parser("read")
    read_parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        read(args.sqlite)


if __name__ == "__main__":
    main()
