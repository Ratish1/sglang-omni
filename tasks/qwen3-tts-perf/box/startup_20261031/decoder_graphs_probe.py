# Startup item 4 probe (10-31): the vocoder's 64 whole-utterance decoder graphs (frames 16 to 256 by 16, batch 1 to 4),
# captured as Qwen3TTSInitialDecodeGraphs.capture does, with --warmups eager warm-ups per shape (main: 2) and the
# decoder's attention on --attention: default (torch's SDPA choice, cuDNN first on H100) or flash (flash and
# memory-efficient SDPA only). Built through create_vocoder_executor with the codec runners' captures skipped. Prints
# the capture seconds split into first warm-up, other warm-ups and capture; each graph's device time per replay; and
# saves one replay's waveform per graph from seeded codes for the comparison across runs.
# usage: python3 decoder_graphs_probe.py --out PREFIX [--warmups 2] [--attention default|flash]
import argparse
import json
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import Qwen3TTSIncrementalCodecCudaGraphRunner
from sglang_omni.models.qwen3_tts.stages import create_vocoder_executor
from sglang_omni.models.qwen3_tts.streaming_vocoder import Qwen3TTSInitialDecodeGraphs


def capture(graphs, warmups, phases):
    stream = graphs.device_module.Stream(device=graphs.device)
    pool = graphs.graph_backend.graph_pool_handle()
    for frames in reversed(graphs.input_frames):
        for batch in reversed(graphs.batch_sizes):
            static_input = torch.zeros((batch, graphs.num_quantizers, frames), dtype=torch.long, device=graphs.device)
            stream.wait_stream(graphs.device_module.current_stream(graphs.device))
            with torch.inference_mode(), graphs.device_module.stream(stream):
                for index in range(warmups):
                    began = time.perf_counter()
                    graphs.decoder(static_input)
                    stream.synchronize()
                    phases["first warm-up" if index == 0 else "other warm-ups"] += time.perf_counter() - began
            began = time.perf_counter()
            with torch.inference_mode(), graphs.graph_backend.capture(pool=pool, stream=stream) as graph:
                static_output = graphs.decoder(static_input)
            stream.synchronize()
            phases["capture"] += time.perf_counter() - began
            graphs.graphs[frames, batch] = graph
            graphs.inputs[frames, batch] = static_input
            graphs.outputs[frames, batch] = static_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--attention", choices=("default", "flash"), default="default")
    args = parser.parse_args()
    phases_by_holder = {}

    def timed_capture(graphs):
        if not graphs.enabled or graphs.graphs:
            return
        else:
            pass
        phases = phases_by_holder.setdefault(id(graphs), {"first warm-up": 0.0, "other warm-ups": 0.0, "capture": 0.0})
        if args.attention == "flash":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                capture(graphs, args.warmups, phases)
        else:
            capture(graphs, args.warmups, phases)

    Qwen3TTSIncrementalCodecCudaGraphRunner.capture = lambda self: None
    Qwen3TTSInitialDecodeGraphs.capture = timed_capture
    began = time.perf_counter()
    scheduler = create_vocoder_executor(args.model, gpu_id=0, dtype="bfloat16", shares_engine_speech_tokenizer=True)
    built = time.perf_counter() - began
    graphs = scheduler.full_decode_graphs
    phases = phases_by_holder[id(graphs)]
    other_holders = len(phases_by_holder) - 1
    generator = torch.Generator(device="cpu").manual_seed(20261031)
    replays = {}
    waveforms = {}
    for key in sorted(graphs.graphs):
        graphs.inputs[key].copy_(torch.randint(0, 1000, tuple(graphs.inputs[key].shape), generator=generator))
        graphs.graphs[key].replay()
        torch.cuda.synchronize()
        waveforms[f"{key[0]}x{key[1]}"] = graphs.outputs[key].float().cpu().clone()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graphs.graphs[key].replay()
        end.record()
        end.synchronize()
        replays[f"{key[0]}x{key[1]}"] = round(start.elapsed_time(end) / 20, 4)
    result = {
        "warmups": args.warmups,
        "attention": args.attention,
        "build_s": round(built, 2),
        "graphs": len(graphs.graphs),
        "other_holders_captured": other_holders,
        "phases_s": {name: round(value, 2) for name, value in phases.items()},
        "replay_sum_ms": round(sum(replays.values()), 3),
        "replays_ms": replays,
    }
    with open(f"{args.out}.json", "w") as handle:
        json.dump(result, handle, indent=1)
    torch.save(waveforms, f"{args.out}.pt")
    print(json.dumps({key: result[key] for key in ("warmups", "attention", "build_s", "graphs", "phases_s", "replay_sum_ms")}), flush=True)


if __name__ == "__main__":
    main()
