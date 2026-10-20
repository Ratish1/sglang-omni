"""One tree's HiFT as its vocoder patches it (real checkpoint): the decode graph's replay time per row and frame tier,
the eager whole utterance call's wall (the buffered path), and the outputs on seeded inputs, saved for a comparison
across trees. usage: PYTHONPATH=<tree> python3 hift_decode_tree_c20261018.py <out.pt>
compare: python3 hift_decode_tree_c20261018.py --compare <a.pt> <b.pt>"""

import math
import statistics
import sys
import time

import torch

ROW_TIERS = (1, 2, 4, 8, 12, 16)
FRAME_TIERS = (64, 128, 256)
EAGER_FRAMES = 400
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def measure(out_path: str) -> None:
    from huggingface_hub import snapshot_download
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    from sglang_omni.models.fun_cosyvoice3 import hift_kernels, stages

    device = torch.device("cuda")
    print(f"tree {hift_kernels.__file__}", flush=True)
    checkpoint_dir = snapshot_download(MODEL, local_files_only=True)
    _, hift = stages.load_cosyvoice3_flow_hift_lightweight(checkpoint_dir, device="cuda")
    stages.keep_hift_constants_on_device(hift, "cuda")
    stages.patch_causal_conv_cache()
    for submodule in list(hift.modules()):
        while is_parametrized(submodule):
            name = next(iter(submodule.parametrizations.keys()))
            remove_parametrizations(submodule, name, leave_parametrized=True)
    stages.patch_hift_snake(hift)
    stages.patch_hift_spectra(hift, device)
    samples_per_frame = int(math.prod(hift.upsample_rates) * hift.istft_params["hop_len"])
    generator = torch.Generator().manual_seed(0)
    result = {"replay_ms": {}, "outputs": {}}
    with torch.inference_mode():
        for frames in FRAME_TIERS:
            for rows in ROW_TIERS:
                mel = (torch.randn(rows, 80, frames, generator=generator) * 2 - 6).to(device)
                source = (torch.randn(rows, 1, frames * samples_per_frame, generator=generator) * 0.1).to(device)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        hift.decode(x=mel, s=source, finalize=True)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = hift.decode(x=mel, s=source, finalize=True)
                times = []
                for _ in range(9):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) / 20)
                graph.replay()
                torch.cuda.synchronize()
                key = f"rows={rows} F={frames}"
                result["replay_ms"][key] = statistics.median(times)
                result["outputs"][key] = output.float().cpu().clone()
                print(f"{key} {statistics.median(times):.3f} ms", flush=True)
                del graph, output
                torch.cuda.empty_cache()
        mel = (torch.randn(1, 80, EAGER_FRAMES, generator=generator) * 2 - 6).to(device)
        for _ in range(5):
            hift.inference(speech_feat=mel, finalize=True)
        torch.cuda.synchronize()
        walls = []
        for _ in range(50):
            start = time.perf_counter()
            speech, _ = hift.inference(speech_feat=mel, finalize=True)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - start) * 1000)
        result["eager_ms"] = statistics.median(walls)
        print(f"eager F={EAGER_FRAMES} {result['eager_ms']:.3f} ms", flush=True)
    torch.save(result, out_path)


def compare(a_path: str, b_path: str) -> None:
    a = torch.load(a_path)
    b = torch.load(b_path)
    ratios = []
    for key, a_ms in a["replay_ms"].items():
        b_ms = b["replay_ms"][key]
        ratios.append(b_ms / a_ms)
        difference = (b["outputs"][key] - a["outputs"][key]).abs()
        int16_differs = ((b["outputs"][key] * 32767).round() != (a["outputs"][key] * 32767).round()).float().mean()
        print(f"{key}: {a_ms:.3f} -> {b_ms:.3f} ms ({b_ms / a_ms - 1:+.2%}) max|d| {difference.max().item():.2e} "
              f"int16 differs {int16_differs.item():.2e}")
    print(f"geomean replay {math.exp(sum(math.log(x) for x in ratios) / len(ratios)) - 1:+.2%}")
    print(f"eager F={EAGER_FRAMES}: {a['eager_ms']:.3f} -> {b['eager_ms']:.3f} ms "
          f"({b['eager_ms'] / a['eager_ms'] - 1:+.2%})")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        measure(sys.argv[1])
