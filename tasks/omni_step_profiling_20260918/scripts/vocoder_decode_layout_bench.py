"""One incremental vocoder decode at every served shape, channels first against channels
last, replayed from a CUDA graph as the served runner does.

Builds two Qwen3TTSIncrementalDecoder over one real decoder: one with the platform check
reporting no CUDA (main's channels-first path) and one as built on CUDA (channels last).
For widths 1, 2 and 8 frames at cohort buckets 1, 2, 4 and 8, captures 10 decode_tensors
calls from a full-width state and reports the median us per decode over 20 replays, and
the max abs difference between the two waveforms.

usage: python vocoder_decode_layout_bench.py
"""

from __future__ import annotations

import statistics

import torch

from sglang_omni.models.qwen3_tts import incremental_codec
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WIDTHS = (1, 2, 8)
BUCKETS = (1, 2, 4, 8)
CALLS = 10
REPLAYS = 20
SLEEP_CYCLES = 1_000_000


def replay_us(incremental, codes) -> tuple[float, torch.Tensor]:
    state = incremental.init_state(
        int(codes.shape[0]), device=codes.device, dtype=torch.bfloat16
    )
    with torch.inference_mode():
        for _ in range(3):
            incremental.decode_tensors(codes, state)
        torch.cuda.synchronize()
        state = incremental.init_state(
            int(codes.shape[0]), device=codes.device, dtype=torch.bfloat16
        )
        waveform = incremental.decode_tensors(codes, state).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(CALLS):
                incremental.decode_tensors(codes, state)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPLAYS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SLEEP_CYCLES)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / CALLS)
    return statistics.median(times), waveform


def main() -> None:
    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    decoder = tokenizer.model.decoder
    config = getattr(tokenizer.model.config, "decoder_config", tokenizer.model.config)
    is_cuda = incremental_codec.current_platform.is_cuda
    incremental_codec.current_platform.is_cuda = lambda: False
    channels_first = Qwen3TTSIncrementalDecoder(decoder)
    incremental_codec.current_platform.is_cuda = is_cuda
    channels_last = Qwen3TTSIncrementalDecoder(decoder)
    assert channels_first.channels_last_weights is None
    assert channels_last.channels_last_weights is not None
    print(torch.cuda.get_device_name(0))
    print(
        f"{'width':>6}{'bucket':>7}{'channels first us':>19}{'channels last us':>18}{'max diff':>11}"
    )
    torch.manual_seed(0)
    for frames in WIDTHS:
        for rows in BUCKETS:
            codes = torch.randint(
                0,
                int(config.codebook_size),
                (rows, int(config.num_quantizers), frames),
                device="cuda",
            )
            first_us, first_waveform = replay_us(channels_first, codes)
            last_us, last_waveform = replay_us(channels_last, codes)
            difference = float(
                (first_waveform.float() - last_waveform.float()).abs().max()
            )
            print(
                f"{frames:>6}{rows:>7}{first_us:>19.1f}{last_us:>18.1f}{difference:>11.2e}"
            )


if __name__ == "__main__":
    main()
