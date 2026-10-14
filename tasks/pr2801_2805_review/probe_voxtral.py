"""Exercise the actual Voxtral graph runners with small CUDA models."""

import json
from pathlib import Path

import torch
from torch import nn

from sglang_omni.models.voxtral_tts.acoustic_transformer import (
    FlowMatchingAudioTransformer,
)
from sglang_omni.models.voxtral_tts.audio_tokenizer import (
    VoxtralCodecGraphs,
    VoxtralTTSAudioTokenizer,
)
from sglang_omni.models.voxtral_tts.pipeline.stages import (
    VoxtralStreamingVocoder,
    VoxtralTTSVocoder,
)
from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel


def acoustic_probe() -> dict[str, int | bool]:
    acoustic = (
        FlowMatchingAudioTransformer(
            {
                "semantic_codebook_size": 16,
                "acoustic_codebook_size": 21,
                "n_acoustic_codebook": 4,
                "acoustic_transformer_args": {
                    "input_dim": 16,
                    "dim": 16,
                    "n_layers": 1,
                    "head_dim": 8,
                    "hidden_dim": 32,
                    "n_heads": 2,
                    "n_kv_heads": 1,
                },
            }
        )
        .to(device="cuda:0", dtype=torch.bfloat16)
        .eval()
    )
    with torch.no_grad():
        acoustic.semantic_codebook_output.weight.zero_()
        acoustic.semantic_codebook_output.weight[2].fill_(1)
        acoustic.acoustic_codebook_output.weight.zero_()
    model = VoxtralSGLangTTSModel.__new__(VoxtralSGLangTTSModel)
    nn.Module.__init__(model)
    model.acoustic_transformer = acoustic
    model.hidden_size = 16
    model.decode_input_embed_buffer = torch.zeros(
        16, 16, device="cuda", dtype=torch.bfloat16
    )
    model.acoustic_graphs = {}
    model.init_acoustic_graphs([1, 2, 4, 8, 16])
    with torch.no_grad():
        for batch_size in [1, 3, 7, 16, 17]:
            hidden = torch.ones(batch_size, 16, device="cuda", dtype=torch.bfloat16)
            bucket = next(
                (width for width in [1, 2, 4, 8, 16] if width >= batch_size), batch_size
            )
            padded = torch.zeros(bucket, 16, device="cuda", dtype=torch.bfloat16)
            padded[:batch_size] = hidden
            torch.manual_seed(2801 + batch_size)
            expected = acoustic(padded)[:batch_size]
            torch.manual_seed(2801 + batch_size)
            actual = model.run_acoustic_transformer(hidden)
            assert torch.equal(
                actual, expected
            ), f"acoustic graph mismatch at B={batch_size}"
        hidden = torch.ones(3, 16, device="cuda", dtype=torch.bfloat16)
        padded = torch.zeros(4, 16, device="cuda", dtype=torch.bfloat16)
        padded[:3] = hidden
        torch.manual_seed(2888)
        expected_first = acoustic(padded)[:3]
        expected_second = acoustic(padded)[:3]
        torch.manual_seed(2888)
        first = model.run_acoustic_transformer(hidden)
        saved = first.clone()
        second = model.run_acoustic_transformer(hidden)
        assert torch.equal(first, expected_first)
        assert torch.equal(second, expected_second)
        assert not torch.equal(first, second), "graph froze the random draw"
        assert torch.equal(first, saved), "later replay overwrote a returned result"
    return {
        "capture_buckets": len(model.acoustic_graphs),
        "batch_cases": 5,
        "fresh_rng": True,
        "output_lifetime": True,
    }


def codec_probe() -> dict[str, int | bool | float]:
    tokenizer = (
        VoxtralTTSAudioTokenizer(
            {
                "semantic_codebook_size": 16,
                "semantic_dim": 8,
                "acoustic_dim": 4,
                "dim": 64,
                "hidden_dim": 128,
                "n_heads": 2,
                "n_kv_heads": 2,
                "head_dim": 32,
                "pretransform_patch_size": 8,
            },
            {"audio_model_args": {}},
        )
        .to(device="cuda:0", dtype=torch.bfloat16)
        .eval()
    )
    with torch.no_grad():
        tokenizer.quantizer.semantic_codebook.embedding_sum.normal_()
    scheduler = VoxtralStreamingVocoder(VoxtralTTSVocoder(tokenizer), device="cuda:0")
    graphs: VoxtralCodecGraphs = scheduler.codec_graphs
    windows = []
    for index in range(35):
        frames = [1, 5, 24, 25, 50, 74, 75][index % 7]
        codes = torch.randint(2, 23, (frames, 5))
        codes[:, 0] = torch.randint(2, 18, (frames,))
        windows.append(codes)
    compared = 0
    with torch.no_grad(), torch.cuda.stream(scheduler.decode_stream):
        for count in [1, 3, 8, 16, 17, 35]:
            actual = graphs.decode(windows[:count])
            for start in range(0, count, 16):
                group = windows[start : min(count, start + 16)]
                bucket = min(width for width in graphs.graphs if width >= len(group))
                padded = torch.zeros(bucket, 5, 75, device="cuda", dtype=torch.long)
                for index, codes in enumerate(group):
                    padded[index, :, : codes.shape[0]] = (codes.t() - 2).to("cuda")
                expected = tokenizer.decode(padded, dtype=torch.bfloat16).float().cpu()
                for index, codes in enumerate(group):
                    samples = codes.shape[0] * tokenizer.downsample_factor
                    torch.testing.assert_close(
                        actual[start + index],
                        expected[index, 0, :samples],
                        rtol=0,
                        atol=0,
                    )
                    compared += 1
        retained = graphs.decode(windows[:3])
        snapshots = [audio.clone() for audio in retained]
        graphs.decode(windows[4:20])
        assert all(
            torch.equal(audio, saved)
            for audio, saved in zip(retained, snapshots, strict=True)
        )
    codes = torch.cat(windows[:7])
    frames = list(codes.unbind(0))
    full = scheduler.decode_utterance(frames)
    state = scheduler.create_stream_state("stream")
    parts = []
    for start in range(0, len(frames), 5):
        scheduler.ingest("stream", state, codes[start : start + 5])
        while scheduler.is_ready(state, is_final=False):
            participants = [("stream", state)]
            parts.append(
                scheduler.run_step(
                    participants, scheduler.build_step_plan(participants)
                )["stream"]
            )
    tail = scheduler.decode_delta("stream", state, is_final=True)
    if tail is not None:
        parts.append(tail)
    else:
        pass
    streaming = torch.cat(parts)
    assert (
        streaming.numel() == full.numel() == len(frames) * tokenizer.downsample_factor
    )
    difference = streaming - full
    signal_power = full.square().mean()
    error_power = difference.square().mean()
    snr = float(10 * torch.log10(signal_power / error_power.clamp_min(1e-30)))
    pools = {entry[0].pool() for entry in graphs.graphs.values()}
    return {
        "graph_eager_windows": compared,
        "large_batches": True,
        "output_lifetime": True,
        "stream_frames": len(frames),
        "stream_samples": streaming.numel(),
        "window_partition_snr_db": snr,
        "private_pool_count": len(pools),
    }


if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.manual_seed(2801)
    results = {"acoustic": acoustic_probe(), "codec": codec_probe()}
    destination = Path("/data/pr2801-2805-review/voxtral-gpu.json")
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
