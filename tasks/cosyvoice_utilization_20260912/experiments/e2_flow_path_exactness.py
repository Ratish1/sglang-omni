#!/usr/bin/env python3
"""E2: the packed Flow paths equal the native singleton path.

Run on the H100 venv at the branch head:

  python tasks/cosyvoice_utilization_20260912/experiments/e2_flow_path_exactness.py \
      --checkpoint /path/to/Fun-CosyVoice3-0.5B-2512 --device cuda:0

Real SeedTTS reference clips supply the conditioning and the speech tokens:
each clip's own speech tokens are split into a hop-aligned prompt and the
generated sequence a stream would have produced, so the Flow sees real token
statistics rather than random ids.

Mel rows compare the frames one call appends to the stream history: the native
path exposes only the accumulated hift_mel that token2wav_chunk returns, so the
new frames are its tail past the history it was given. Waveform rows compare
the emitted delta.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import torch
from cosyvoice.flow.DiT import dit as cosyvoice_dit
from cosyvoice.transformer.convolution import CausalConv1d
from cosyvoice.utils.mask import add_optional_chunk_mask

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    _align_flow_prompt,
    _load_prompt_audio,
    _load_prompt_audio_24k,
)
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    _patch_chunk_mask,
    generate_flow,
    load_cosyvoice3_flow_hift,
    pack_flow_inputs,
    split_generated_mels,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
)
from sglang_omni.models.fun_cosyvoice3.utils import (
    SpeakerEncoder,
    SpeechTokenizerV3,
    extract_prompt_speech_feat,
)

# note(ratish): captured before load_cosyvoice3_flow_hift and _patch_chunk_mask
# install their patches, so the reference runs can call the unpatched functions.
ORIGINAL_CONV_FORWARD = CausalConv1d.forward
ORIGINAL_CHUNK_MASK = add_optional_chunk_mask

PROMPT_AUDIO_SR = 16000
FLOW_AUDIO_SR = 24000

# note(ratish): a multiple of the hop, so prompt_token_pad is zero and the
# generated windows are exactly the ones the streaming scheduler asks for.
PROMPT_TOKENS = 2 * TOKEN_HOP_LEN

# (token_offset, hop_len) for the first two causal hops; hop doubles after one
# successful chunk. One stream per hop, so the mixed batch rows differ.
HOPS = ((0, TOKEN_HOP_LEN), (TOKEN_HOP_LEN, 2 * TOKEN_HOP_LEN))


@dataclass(frozen=True)
class Stream:
    sample_id: str
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    tokens: torch.Tensor


def hop_window(token_offset: int, hop_len: int) -> int:
    return token_offset + hop_len + PRE_LOOKAHEAD_LEN


def flow_input(stream: Stream, window: int) -> FlowBatchInput:
    return FlowBatchInput(
        token=stream.tokens[:, :window],
        prompt_token=stream.prompt_token,
        prompt_feat=stream.prompt_feat,
        embedding=stream.embedding,
    )


def build_streams(checkpoint: str, device: str, samples: int) -> list[Stream]:
    speech_tokenizer = SpeechTokenizerV3(
        os.path.join(checkpoint, "speech_tokenizer_v3.onnx"), device=device
    )
    speaker_encoder = SpeakerEncoder(
        os.path.join(checkpoint, "campplus.onnx"), device=device
    )
    needed = PROMPT_TOKENS + hop_window(*HOPS[-1])
    streams: list[Stream] = []
    references: set[str] = set()
    for sample in load_seedtts_samples(
        SEEDTTS_DATASET_ID,
        samples,
        split="en",
        revision=SEEDTTS_DATASET_REVISION,
    ):
        if sample.ref_audio in references:
            continue
        references.add(sample.ref_audio)
        audio_16k = _load_prompt_audio(sample.ref_audio)
        audio_24k = _load_prompt_audio_24k(sample.ref_audio)
        token, feat = _align_flow_prompt(
            speech_tokenizer.extract_speech_token(audio_16k, PROMPT_AUDIO_SR),
            extract_prompt_speech_feat(audio_24k, FLOW_AUDIO_SR),
        )
        if token.shape[1] < needed:
            continue
        streams.append(
            Stream(
                sample_id=sample.sample_id,
                prompt_token=token[:, :PROMPT_TOKENS].contiguous(),
                prompt_feat=feat[:, : PROMPT_TOKENS * TOKEN_MEL_RATIO].contiguous(),
                embedding=speaker_encoder.extract_embedding(audio_16k, PROMPT_AUDIO_SR),
                tokens=token[:, PROMPT_TOKENS:].contiguous(),
            )
        )
        if len(streams) == len(HOPS):
            return streams
    raise RuntimeError(
        f"Only {len(streams)} of the first {samples} SeedTTS EN clips carry the "
        f"{needed} speech tokens the hop ladder needs"
    )


Row = tuple[str, float, float, float, float]


def compare(name: str, value: torch.Tensor, reference: torch.Tensor) -> Row:
    assert value.shape == reference.shape, (
        name,
        tuple(value.shape),
        tuple(reference.shape),
    )
    reference = reference.to(torch.float64)
    diff = value.to(torch.float64) - reference
    max_abs = float(diff.abs().max())
    # note(ratish): max_rel blows up on near-silent samples, so the SNR of the
    # whole tensor is the number that says whether a difference is audible.
    snr_db = 20 * torch.log10(reference.norm() / diff.norm()).item()
    return (
        name,
        max_abs,
        max_abs / float(reference.abs().max()),
        float(diff.abs().mean()),
        snr_db,
    )


def print_table(title: str, rows: list[Row]) -> None:
    print(f"\n{title}")
    print(f"{'row':<52}{'max_abs':>13}{'max_rel':>13}{'mean_abs':>13}{'snr_db':>10}")
    for name, max_abs, max_rel, mean_abs, snr_db in rows:
        print(
            f"{name:<52}{max_abs:>13.3e}{max_rel:>13.3e}{mean_abs:>13.3e}{snr_db:>10.1f}"
        )
    print(
        f"largest max_rel: {max(row[2] for row in rows):.3e}, "
        f"lowest snr_db: {min(row[4] for row in rows):.1f}"
    )


def packed_chain(
    vocoder: CosyVoice3Vocoder,
    stream: Stream,
    hops: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor | None, int]:
    hift_mel: torch.Tensor | None = None
    speech_offset = 0
    for token_offset, hop_len in hops:
        mel = vocoder.hop_batch(
            [flow_input(stream, hop_window(token_offset, hop_len))]
        )[0]
        _, hift_mel, speech_offset = vocoder.hift_delta(
            mel[:, :, token_offset * TOKEN_MEL_RATIO :],
            hift_mel=hift_mel,
            speech_offset=speech_offset,
            finalize=False,
        )
    return hift_mel, speech_offset


def native_chain(
    vocoder: CosyVoice3Vocoder,
    stream: Stream,
    hops: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor | None, int]:
    hift_mel: torch.Tensor | None = None
    speech_offset = 0
    for token_offset, hop_len in hops:
        _, hift_mel, speech_offset = vocoder.token2wav_chunk(
            token=stream.tokens[:, : hop_window(token_offset, hop_len)],
            prompt_token=stream.prompt_token,
            prompt_feat=stream.prompt_feat,
            embedding=stream.embedding,
            token_offset=token_offset,
            streaming=True,
            finalize=False,
            hift_mel=hift_mel,
            speech_offset=speech_offset,
        )
    return hift_mel, speech_offset


def measure_repeat_noise(vocoder: CosyVoice3Vocoder, stream: Stream) -> list[Row]:
    token_offset, hop_len = HOPS[0]
    window = hop_window(token_offset, hop_len)
    native = [
        vocoder.token2wav_chunk(
            token=stream.tokens[:, :window],
            prompt_token=stream.prompt_token,
            prompt_feat=stream.prompt_feat,
            embedding=stream.embedding,
            token_offset=token_offset,
            streaming=True,
            finalize=False,
            hift_mel=None,
            speech_offset=0,
        )
        for _ in range(2)
    ]
    packed = []
    for _ in range(2):
        mel = vocoder.hop_batch([flow_input(stream, window)])[0]
        packed.append(
            vocoder.hift_delta(mel, hift_mel=None, speech_offset=0, finalize=False)
        )
    label = f"{stream.sample_id} offset={token_offset} win={window}"
    return [
        compare(f"{label} native twice mel", native[1][1], native[0][1]),
        compare(f"{label} native twice wav", native[1][0], native[0][0]),
        compare(f"{label} packed twice mel", packed[1][1], packed[0][1]),
        compare(f"{label} packed twice wav", packed[1][0], packed[0][0]),
    ]


def measure_native_vs_packed_hops(
    vocoder: CosyVoice3Vocoder, streams: list[Stream]
) -> list[Row]:
    rows: list[Row] = []
    for stream in streams:
        native_mel: torch.Tensor | None = None
        native_offset = 0
        packed_mel: torch.Tensor | None = None
        packed_offset = 0
        for token_offset, hop_len in HOPS:
            window = hop_window(token_offset, hop_len)
            history = 0 if native_mel is None else native_mel.shape[-1]
            native_delta, native_mel, native_offset = vocoder.token2wav_chunk(
                token=stream.tokens[:, :window],
                prompt_token=stream.prompt_token,
                prompt_feat=stream.prompt_feat,
                embedding=stream.embedding,
                token_offset=token_offset,
                streaming=True,
                finalize=False,
                hift_mel=native_mel,
                speech_offset=native_offset,
            )
            mel = vocoder.hop_batch([flow_input(stream, window)])[0]
            packed_delta, packed_mel, packed_offset = vocoder.hift_delta(
                mel[:, :, token_offset * TOKEN_MEL_RATIO :],
                hift_mel=packed_mel,
                speech_offset=packed_offset,
                finalize=False,
            )
            label = f"{stream.sample_id} offset={token_offset} win={window}"
            rows.append(
                compare(
                    f"{label} mel",
                    packed_mel[:, :, history:],
                    native_mel[:, :, history:],
                )
            )
            rows.append(compare(f"{label} wav", packed_delta, native_delta))
    return rows


def measure_mixed_offset_batch(
    vocoder: CosyVoice3Vocoder, streams: list[Stream]
) -> list[Row]:
    plan = []
    for index, (stream, (token_offset, hop_len)) in enumerate(
        zip(streams, HOPS, strict=True)
    ):
        hift_mel, speech_offset = packed_chain(vocoder, stream, HOPS[:index])
        plan.append(
            (
                stream,
                token_offset,
                hop_window(token_offset, hop_len),
                hift_mel,
                speech_offset,
            )
        )
    mixed = vocoder.hop_batch(
        [flow_input(stream, window) for stream, _, window, _, _ in plan]
    )
    rows: list[Row] = []
    for (stream, token_offset, window, hift_mel, speech_offset), mixed_mel in zip(
        plan, mixed, strict=True
    ):
        solo_mel = vocoder.hop_batch([flow_input(stream, window)])[0]
        offset_frames = token_offset * TOKEN_MEL_RATIO
        mixed_delta, _, _ = vocoder.hift_delta(
            mixed_mel[:, :, offset_frames:],
            hift_mel=hift_mel,
            speech_offset=speech_offset,
            finalize=False,
        )
        solo_delta, _, _ = vocoder.hift_delta(
            solo_mel[:, :, offset_frames:],
            hift_mel=hift_mel,
            speech_offset=speech_offset,
            finalize=False,
        )
        label = f"{stream.sample_id} offset={token_offset} win={window}"
        rows.append(
            compare(
                f"{label} mel",
                mixed_mel[:, :, offset_frames:],
                solo_mel[:, :, offset_frames:],
            )
        )
        rows.append(compare(f"{label} wav", mixed_delta, solo_delta))
    return rows


def measure_native_vs_packed_leftover(
    vocoder: CosyVoice3Vocoder, streams: list[Stream]
) -> list[Row]:
    token_offset = HOPS[-1][0] + HOPS[-1][1]
    native_state = [native_chain(vocoder, stream, HOPS) for stream in streams]
    packed_state = [packed_chain(vocoder, stream, HOPS) for stream in streams]
    batched = vocoder.leftover_batch(
        [flow_input(stream, stream.tokens.shape[1]) for stream in streams]
    )
    rows: list[Row] = []
    for stream, native, packed, batched_mel in zip(
        streams, native_state, packed_state, batched, strict=True
    ):
        offset_frames = token_offset * TOKEN_MEL_RATIO
        native_delta, native_hift_mel, _ = vocoder.token2wav_chunk(
            token=stream.tokens,
            prompt_token=stream.prompt_token,
            prompt_feat=stream.prompt_feat,
            embedding=stream.embedding,
            token_offset=token_offset,
            streaming=False,
            finalize=True,
            hift_mel=native[0],
            speech_offset=native[1],
        )
        native_mel = native_hift_mel[:, :, native[0].shape[-1] :]
        batched_delta, _, _ = vocoder.hift_delta(
            batched_mel[:, :, offset_frames:],
            hift_mel=packed[0],
            speech_offset=packed[1],
            finalize=True,
        )
        solo_mel = vocoder.leftover_batch([flow_input(stream, stream.tokens.shape[1])])[
            0
        ]
        solo_delta, _, _ = vocoder.hift_delta(
            solo_mel[:, :, offset_frames:],
            hift_mel=packed[0],
            speech_offset=packed[1],
            finalize=True,
        )
        label = f"{stream.sample_id} offset={token_offset} n={stream.tokens.shape[1]}"
        rows.append(
            compare(
                f"{label} batch vs native mel",
                batched_mel[:, :, offset_frames:],
                native_mel,
            )
        )
        rows.append(
            compare(f"{label} batch vs native wav", batched_delta, native_delta)
        )
        rows.append(
            compare(
                f"{label} batch vs solo mel",
                batched_mel[:, :, offset_frames:],
                solo_mel[:, :, offset_frames:],
            )
        )
        rows.append(compare(f"{label} batch vs solo wav", batched_delta, solo_delta))
    return rows


def _padded_hops(
    vocoder: CosyVoice3Vocoder, items: list[FlowBatchInput]
) -> list[torch.Tensor]:
    flow = vocoder.flow
    packed = pack_flow_inputs(flow.flow, items)
    with torch.autocast(
        device_type="cuda",
        dtype=vocoder.autocast_dtype,
        enabled=vocoder.autocast_dtype is not None,
    ):
        generated = generate_flow(flow, packed, streaming=True, finalize=False)
    lookahead = flow.flow.pre_lookahead_len
    return split_generated_mels(
        flow.flow,
        packed,
        generated,
        token_lengths=tuple(
            max(length - lookahead, 0) for length in packed.combined_token_lengths
        ),
        target_token_lengths=tuple(
            max(length - lookahead, 0) for length in packed.target_token_lengths
        ),
    )


def _padded_finals(
    vocoder: CosyVoice3Vocoder, items: list[FlowBatchInput]
) -> list[torch.Tensor]:
    flow = vocoder.flow
    packed = pack_flow_inputs(flow.flow, items)
    with torch.autocast(
        device_type="cuda",
        dtype=vocoder.autocast_dtype,
        enabled=vocoder.autocast_dtype is not None,
    ):
        generated = generate_flow(flow, packed, streaming=False, finalize=True)
    return split_generated_mels(
        flow.flow,
        packed,
        generated,
        token_lengths=packed.combined_token_lengths,
        target_token_lengths=packed.target_token_lengths,
    )


def measure_packed_rows_vs_padded_rows(
    vocoder: CosyVoice3Vocoder, streams: list[Stream]
) -> list[Row]:
    """The packed sequence with per row attention against the padded DiT
    call on the same rows, mixed hop windows and then the finals."""
    rows: list[Row] = []
    hop_items = [
        flow_input(stream, hop_window(*HOPS[index % len(HOPS)]))
        for index, stream in enumerate(streams)
    ]
    for stream, packed_mel, padded_mel in zip(
        streams,
        vocoder.hop_batch(hop_items),
        _padded_hops(vocoder, hop_items),
        strict=True,
    ):
        rows.append(compare(f"{stream.sample_id} hop mel", packed_mel, padded_mel))
    final_items = [flow_input(stream, stream.tokens.shape[1]) for stream in streams]
    for stream, packed_mel, padded_mel in zip(
        streams,
        vocoder.leftover_batch(final_items),
        _padded_finals(vocoder, final_items),
        strict=True,
    ):
        rows.append(compare(f"{stream.sample_id} final mel", packed_mel, padded_mel))
    return rows


def measure_bf16_paths_against_float32(
    vocoder: CosyVoice3Vocoder, flow: object, hift: object, streams: list[Stream]
) -> list[Row]:
    """The padded call under bfloat16 autocast (the path that shipped) and
    the packed rows under the same autocast, each against the float32 padded
    call. The packed rows must sit exactly where the padded call sits."""
    serving = CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    hop_items = [
        flow_input(stream, hop_window(*HOPS[index % len(HOPS)]))
        for index, stream in enumerate(streams)
    ]
    final_items = [flow_input(stream, stream.tokens.shape[1]) for stream in streams]
    truth = {
        "hop": _padded_hops(vocoder, hop_items),
        "final": _padded_finals(vocoder, final_items),
    }
    paths = {
        "padded": (
            _padded_hops(serving, hop_items),
            _padded_finals(serving, final_items),
        ),
        "packed": (
            serving.hop_batch(hop_items),
            serving.leftover_batch(final_items),
        ),
    }
    rows: list[Row] = []
    for kind_index, kind in enumerate(("hop", "final")):
        for name, mels in paths.items():
            for stream, mel, reference in zip(
                streams, mels[kind_index], truth[kind], strict=True
            ):
                rows.append(
                    compare(f"{stream.sample_id} {kind} {name}", mel, reference)
                )
    return rows


def measure_load_time_patches(
    vocoder: CosyVoice3Vocoder,
    stream: Stream,
    patched_conv_forward: Callable[..., torch.Tensor],
) -> list[Row]:
    token_offset, hop_len = HOPS[0]
    window = hop_window(token_offset, hop_len)
    item = flow_input(stream, window)

    cosyvoice_dit.add_optional_chunk_mask = ORIGINAL_CHUNK_MASK
    reference_mel = vocoder.hop_batch([item])[0]
    _patch_chunk_mask()
    patched_mel = vocoder.hop_batch([item])[0]

    tail = patched_mel[:, :, token_offset * TOKEN_MEL_RATIO :]
    CausalConv1d.forward = ORIGINAL_CONV_FORWARD
    reference_delta, _, _ = vocoder.hift_delta(
        tail, hift_mel=None, speech_offset=0, finalize=False
    )
    CausalConv1d.forward = patched_conv_forward
    patched_delta, _, _ = vocoder.hift_delta(
        tail, hift_mel=None, speech_offset=0, finalize=False
    )

    label = f"{stream.sample_id} offset={token_offset} win={window}"
    return [
        compare(f"{label} chunk mask patch mel", patched_mel, reference_mel),
        compare(f"{label} conv cache patch wav", patched_delta, reference_delta),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"checkpoint {args.checkpoint}")
    print(f"branch head {head}")
    print(f"torch {torch.__version__}")
    print(f"device {args.device} {torch.cuda.get_device_name(args.device)}")
    print(
        "mel rows compare the frames one call appends to the stream history; "
        "wav rows compare the emitted waveform delta."
    )

    flow, hift = load_cosyvoice3_flow_hift(args.checkpoint, args.device)
    patched_conv_forward = CausalConv1d.forward
    _patch_chunk_mask()
    vocoder = CosyVoice3Vocoder(flow, hift)
    streams = build_streams(args.checkpoint, args.device, args.samples)
    print(f"streams {[stream.sample_id for stream in streams]}")

    with torch.inference_mode():
        print_table(
            "0. the same path run twice, the noise floor",
            measure_repeat_noise(vocoder, streams[0]),
        )
        print_table(
            "1. native token2wav_chunk hop vs packed hop_batch size 1",
            measure_native_vs_packed_hops(vocoder, streams),
        )
        print_table(
            "2. mixed offset hop_batch vs solo hop_batch rows",
            measure_mixed_offset_batch(vocoder, streams),
        )
        print_table(
            "3. native leftover vs packed leftover_batch",
            measure_native_vs_packed_leftover(vocoder, streams),
        )
        print_table(
            "4. load time patches vs the unpatched cosyvoice functions",
            measure_load_time_patches(vocoder, streams[0], patched_conv_forward),
        )
        print_table(
            "5. packed rows with per row attention vs the padded DiT call",
            measure_packed_rows_vs_padded_rows(vocoder, streams),
        )
        print_table(
            "6. the padded and the packed call under bfloat16 against the float32 call",
            measure_bf16_paths_against_float32(vocoder, flow, hift, streams),
        )


if __name__ == "__main__":
    main()
