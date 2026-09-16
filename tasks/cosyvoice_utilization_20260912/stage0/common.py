"""Shared setup for the stage 0 experiments: the vocoder as the serving stage
builds it, and SeedTTS EN reference clips turned into token streams.

Each clip's own speech tokens are split into a prompt of the length the caller
asks for and the sequence a stream would have generated, so the Flow sees real
token statistics.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

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
    load_cosyvoice3_flow_hift,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_MEL_RATIO,
)
from sglang_omni.models.fun_cosyvoice3.utils import (
    SpeakerEncoder,
    SpeechTokenizerV3,
    extract_prompt_speech_feat,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PROMPT_AUDIO_SR = 16000
FLOW_AUDIO_SR = 24000


@dataclass(frozen=True)
class Stream:
    sample_id: str
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    tokens: torch.Tensor


def provenance(device: str) -> dict[str, str]:
    info = {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(device),
    }
    for package in (
        "sglang",
        "sgl-kernel",
        "sglang-kernel",
        "kernels",
        "flashinfer-python",
        "x-transformers",
    ):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = "absent"
    for key, value in info.items():
        print(f"{key} {value}")
    return info


def load_vocoder(
    model: str, device: str, autocast_dtype: torch.dtype | None
) -> tuple[str, CosyVoice3Vocoder]:
    checkpoint = resolve_checkpoint(model)
    flow, hift = load_cosyvoice3_flow_hift(checkpoint, device)
    _patch_chunk_mask()
    return checkpoint, CosyVoice3Vocoder(flow, hift, autocast_dtype=autocast_dtype)


def build_streams(
    checkpoint: str,
    device: str,
    *,
    count: int,
    prompt_tokens: int | Sequence[int],
    min_generated: int,
    samples: int = 1088,
) -> list[Stream]:
    """One stream per reference clip. ``prompt_tokens`` is the prompt length
    every stream takes, or one length per stream in order."""
    speech_tokenizer = SpeechTokenizerV3(
        os.path.join(checkpoint, "speech_tokenizer_v3.onnx"), device=device
    )
    speaker_encoder = SpeakerEncoder(
        os.path.join(checkpoint, "campplus.onnx"), device=device
    )
    streams: list[Stream] = []
    references: set[str] = set()
    for sample in load_seedtts_samples(
        SEEDTTS_DATASET_ID, samples, split="en", revision=SEEDTTS_DATASET_REVISION
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
        if isinstance(prompt_tokens, int):
            wanted = prompt_tokens
        else:
            wanted = int(prompt_tokens[len(streams)])
        if token.shape[1] < wanted + min_generated:
            continue
        streams.append(
            Stream(
                sample_id=sample.sample_id,
                prompt_token=token[:, :wanted].contiguous(),
                prompt_feat=feat[:, : wanted * TOKEN_MEL_RATIO].contiguous(),
                embedding=speaker_encoder.extract_embedding(audio_16k, PROMPT_AUDIO_SR),
                tokens=token[:, wanted:].contiguous(),
            )
        )
        if len(streams) == count:
            return streams
    raise RuntimeError(
        f"only {len(streams)} of the first {samples} SeedTTS EN references carry "
        f"the requested prompt plus {min_generated} speech tokens"
    )


def extend_tokens(streams: list[Stream], count: int, needed: int) -> list[Stream]:
    """The first ``count`` streams with their generated tokens grown to
    ``needed`` by appending the other streams' tokens; the causal structure
    under test does not depend on the values."""
    extended = []
    for index in range(count):
        parts = [streams[index].tokens]
        cursor = index + 1
        while sum(part.shape[1] for part in parts) < needed:
            parts.append(streams[cursor % len(streams)].tokens)
            cursor += 1
        extended.append(
            replace(streams[index], tokens=torch.cat(parts, dim=1)[:, :needed])
        )
    return extended


def hop_window(token_offset: int, hop_len: int) -> int:
    return token_offset + hop_len + PRE_LOOKAHEAD_LEN


def flow_input(stream: Stream, window: int) -> FlowBatchInput:
    return FlowBatchInput(
        token=stream.tokens[:, :window],
        prompt_token=stream.prompt_token,
        prompt_feat=stream.prompt_feat,
        embedding=stream.embedding,
    )


def compare(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    assert value.shape == reference.shape, (tuple(value.shape), tuple(reference.shape))
    value64 = value.to(torch.float64)
    reference64 = reference.to(torch.float64)
    diff = value64 - reference64
    diff_norm = float(diff.norm())
    snr_db = (
        math.inf
        if diff_norm == 0.0
        else 20 * math.log10(float(reference64.norm()) / diff_norm)
    )
    return {
        "max_abs": float(diff.abs().max()),
        "snr_db": snr_db,
        "equal": bool(torch.equal(value, reference)),
    }
