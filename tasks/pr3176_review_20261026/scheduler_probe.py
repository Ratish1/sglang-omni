"""Compare trained first-chunk and continuation mechanics with the exact parent scheduler."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
    Qwen3TTSStreamState,
)

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer

RESULTS = Path("/data/review3176oct26")
BASE = "ab81936454eaec8d8b8b0e31446f5ee563020f0d"


def load_parent() -> type[Qwen3TTSStreamingVocoderScheduler]:
    modules = {}
    for name in ("incremental_codec", "codec_state_arena", "streaming_vocoder"):
        source = subprocess.run(
            ["git", "show", f"{BASE}:sglang_omni/models/qwen3_tts/{name}.py"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        path = RESULTS / f"parent_{name}.py"
        path.write_text(source)
        specification = importlib.util.spec_from_file_location(
            f"pr3176_parent_{name}", path
        )
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        modules[name] = module
    parent_decoder = modules["incremental_codec"]
    parent_arena = modules["codec_state_arena"]
    parent_arena.Qwen3TTSIncrementalCodecState = (
        parent_decoder.Qwen3TTSIncrementalCodecState
    )
    parent_scheduler = modules["streaming_vocoder"]
    parent_scheduler.Qwen3TTSIncrementalDecoder = (
        parent_decoder.Qwen3TTSIncrementalDecoder
    )
    parent_scheduler.Qwen3TTSCodecStateArena = parent_arena.Qwen3TTSCodecStateArena
    return parent_scheduler.Qwen3TTSStreamingVocoderScheduler


def collect_audio(
    scheduler: Qwen3TTSStreamingVocoderScheduler,
) -> dict[str, torch.Tensor]:
    waveforms = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        assert message.type == "stream", message
        assert isinstance(message.data, dict)
        waveforms[message.request_id] = torch.from_numpy(
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32).copy()
        )
    return waveforms


def compare_waveforms(
    parent: torch.Tensor, head: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | int | bool]:
    assert parent.shape == head.shape == reference.shape
    assert torch.isfinite(parent).all() and torch.isfinite(head).all()
    signal = reference.double().square().sum()

    def snr(waveform: torch.Tensor) -> float:
        error = (waveform.double() - reference.double()).square().sum()
        return float(10 * torch.log10(signal / error.clamp_min(1e-30)))

    return {
        "samples": head.numel(),
        "parent_snr_to_fp32_db": snr(parent),
        "head_snr_to_fp32_db": snr(head),
        "max_parent_head_difference": float((parent - head).abs().max()),
        "bit_equal": torch.equal(parent, head),
    }


def main() -> None:
    torch.manual_seed(3176)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    fp32_decoder = copy.deepcopy(decoder).float()
    schedulers = []
    for scheduler_class in (load_parent(), Qwen3TTSStreamingVocoderScheduler):
        scheduler = scheduler_class(
            tokenizer,
            device="cuda:0",
            async_decode=True,
            enable_stateful_codec_decoder=True,
            codec_state_slots=8,
            incremental_codec_cuda_graph=True,
            incremental_codec_compile=True,
            incremental_codec_cuda_graph_min_free_gb=0,
            has_reference_prefixed_first_chunks=True,
            followup_worker_count=1,
            followup_max_batch_size=4,
            suppress_bootstrap_silence=False,
        )
        schedulers.append(scheduler)
    for scheduler in schedulers:
        scheduler.warmup_now()
        scheduler.initial_worker = object()
        scheduler.worker_ctx.stream = scheduler.followup_decode_streams[0]
        scheduler.worker_ctx.incremental_graphs = (
            scheduler.followup_incremental_graph_holders[0]
        )
        scheduler.worker_ctx.graphs = scheduler.followup_graph_holders[0]
    parent, head = schedulers
    reach = head.incremental_decoder.first_chunk_reference_frames(1)
    assert reach == 580, reach
    references = {
        "short": 64,
        "before": reach - 1,
        "at": reach,
        "after": reach + 1,
        "long": 1100,
    }
    histories = {}
    generator = torch.Generator(device="cuda").manual_seed(3176)
    records = []

    def ingest(
        scheduler: Qwen3TTSStreamingVocoderScheduler,
        request_id: str,
        codes: torch.Tensor,
        reference_frames: int | None = None,
    ) -> Qwen3TTSStreamState:
        scheduler.decode_stream.wait_stream(torch.cuda.current_stream())
        with scheduler.decode_stream_context():
            if reference_frames is not None:
                state = scheduler.create_stream_state(request_id)
                scheduler.stream_states[request_id] = state
                scheduler.latch_stream_contract(
                    request_id,
                    state,
                    {
                        "num_quantizers": decoder.config.num_quantizers,
                        "ref_code_len": reference_frames,
                    },
                    origin="stream metadata",
                )
                state.initial_pending = True
            else:
                state = scheduler.stream_states[request_id]
            scheduler.ingest(
                request_id, state, scheduler.validate_chunk(request_id, state, codes)
            )
        return state

    for request_id, reference_frames in references.items():
        codes = torch.randint(
            decoder.config.codebook_size,
            (reference_frames + 1, decoder.config.num_quantizers),
            generator=generator,
            device="cuda",
        )
        histories[request_id] = codes
        for scheduler in schedulers:
            ingest(scheduler, request_id, codes, reference_frames)

    initial_pcm: list[dict[str, torch.Tensor]] = [{}, {}]
    for request_ids in (("short",), ("before", "at", "after"), ("long",)):
        outputs = []
        for scheduler in schedulers:
            entries = [
                (request_id, scheduler.stream_states[request_id])
                for request_id in request_ids
            ]
            with scheduler.decode_stream_context():
                scheduler.run_initial_batch(entries)
            outputs.append(collect_audio(scheduler))
            initial_pcm[len(outputs) - 1].update(outputs[-1])
            for request_id, state in entries:
                assert state.codec_frame_position == references[request_id] + 1
                assert (
                    int(scheduler.codec_arena.storage.frame_positions[state.codec_slot])
                    == references[request_id] + 1
                )
                assert state.emitted_generated_frames == 1
                assert not state.incremental_codec_fallback
        for request_id in request_ids:
            codes = histories[request_id].T.unsqueeze(0)
            reference = (
                fp32_decoder(codes)[..., -decoder.total_upsample :]
                .float()
                .cpu()
                .flatten()
            )
            record = {
                "request": request_id,
                "phase": "initial",
                **compare_waveforms(
                    outputs[0][request_id], outputs[1][request_id], reference
                ),
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    for width in (2, 4):
        request_ids = ("long", "after", "before", "at")
        for request_id in request_ids:
            codes = torch.randint(
                decoder.config.codebook_size,
                (width, decoder.config.num_quantizers),
                generator=generator,
                device="cuda",
            )
            histories[request_id] = torch.cat((histories[request_id], codes))
            for scheduler in schedulers:
                state = ingest(scheduler, request_id, codes)
                state.followup_pending = True
        outputs = []
        for scheduler in schedulers:
            entries = [
                (request_id, scheduler.stream_states[request_id])
                for request_id in request_ids
            ]
            with torch.cuda.stream(scheduler.followup_decode_streams[0]):
                scheduler.run_followup_batch(entries)
                scheduler.drain_pending_incremental(keep=0)
            outputs.append(collect_audio(scheduler))
            for request_id, state in entries:
                assert state.codec_frame_position == len(histories[request_id])
                assert int(
                    scheduler.codec_arena.storage.frame_positions[state.codec_slot]
                ) == len(histories[request_id])
                assert not state.incremental_codec_fallback
        for request_id in request_ids:
            reference = (
                fp32_decoder(histories[request_id].T.unsqueeze(0))[
                    ..., -width * decoder.total_upsample :
                ]
                .float()
                .cpu()
                .flatten()
            )
            record = {
                "request": request_id,
                "phase": f"follow-{width}",
                **compare_waveforms(
                    outputs[0][request_id], outputs[1][request_id], reference
                ),
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    for scheduler_index, scheduler in enumerate(schedulers):
        assert not scheduler.codec_slots_in_flight
        for tensor in scheduler.codec_arena._buffers(scheduler.codec_arena.storage):
            assert not tensor[7].any()
        with scheduler.decode_stream_context():
            state = scheduler.stream_states.pop("long")
            old_slot = state.codec_slot
            scheduler.release_codec_slot(state)
            new_state = ingest(scheduler, "reused", histories["long"][:1101], 1100)
            scheduler.run_initial_batch([("reused", new_state)])
            assert new_state.codec_slot == old_slot
            assert new_state.codec_frame_position == 1101
            assert not new_state.incremental_codec_fallback
        reused = collect_audio(scheduler)
        assert set(reused) == {"reused"}
        assert torch.equal(reused["reused"], initial_pcm[scheduler_index]["long"])
    result = {
        "base": BASE,
        "reach": reach,
        "records": records,
        "parent_graphs": parent.codec_state_stats(),
        "head_graphs": head.codec_state_stats(),
    }
    (RESULTS / "trained-mechanics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "trained first-chunk, mixed endpoints, continuation and slot reuse checked",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
