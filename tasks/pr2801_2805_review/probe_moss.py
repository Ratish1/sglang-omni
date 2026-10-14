"""Check native MOSS-v1 graph/session mechanics and request failure isolation."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.moss_tts.attention import MossAudioTokenizerStreamingModule
from sglang_omni.models.moss_tts.audio_tokenizer import load_moss_audio_vocoder
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.streaming_vocoder import (
    MossStreamingVocoderScheduler,
    streaming_codec_view,
)
from sglang_omni.models.moss_tts.vocoder import MossTTSVocoder
from sglang_omni.models.moss_tts_local.streaming_vocoder import CodecStreamSession
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.moss_tts.test_streaming_vocoder import (
    apply_delay_pattern,
    drain,
    item,
    make_payload,
    make_scheduler,
    reference_waveform,
    streamed_audio,
)


def failure_probe() -> dict[str, bool]:
    scheduler, codec = make_scheduler(stream_slots=3, initial_chunk_frames=2)
    rows = {
        request_id: torch.randint(
            0, 99, (4, 3), generator=torch.Generator().manual_seed(index)
        )
        for index, request_id in enumerate(["a", "b", "c"])
    }
    delayed = {
        request_id: apply_delay_pattern(codes) for request_id, codes in rows.items()
    }
    for request_id in rows:
        scheduler.handle_streaming_new_request(
            request_id, make_payload(request_id, delayed[request_id])
        )
    original = codec.decode_streaming_tensors

    def fail_after_advance(codes, lengths, slots, valid_rows):
        original(codes, lengths, slots, valid_rows)
        raise RuntimeError("intentional partial decode failure")

    codec.decode_streaming_tensors = fail_after_advance
    scheduler.on_stream_chunk_batch(
        [
            ("a", item(delayed["a"][:4])),
            ("b", item(delayed["b"][:4])),
            ("c", item(delayed["c"][:3])),
        ]
    )
    messages = drain(scheduler)
    assert {message.request_id for message in messages if message.type == "error"} == {
        "a",
        "b",
    }
    assert set(scheduler.stream_states) == {"c"}
    assert len(scheduler.session.stream_slots_in_use) == 1
    codec.decode_streaming_tensors = original
    scheduler.handle_stream_chunk("c", item(delayed["c"][3:]))
    scheduler.handle_stream_done("c")
    np.testing.assert_array_equal(
        streamed_audio(drain(scheduler), "c"), reference_waveform(rows["c"]).numpy()
    )
    assert not scheduler.session.stream_slots_in_use
    scheduler.shutdown_stream_states()
    assert scheduler.session is None
    return {
        "failed_participants_released": True,
        "unaffected_stream_preserved": True,
        "shutdown_closes_session": True,
    }


def native_probe(model_path: str) -> dict[str, int | bool | float]:
    torch.cuda.set_device(1)
    audio_vocoder = load_moss_audio_vocoder(
        model_path,
        device="cuda:1",
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    )
    codec = audio_vocoder.model
    assert codec.uses_moss_audio_tokenizer_v1_weights
    count = codec.quantizer.num_quantizers
    vocabulary = codec.quantizer.quantizers[0].codebook.num_embeddings
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            n_vq=count,
            audio_pad_code=vocabulary,
            sampling_rate=audio_vocoder.sample_rate,
        )
    )
    vocoder = MossTTSVocoder(
        processor, audio_vocoder, "cuda:1", compute_dtype=torch.bfloat16
    )
    view = streaming_codec_view(codec)
    session = CodecStreamSession(view, stream_slots=4, n_vq=count)
    captured = session.warmup_cuda_graph([1, 4, 7, 16], min_free_gb=3)
    assert captured == [1, 4, 7, 16]
    runner = session.cg_runner
    assert runner is not None
    raw = torch.randint(
        0, vocabulary, (count, 3, 40), generator=torch.Generator().manual_seed(2805)
    )
    offline_rows = raw[:, 0, :12].t().contiguous()
    offline_before = vocoder.decode_segments([offline_rows])[0].clone()

    def run_trace(graphed: bool) -> list[torch.Tensor]:
        session.cg_runner = runner if graphed else None
        leased = [session.acquire() for _ in range(4)]
        assert leased == [3, 2, 1, 0]
        retained = []
        snapshots = []
        for inputs in [
            {3: raw[:, 0, :4], 2: raw[:, 1, :4]},
            {1: raw[:, 2, :7], 3: raw[:, 0, 4:11]},
        ]:
            decoded = session.step(inputs)
            retained.extend(decoded.values())
            snapshots.extend(audio.clone() for audio in decoded.values())
        offline_during = vocoder.decode_segments([offline_rows])[0]
        torch.testing.assert_close(offline_during, offline_before, rtol=0, atol=0)
        session.release(2)
        assert session.acquire() == 2
        for inputs in [
            {2: raw[:, 1, 4:8], 1: raw[:, 2, 7:11], 3: raw[:, 0, 11:15]},
            {3: raw[:, 0, 15:16], 2: raw[:, 1, 8:9]},
            {3: raw[:, 0, 16:32], 2: raw[:, 1, 9:25]},
        ]:
            decoded = session.step(inputs)
            retained.extend(decoded.values())
            snapshots.extend(audio.clone() for audio in decoded.values())
        assert all(
            torch.equal(audio, saved)
            for audio, saved in zip(retained, snapshots, strict=True)
        )
        for slot in [0, 1, 2, 3]:
            session.release(slot)
        return retained

    try:
        eager = run_trace(False)
        graph_steps_before = sum(session.cg_graph_t.values())
        graphed = run_trace(True)
        assert sum(session.cg_graph_t.values()) == graph_steps_before + 5
        for expected, actual in zip(eager, graphed, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(
            module.streaming_state is None
            for module in codec.decoder.modules()
            if isinstance(module, MossAudioTokenizerStreamingModule)
        )
    finally:
        session.close()

    scheduler = MossStreamingVocoderScheduler(
        vocoder, stream_slots=4, vocoder_cuda_graph=False
    )
    first = raw[:, 0, :9].t().contiguous()
    second = raw[:, 1, :6].t().contiguous()
    codes = torch.cat([first, torch.full((1, count), vocabulary), second])
    delayed = torch.full(
        (codes.shape[0] + count - 1, count), vocabulary, dtype=torch.long
    )
    for channel in range(count):
        delayed[channel : channel + codes.shape[0], channel] = codes[:, channel]
    request = StagePayload(
        request_id="native",
        request=OmniRequest(inputs="test", params={"stream": True}),
        data=MossTTSState(delayed_audio_codes=delayed).to_dict(),
    )
    scheduler.handle_streaming_new_request("native", request)
    for index, row in enumerate(delayed):
        scheduler.handle_stream_chunk(
            "native",
            StreamItem(
                chunk_id=index,
                data=row,
                from_stage="tts_engine",
                metadata={
                    "modality": "audio_codes",
                    "stream": True,
                    "n_vq": count,
                    "audio_pad_code": vocabulary,
                    "sample_rate": audio_vocoder.sample_rate,
                },
            ),
        )
    scheduler.handle_stream_done("native")
    messages = drain(scheduler)
    assert not any(message.type == "error" for message in messages)
    assert sum(message.type == "result" for message in messages) == 1
    samples = sum(
        len(message.data["audio_waveform"]) // 4
        for message in messages
        if message.type == "stream"
    )
    assert samples == 15 * codec.downsample_rate
    assert not scheduler.session.stream_slots_in_use
    scheduler.shutdown_stream_states()
    assert scheduler.session is None
    offline_after = vocoder.decode_segments([offline_rows])[0]
    torch.testing.assert_close(offline_after, offline_before, rtol=0, atol=0)
    return {
        "quantizers": count,
        "captured_frame_sizes": len(captured),
        "graph_steps": 5,
        "pcm_bit_identical": True,
        "nonstream_stateless": True,
        "pad_segments_valid_frames": 15,
        "pad_segments_samples": samples,
        "output_lifetime": True,
    }


def default_capacity_probe(
    model_path: str,
) -> dict[str, int | bool | float | list[int]]:
    torch.cuda.set_device(1)
    torch.cuda.set_per_process_memory_fraction(0.18, 1)
    audio_vocoder = load_moss_audio_vocoder(
        model_path,
        device="cuda:1",
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    )
    codec = audio_vocoder.model
    count = codec.quantizer.num_quantizers
    vocabulary = codec.quantizer.quantizers[0].codebook.num_embeddings
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            n_vq=count,
            audio_pad_code=vocabulary,
            sampling_rate=audio_vocoder.sample_rate,
        )
    )
    vocoder = MossTTSVocoder(
        processor, audio_vocoder, "cuda:1", compute_dtype=torch.bfloat16
    )
    scheduler = MossStreamingVocoderScheduler(vocoder)
    scheduler.warmup_now()
    session = scheduler.session
    assert session is not None
    captured = session.captured_frames()
    slots = [session.acquire() for _ in range(32)]
    assert session.acquire() is None
    raw = torch.randint(
        0, vocabulary, (count, 16), generator=torch.Generator().manual_seed(2810)
    )
    decoded = session.step({slot: raw for slot in slots})
    assert len(decoded) == 32
    assert all(
        audio.numel() == 16 * codec.downsample_rate for audio in decoded.values()
    )
    result = {
        "allocator_fraction": 0.18,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(1) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(1) / 2**30,
        "captured_frame_sizes": captured,
        "graph_count": len(session.cg_runner.graphs) if session.cg_runner else 0,
        "all_32_slots_decoded": True,
    }
    for slot in slots:
        session.release(slot)
    scheduler.shutdown_stream_states()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--default-capacity", action="store_true")
    arguments = parser.parse_args()
    if arguments.default_capacity:
        results = {"default_capacity": default_capacity_probe(arguments.model)}
        filename = "moss-capacity.json"
    else:
        results = {
            "failure_isolation": failure_probe(),
            "native": native_probe(arguments.model),
        }
        filename = "moss-gpu.json"
    Path(f"/data/pr2801-2805-review/{filename}").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results))
