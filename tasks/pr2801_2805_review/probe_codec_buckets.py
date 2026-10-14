"""Verify native codec state and graph replay at the merged wider batch bucket."""

import argparse
import json
from pathlib import Path

import torch

from sglang_omni.models.moss_tts.audio_tokenizer import load_moss_audio_vocoder
from sglang_omni.models.moss_tts.codec_session import CodecStreamSession


def verify_codec(
    model_path: str,
    *,
    device_index: int,
    quantizer_count: int,
    frame_counts: list[int],
    output_path: Path,
) -> None:
    torch.cuda.set_device(device_index)
    codec = load_moss_audio_vocoder(
        model_path,
        device=f"cuda:{device_index}",
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    ).model
    session = CodecStreamSession(codec, stream_slots=32, n_vq=quantizer_count)
    try:
        captured = session.warmup_cuda_graph(frame_counts)
        assert captured == sorted(frame_counts)
        runner = session.cg_runner
        assert runner is not None
        assert set(runner.capture_sizes) == {
            (batch_size, frame_count)
            for batch_size in session.graph_batch_sizes()
            for frame_count in frame_counts
        }
        slots = [session.acquire() for _ in range(32)]
        assert all(isinstance(slot, int) for slot in slots)
        assert session.acquire() is None
        vocabulary_size = codec.quantizer.quantizers[0].codebook.num_embeddings
        codes = torch.randint(
            0,
            vocabulary_size,
            (quantizer_count, 32, max(frame_counts)),
            generator=torch.Generator().manual_seed(2805),
        )
        cohorts = [
            slots[:17],
            [slots[0], *reversed(slots[8:24])],
            [slots[31], *slots[1:17]],
        ]
        assert all(len(set(cohort)) == 17 for cohort in cohorts)
        assert slots[31] not in cohorts[0] and slots[31] not in cohorts[1]
        assert next(size for size in session.graph_batch_sizes() if size >= 17) == 24
        inputs = [
            {slot: codes[:, slot, :frame_count] for slot in cohort}
            for cohort, frame_count in zip(cohorts, frame_counts, strict=True)
        ]
        session.cg_runner = None
        fresh_output = session.step(inputs[2])[slots[31]].clone()

        def run_trace(graphed: bool) -> list[dict[int, torch.Tensor]]:
            session.cg_runner = runner if graphed else None
            session.reset_slots(slots)
            outputs = [session.step(slot_codes) for slot_codes in inputs]
            assert torch.equal(outputs[2][slots[31]], fresh_output)
            released_slot = slots[0]
            session.release(released_slot)
            assert session.acquire() == released_slot
            retained = [
                {slot: waveform.clone() for slot, waveform in output.items()}
                for output in outputs
            ]
            outputs.append(session.step(inputs[0]))
            assert torch.equal(outputs[3][released_slot], outputs[0][released_slot])
            for output, snapshot in zip(outputs[:3], retained, strict=True):
                for slot, waveform in snapshot.items():
                    assert torch.equal(output[slot], waveform)
            for output in outputs:
                assert len(output) == 17
                for waveform in output.values():
                    assert waveform.device.type == "cpu"
                    assert waveform.dtype == torch.float32
                    assert waveform.shape[0] == codec.number_channels
                    assert waveform.numel() > 0 and torch.isfinite(waveform).all()
            return outputs

        eager = run_trace(False)
        graph_steps_before = sum(session.cg_graph_t.values())
        graphed = run_trace(True)
        assert sum(session.cg_graph_t.values()) == graph_steps_before + 4
        assert sum(session.cg_eager_t.values()) == 0
        for actual, expected in zip(graphed, eager, strict=True):
            for slot, waveform in expected.items():
                assert torch.equal(actual[slot], waveform)
        for slot in slots:
            session.release(slot)
        assert not session.stream_slots_in_use
        report = {
            "model": model_path,
            "device_index": device_index,
            "quantizers": quantizer_count,
            "channels": codec.number_channels,
            "frame_counts": frame_counts,
            "active_rows": 17,
            "execution_bucket": 24,
            "capture_keys": len(runner.capture_sizes),
            "graph_steps": sum(session.cg_graph_t.values()),
            "eager_fallback_steps": sum(session.cg_eager_t.values()),
            "pcm_bit_identical": True,
            "inactive_slot_preserved": True,
            "released_slot_starts_fresh": True,
            "cpu_output_lifetime": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
        }
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--quantizers", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    verify_codec(
        options.model,
        device_index=options.device_index,
        quantizer_count=options.quantizers,
        frame_counts=options.frames,
        output_path=options.output,
    )
