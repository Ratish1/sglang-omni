# SPDX-License-Identifier: Apache-2.0
"""Compare incremental warm windows after restoring the same native codec state."""

import argparse
import json
from pathlib import Path

import torch

from scripts.debug_qwen3_omni_codec_waveform import load_native_scheduler
from scripts.debug_qwen3_omni_convnext_projection import difference
from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool


def snapshot(
    state_views: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in state_views.items()}


def restore(
    state_views: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    for name, value in state_views.items():
        value.copy_(expected[name])
        assert torch.equal(value, expected[name]), name


def serial_window(
    codec: Qwen3OmniIncrementalCodec,
    pool: CodecStatePool,
    codes: torch.Tensor,
    slots: list[int],
    *,
    first_decode: bool,
) -> torch.Tensor:
    return torch.cat(
        [
            codec.decode_slots(
                codes[row_index : row_index + 1],
                pool,
                pool.stage_index([slot]),
                first_decode=first_decode,
            )
            for row_index, slot in enumerate(slots)
        ]
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--witness-directory", type=Path, required=True)
    parser.add_argument("--cold-window-report", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    arguments = parser.parse_args()

    scheduler = load_native_scheduler(
        arguments.server_config, arguments.model_path, arguments.gpu_id
    )
    codec = scheduler.incremental_codec
    pool = scheduler.codec_pool
    assert codec is not None and pool is not None
    cold_report = json.loads(arguments.cold_window_report.read_bytes())
    codec_histories = json.loads(
        (arguments.witness_directory / "codec-histories.json").read_bytes()
    )
    request_ids = cold_report["peer_request_ids"]
    frame_rows = [
        torch.tensor(
            codec_histories[request_id],
            device=scheduler.device,
            dtype=torch.long,
        )
        for request_id in request_ids
    ]
    for frames in frame_rows:
        eos_positions = (frames[:, 0] == scheduler.codec_eos_token_id).nonzero()
        assert eos_positions.flatten().tolist() == [len(frames) - 1]
    first_frame_count = (
        scheduler.initial_codec_chunk_frames or scheduler.stream_chunk_size
    )
    slots = list(range(len(frame_rows)))
    initial_codes = torch.stack(
        [frames[:first_frame_count].T for frames in frame_rows]
    ).contiguous()
    assert list(initial_codes.shape) == cold_report["code_shape"]
    pool.zero_slots(pool.stage_index(slots))
    serial_window(codec, pool, initial_codes, slots, first_decode=True)
    slot_count = len(slots)
    state_views = {
        "position": pool.frame_positions[:slot_count],
        **{
            f"history.{name}": tensor[:slot_count]
            for name, tensor in pool.histories.items()
        },
        **{
            f"overlap.{name}": tensor[:slot_count]
            for name, tensor in pool.overlaps.items()
        },
        **{
            f"{kind}.{layer_index}": tensor.view(
                -1, pool.ring_frames, tensor.shape[-1]
            )[:slot_count]
            for kind, layers in (("key", pool.keys), ("value", pool.values))
            for layer_index, tensor in enumerate(layers)
        },
    }
    windows = []
    window_frames = scheduler.stream_chunk_size
    final_frame = max(len(frames) - 1 for frames in frame_rows)
    for start_frame in range(first_frame_count, final_frame, window_frames):
        end_frame = start_frame + window_frames
        active_slots = [
            slot
            for slot, frames in enumerate(frame_rows)
            if end_frame <= len(frames) - 1
        ]
        if len(active_slots) < 2:
            break
        else:
            pass
        assert pool.frame_positions[active_slots].tolist() == [start_frame] * len(
            active_slots
        )
        codes = torch.stack(
            [frame_rows[slot][start_frame:end_frame].T for slot in active_slots]
        ).contiguous()
        before = snapshot(state_views)
        solo_waveforms = serial_window(
            codec, pool, codes, active_slots, first_decode=False
        )
        serial_after = snapshot(state_views)
        restore(state_views, before)
        peer_waveforms = codec.decode_slots(
            codes, pool, pool.stage_index(active_slots), first_decode=False
        )
        peer_after = snapshot(state_views)
        restore(state_views, before)
        repeated_waveforms = serial_window(
            codec, pool, codes, active_slots, first_decode=False
        )
        assert torch.equal(solo_waveforms, repeated_waveforms)
        assert all(
            torch.equal(tensor, serial_after[name])
            for name, tensor in state_views.items()
        )
        windows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "rows": {
                    request_ids[slot]: {
                        "waveform": difference(
                            solo_waveforms[row], peer_waveforms[row]
                        ),
                        "shape_sensitive_state": {
                            name: difference(solo_after[slot], peer_after[name][slot])
                            for name, solo_after in serial_after.items()
                            if not torch.equal(solo_after[slot], peer_after[name][slot])
                        },
                    }
                    for row, slot in enumerate(active_slots)
                },
            }
        )
    assert windows
    print(
        json.dumps(
            {
                "codec_ring_frames_per_slot": pool.ring_frames,
                "retained_frames": pool.retained_frames,
                "paged_attention": codec.is_paged_attention,
                "windows": windows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
