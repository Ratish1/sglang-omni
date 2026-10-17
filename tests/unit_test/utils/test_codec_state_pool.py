# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.platforms import current_platform
from sglang_omni.utils.codec_state_pool import (
    CodecStatePool,
    history_exchange,
    history_exchange_reference,
    launch_history_exchange,
    launch_overlap_exchange,
    overlap_exchange,
    overlap_exchange_reference,
    ring_addresses,
)
from sglang_omni.utils.incremental_codec import IncrementalCodecStateSpec


def test_zeroing_capture_slots_leaves_every_other_slot() -> None:
    spec = IncrementalCodecStateSpec(
        conv_histories=(("a", 3, 6), ("b", 2, 0)),
        transconv_overlaps=(("c", 4, 2),),
        num_layers=2,
        num_key_value_heads=2,
        head_dim=4,
        retained_context=5,
    )
    pool = CodecStatePool(
        spec,
        num_slots=3,
        capture_slots=2,
        ring_frames=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tensors = [
        *pool.histories.values(),
        *pool.overlaps.values(),
        *pool.keys,
        *pool.values,
        pool.frame_positions,
    ]
    for tensor in tensors:
        tensor.fill_(1)

    pool.zero_slots(torch.tensor(pool.scratch_slots))

    assert set(pool.histories) == {"a"}
    assert pool.scratch_slots == (3, 4)
    for tensor in [*pool.histories.values(), *pool.overlaps.values()]:
        assert not tensor[3:].any() and tensor[:3].all()
    for ring in [*pool.keys, *pool.values]:
        assert not ring[3 * 8 : pool.dump_row].any() and ring[: 3 * 8].all()
    assert pool.frame_positions.tolist() == [1, 1, 1, 0, 0]


@pytest.mark.parametrize("history_samples", [2, 6, 54])
@pytest.mark.parametrize("chunks", [(10, 10, 7), (1, 2, 1, 3, 40), (32, 1, 32)])
def test_history_exchange_builds_the_causal_window_of_the_whole_stream(
    history_samples: int, chunks: tuple[int, ...]
) -> None:
    torch.manual_seed(history_samples + len(chunks))
    slots, channels = 5, 3
    slot_index = torch.tensor([3, 1])
    history_pool = torch.randn(slots, history_samples, channels)
    untouched = history_pool[[0, 2, 4]].clone()
    stream = torch.randn(2, sum(chunks), channels)
    whole = F.pad(stream, (0, 0, history_samples, 0))
    start = 0
    for index, new_samples in enumerate(chunks):
        padded = torch.full((2, history_samples + new_samples + 2, channels), 7.0)
        padded[:, history_samples : history_samples + new_samples] = stream[
            :, start : start + new_samples
        ]

        history_exchange(
            padded,
            history_pool,
            slot_index,
            new_samples=new_samples,
            is_cold=index == 0,
        )

        torch.testing.assert_close(
            padded[:, : history_samples + new_samples],
            whole[:, start : start + history_samples + new_samples],
            rtol=0,
            atol=0,
        )
        assert not padded[:, history_samples + new_samples :].any()
        start += new_samples
    torch.testing.assert_close(
        history_pool[slot_index], whole[:, -history_samples:], rtol=0, atol=0
    )
    torch.testing.assert_close(history_pool[[0, 2, 4]], untouched, rtol=0, atol=0)


@pytest.mark.parametrize("stride", [3, 5, 8])
@pytest.mark.parametrize("chunks", [(4, 4, 2), (1, 1, 1, 5)])
def test_overlap_exchange_streams_the_whole_transposed_conv(
    stride: int, chunks: tuple[int, ...]
) -> None:
    torch.manual_seed(stride + len(chunks))
    weight = torch.randn(4, 3, 2 * stride)
    inputs = torch.randn(2, 4, sum(chunks))
    whole = F.conv_transpose1d(inputs, weight, stride=stride)
    overlap_pool = torch.randn(4, stride, 3)
    slot_index = torch.tensor([2, 0])
    emitted = []
    start = 0
    for index, frames in enumerate(chunks):
        output = F.conv_transpose1d(
            inputs[..., start : start + frames], weight, stride=stride
        ).transpose(1, 2)

        overlap_exchange(
            output,
            overlap_pool,
            slot_index,
            emitted_samples=frames * stride,
            is_cold=index == 0,
        )

        emitted.append(output[:, : frames * stride])
        start += frames
    torch.testing.assert_close(
        torch.cat(emitted, dim=1).transpose(1, 2), whole[..., : start * stride]
    )


@pytest.mark.parametrize("start_position", [0, 3, 71, 75, 81, 200])
def test_ring_addresses_list_each_window_in_position_order_on_distinct_rows(
    start_position: int,
) -> None:
    ring_frames, retained_frames, new_frames = 81, 71, 10
    slot_index = torch.tensor([2, 5])
    addresses = ring_addresses(
        slot_index,
        torch.tensor([start_position, 0]),
        new_frames,
        ring_frames,
        retained_frames,
    )

    for row, start in enumerate((start_position, 0)):
        first = max(0, start - retained_frames)
        positions = list(range(first, start + new_frames))
        expected = [
            int(slot_index[row]) * ring_frames + position % ring_frames
            for position in positions
        ]
        assert int(addresses.window_lengths[row]) == len(positions)
        window = addresses.window_rows[row, : len(positions)].tolist()
        assert window == expected
        assert len(set(window)) == len(window)
        assert (
            addresses.write_rows[row * new_frames : (row + 1) * new_frames].tolist()
            == expected[-new_frames:]
        )
        assert addresses.frame_positions[
            row * new_frames : (row + 1) * new_frames
        ].tolist() == list(range(start, start + new_frames))


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the exchange kernels run on CUDA"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_channels_first", [False, True])
@pytest.mark.parametrize("is_cold", [False, True])
@pytest.mark.parametrize(
    "history_samples,new_samples,tail_samples",
    [
        (2, 1, 0),
        (6, 2, 0),
        (6, 4, 0),
        (6, 40, 2),
        (18, 320, 0),
        (54, 32, 2),
        (54, 320, 8),
    ],
)
def test_history_exchange_kernel_equals_the_reference(
    dtype: torch.dtype,
    is_channels_first: bool,
    is_cold: bool,
    history_samples: int,
    new_samples: int,
    tail_samples: int,
) -> None:
    torch.manual_seed(history_samples * new_samples)
    channels = 200
    slot_index = torch.tensor([4, 1, 6], device="cuda")
    history_pool = torch.randn(8, history_samples, channels, device="cuda").to(dtype)
    samples = history_samples + new_samples + tail_samples
    if is_channels_first:
        padded = (
            torch.randn(3, channels, samples, device="cuda").to(dtype).transpose(1, 2)
        )
    else:
        padded = torch.randn(3, samples, channels, device="cuda").to(dtype)
    expected_padded, expected_pool = padded.clone(), history_pool.clone()

    history_exchange_reference(
        expected_padded, expected_pool, slot_index, new_samples, is_cold
    )
    launch_history_exchange(padded, history_pool, slot_index, new_samples, is_cold)

    assert torch.equal(padded, expected_padded)
    assert torch.equal(history_pool, expected_pool)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the exchange kernels run on CUDA"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_channels_first", [False, True])
@pytest.mark.parametrize("is_cold", [False, True])
@pytest.mark.parametrize(
    "overlap_samples,emitted_samples", [(3, 3), (4, 640), (5, 160), (8, 32), (8, 320)]
)
def test_overlap_exchange_kernel_equals_the_reference(
    dtype: torch.dtype,
    is_channels_first: bool,
    is_cold: bool,
    overlap_samples: int,
    emitted_samples: int,
) -> None:
    torch.manual_seed(overlap_samples * emitted_samples)
    channels = 200
    slot_index = torch.tensor([4, 1, 6], device="cuda")
    overlap_pool = torch.randn(8, overlap_samples, channels, device="cuda").to(dtype)
    samples = emitted_samples + overlap_samples
    if is_channels_first:
        output = (
            torch.randn(3, channels, samples, device="cuda").to(dtype).transpose(1, 2)
        )
    else:
        output = torch.randn(3, samples, channels, device="cuda").to(dtype)
    expected_output, expected_pool = output.clone(), overlap_pool.clone()

    overlap_exchange_reference(
        expected_output, expected_pool, slot_index, emitted_samples, is_cold
    )
    launch_overlap_exchange(output, overlap_pool, slot_index, emitted_samples, is_cold)

    assert torch.equal(output, expected_output)
    assert torch.equal(overlap_pool, expected_pool)
