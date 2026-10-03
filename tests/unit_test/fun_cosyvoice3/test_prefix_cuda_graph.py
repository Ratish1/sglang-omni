# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import (
    DEFAULT_PREFIX_CUDA_GRAPH_ENVELOPES,
    MAX_SLACK_FRAMES,
    PREFIX_CUDA_GRAPH_CHUNK_FRAMES,
    PrefixCudaGraphEnvelope,
    route_prefix_cuda_graph_envelope,
)


def test_default_prefix_cuda_graph_envelopes_are_frozen_and_valid() -> None:
    assert len(DEFAULT_PREFIX_CUDA_GRAPH_ENVELOPES) == 17
    names = [envelope.name for envelope in DEFAULT_PREFIX_CUDA_GRAPH_ENVELOPES]
    assert len(set(names)) == len(names)
    assert names == [
        "B1-N300-M300-E3072",
        "B1-N450-M450-E512",
        "B3-N400-M200-E1536",
        "B4-N900-M400-E2048",
        "B2-N500-M350-E2048",
        "B3-N650-M350-E3072",
        "B3-N1100-M400-E512",
        "B1-N100-M100-E512",
        "B2-N200-M100-E512",
        "B3-N500-M300-E1024",
        "B4-N1200-M400-E1024",
        "B3-N850-M350-E1024",
        "B5-N1150-M350-E512",
        "B8-N1700-M300-E1536",
        "B2-N600-M300-E512",
        "B4-N700-M300-E1024",
        "B6-N1450-M400-E2048",
    ]
    for envelope in DEFAULT_PREFIX_CUDA_GRAPH_ENVELOPES:
        assert envelope.batch_size > 0
        assert envelope.new_frame_count > 0
        assert envelope.max_new_frame_count > 0
        assert envelope.max_total_frame_count > 0
        assert len(envelope.capture_row_new_frames) == envelope.batch_size
        assert sum(envelope.capture_row_new_frames) == envelope.new_frame_count
        assert max(envelope.capture_row_new_frames) <= envelope.max_new_frame_count
        assert max(envelope.capture_row_new_frames) <= envelope.max_total_frame_count
        assert all(
            frame_count > 0 and frame_count % PREFIX_CUDA_GRAPH_CHUNK_FRAMES == 0
            for frame_count in envelope.capture_row_new_frames
        )
        for qualified_total in envelope.qualified_new_frame_totals:
            assert qualified_total > 0
            assert qualified_total % PREFIX_CUDA_GRAPH_CHUNK_FRAMES == 0
            slack_frame_count = envelope.new_frame_count - qualified_total
            assert 0 <= slack_frame_count <= MAX_SLACK_FRAMES
            assert slack_frame_count % PREFIX_CUDA_GRAPH_CHUNK_FRAMES == 0


def test_prefix_cuda_graph_routes_smallest_qualified_envelope() -> None:
    envelopes = (
        PrefixCudaGraphEnvelope("small", 1, 100, 100, 512, (50, 100), (100,)),
        PrefixCudaGraphEnvelope("large", 1, 150, 150, 1024, (100, 150), (150,)),
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=1,
            total_new_frame_count=50,
            max_new_frame_count=50,
            max_total_frame_count=512,
            envelopes=envelopes,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=2,
            total_new_frame_count=100,
            max_new_frame_count=100,
            max_total_frame_count=512,
            envelopes=envelopes,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=1,
            total_new_frame_count=150,
            max_new_frame_count=151,
            max_total_frame_count=1024,
            envelopes=envelopes,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=1,
            total_new_frame_count=150,
            max_new_frame_count=150,
            max_total_frame_count=1025,
            envelopes=envelopes,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=1,
            total_new_frame_count=75,
            max_new_frame_count=75,
            max_total_frame_count=512,
            envelopes=envelopes,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=1,
            total_new_frame_count=100,
            max_new_frame_count=120,
            max_total_frame_count=600,
            envelopes=envelopes,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            batch_size=1,
            total_new_frame_count=100,
            max_new_frame_count=100,
            max_total_frame_count=512,
            envelopes=envelopes,
        )
        == envelopes[0]
    )


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        (
            {"enable_dit_torch_compile": False, "flow_prefix_cache_gb": 24.0},
            ValueError,
            "enable_dit_torch_compile",
        ),
        (
            {"enable_dit_torch_compile": True, "flow_prefix_cache_gb": 0.0},
            ValueError,
            "flow_prefix_cache_gb",
        ),
        (
            {"enable_dit_torch_compile": True, "flow_prefix_cache_gb": 24.0},
            RuntimeError,
            "available CUDA",
        ),
    ],
)
def test_prefix_cuda_graph_explicit_enable_requires_cuda_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, bool | float],
    exception: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )

    with pytest.raises(exception, match=message):
        stages.create_vocoder_executor(
            "model",
            device="cpu",
            enable_flow_prefix_cuda_graph=True,
            **kwargs,
        )
