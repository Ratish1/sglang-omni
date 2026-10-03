# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import (
    PrefixCudaGraphEnvelope,
    prefix_cuda_graph_envelopes_from_capture_shapes,
    route_prefix_cuda_graph_envelope,
)


def test_default_prefix_cuda_graph_capture_shapes_are_frozen_and_valid() -> None:
    expected_capture_shapes = (
        (1, 300, 300, 3072, (300,)),
        (1, 450, 450, 512, (450,)),
        (3, 400, 200, 1536, (200, 100, 100)),
        (4, 900, 400, 2048, (400, 200, 150, 150)),
        (2, 500, 350, 2048, (350, 150)),
        (3, 650, 350, 3072, (350, 150, 150)),
        (3, 1100, 400, 512, (400, 350, 350)),
        (1, 100, 100, 512, (100,)),
        (2, 200, 100, 512, (100, 100)),
        (3, 500, 300, 1024, (300, 100, 100)),
        (4, 1200, 400, 1024, (400, 300, 250, 250)),
        (3, 850, 350, 1024, (350, 250, 250)),
        (5, 1150, 350, 512, (350, 200, 200, 200, 200)),
        (8, 1700, 300, 1536, (300, 300, 250, 200, 200, 150, 150, 150)),
        (2, 600, 300, 512, (300, 300)),
        (4, 700, 300, 1024, (300, 150, 150, 100)),
        (6, 1450, 400, 2048, (400, 250, 200, 200, 200, 200)),
    )
    assert FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES == (
        expected_capture_shapes
    )
    stages.verify_prefix_cuda_graph_capture_shapes(
        FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES
    )
    envelopes = prefix_cuda_graph_envelopes_from_capture_shapes(
        FUN_COSYVOICE3_DEFAULT_PREFIX_CUDA_GRAPH_CAPTURE_SHAPES
    )
    assert len(envelopes) == 17
    names = [envelope.name for envelope in envelopes]
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
    assert (
        tuple(
            (
                envelope.batch_size,
                envelope.new_frame_count,
                envelope.max_new_frame_count,
                envelope.max_total_frame_count,
                envelope.capture_row_new_frames,
            )
            for envelope in envelopes
        )
        == expected_capture_shapes
    )


@pytest.mark.parametrize(
    ("capture_shapes", "message"),
    [
        (((1, 100, 100, 512, (75,)),), "multiples"),
        (((1, 100, 100, 512, (50,)),), "sum to N"),
        (((2, 200, 100, 512, (100,)),), "equal B"),
    ],
)
def test_verify_prefix_cuda_graph_capture_shapes_rejects_invalid_overrides(
    capture_shapes: tuple[tuple[int, int, int, int, tuple[int, ...]], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        stages.verify_prefix_cuda_graph_capture_shapes(capture_shapes)


def test_prefix_cuda_graph_routes_physical_envelopes() -> None:
    envelopes = (
        PrefixCudaGraphEnvelope("B1-N100-M100-E512", 1, 100, 100, 512, (100,)),
        PrefixCudaGraphEnvelope("B2-N200-M100-E512", 2, 200, 100, 512, (100, 100)),
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=envelopes,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50, 50],
            total_frame_counts=[50, 50],
            envelopes=envelopes,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50, 50],
            total_frame_counts=[50, 50],
            envelopes=(envelopes[0],),
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 50],
            total_frame_counts=[100, 50],
            envelopes=envelopes,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 100],
            total_frame_counts=[100, 100],
            envelopes=envelopes,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=envelopes,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[75],
            total_frame_counts=[75],
            envelopes=envelopes,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N200-M200-E512", 1, 200, 200, 512, (200,)),
            ),
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N125-M100-E512", 1, 125, 100, 512, (125,)),
            ),
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[150],
            total_frame_counts=[150],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N200-M100-E512", 1, 200, 100, 512, (200,)),
            ),
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[600],
            envelopes=envelopes,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 100],
            total_frame_counts=[100, 100],
            envelopes=(
                PrefixCudaGraphEnvelope(
                    "B2-N200-M100-E512", 2, 200, 100, 512, (100, 100)
                ),
                PrefixCudaGraphEnvelope(
                    "B2-N300-M150-E1024", 2, 300, 150, 1024, (150, 150)
                ),
            ),
        ).name
        == "B2-N200-M100-E512"
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
