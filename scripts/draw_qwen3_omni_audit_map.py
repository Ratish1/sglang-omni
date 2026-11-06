# SPDX-License-Identifier: Apache-2.0
"""Draw source-backed model paths and the enumerated modality routing matrix."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

CANVAS_WIDTH_PIXELS = 2600
CANVAS_HEIGHT_PIXELS = 2800
CANVAS_DPI = 180
PREVIEW_WIDTH_PIXELS = 1800


@dataclass(frozen=True, kw_only=True)
class Node:
    identifier: str
    label: str
    bounds: tuple[float, float, float, float]
    color: str


@dataclass(frozen=True, kw_only=True)
class Connection:
    source: str
    destination: str
    points: tuple[tuple[float, float], ...]


def boundary_contact(point: tuple[float, float], node: Node) -> bool:
    x, y = point
    left, bottom, width, height = node.bounds
    return (
        left <= x <= left + width
        and bottom <= y <= bottom + height
        and (
            math.isclose(x, left)
            or math.isclose(x, left + width)
            or math.isclose(y, bottom)
            or math.isclose(y, bottom + height)
        )
    )


def segment_enters_node(
    start: tuple[float, float],
    end: tuple[float, float],
    node: Node,
) -> bool:
    left, bottom, width, height = node.bounds
    clearance = 0.15
    left -= clearance
    bottom -= clearance
    right = left + width + 2 * clearance
    top = bottom + height + 2 * clearance
    if math.isclose(start[1], end[1]):
        return bottom < start[1] < top and max(min(start[0], end[0]), left) < min(
            max(start[0], end[0]), right
        )
    elif math.isclose(start[0], end[0]):
        return left < start[0] < right and max(min(start[1], end[1]), bottom) < min(
            max(start[1], end[1]), top
        )
    else:
        raise ValueError("Every connector must be an explicit orthogonal route")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = json.loads(arguments.routes.read_text())
    dimensions = report["model_dimensions"]
    if report["case_count"] != len(report["cases"]):
        raise ValueError("Incomplete route inventory")
    else:
        pass
    stage_names = {
        stage["name"] for stage in report["stage_graphs"]["speech_colocated"]
    }
    if stage_names != {
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    }:
        raise ValueError("The source stage graph changed")
    else:
        pass
    nodes = [
        Node(identifier="text", label="Text", bounds=(1, 88, 10, 5), color="#e8f1ff"),
        Node(identifier="image", label="Image", bounds=(1, 81, 10, 5), color="#e8f1ff"),
        Node(identifier="video", label="Video", bounds=(1, 74, 10, 5), color="#e8f1ff"),
        Node(identifier="audio", label="Audio", bounds=(1, 67, 10, 5), color="#e8f1ff"),
        Node(
            identifier="preprocessing",
            label="Preprocessing\n\nchat template\nmedia decode\nfeatures + grids\ntokens + masks",
            bounds=(17, 67, 13, 26),
            color="#f1f4f8",
        ),
        Node(
            identifier="image_encoder",
            label="Vision encoder\nimage + video",
            bounds=(37, 85, 14, 8),
            color="#eeeaff",
        ),
        Node(
            identifier="tokens",
            label="Text token IDs",
            bounds=(37, 77, 14, 5),
            color="#e8f1ff",
        ),
        Node(
            identifier="audio_encoder",
            label="Audio encoder\nexplicit + video audio",
            bounds=(37, 67, 14, 8),
            color="#eeeaff",
        ),
        Node(
            identifier="thinker",
            label="Thinker\n\nmedia scatter\nM-RoPE\nvisual deepstack\ntext generation\nhidden-state handoff",
            bounds=(59, 67, 13, 26),
            color="#fff1d5",
        ),
        Node(
            identifier="decode",
            label="Decode\ntext output",
            bounds=(79, 86, 18, 7),
            color="#e6f4ea",
        ),
        Node(
            identifier="talker_ar",
            label="Talker\ntext + hidden projections\nprefix cache + codec loop",
            bounds=(79, 70, 18, 12),
            color="#e6f4ea",
        ),
        Node(
            identifier="code2wav",
            label="Code2Wav\n16-group codec frames",
            bounds=(79, 60, 18, 7),
            color="#e6f4ea",
        ),
        Node(
            identifier="waveform",
            label="Audio output\n24 kHz PCM",
            bounds=(59, 60, 13, 6),
            color="#e6f4ea",
        ),
        Node(
            identifier="prefix",
            label="Projected prompt\n+ prefix KV cache",
            bounds=(2, 20, 18, 8),
            color="#fff1d5",
        ),
        Node(
            identifier="backbone",
            label=f"Talker step\n{dimensions['talker_layers']} layers\ntop-{dimensions['talker_selected_experts']} + shared expert",
            bounds=(27, 20, 18, 8),
            color="#e6f4ea",
        ),
        Node(
            identifier="primary",
            label="Primary codec group\nsampled from logits",
            bounds=(53, 20, 18, 8),
            color="#e6f4ea",
        ),
        Node(
            identifier="residual",
            label=f"Residual predictor\n{dimensions['predictor_layers']} dense layers\n{dimensions['codec_groups'] - 1} more groups",
            bounds=(80, 20, 18, 8),
            color="#e6f4ea",
        ),
        Node(
            identifier="feedback",
            label=f"Sum {dimensions['codec_groups']} codec embeddings\n+ next text row (or padding)",
            bounds=(51, 6, 23, 8),
            color="#e8f1ff",
        ),
    ]
    connections = [
        Connection(
            source=name,
            destination="preprocessing",
            points=((11, height), (17, height)),
        )
        for name, height in (
            ("text", 90.5),
            ("image", 83.5),
            ("video", 76.5),
            ("audio", 69.5),
        )
    ]
    connections.extend(
        [
            Connection(
                source="preprocessing",
                destination="image_encoder",
                points=((30, 89), (37, 89)),
            ),
            Connection(
                source="preprocessing",
                destination="tokens",
                points=((30, 79.5), (37, 79.5)),
            ),
            Connection(
                source="preprocessing",
                destination="audio_encoder",
                points=((30, 71), (37, 71)),
            ),
            Connection(
                source="image_encoder",
                destination="thinker",
                points=((51, 89), (59, 89)),
            ),
            Connection(
                source="tokens", destination="thinker", points=((51, 79.5), (59, 79.5))
            ),
            Connection(
                source="audio_encoder",
                destination="thinker",
                points=((51, 71), (59, 71)),
            ),
            Connection(
                source="thinker", destination="decode", points=((72, 89.5), (79, 89.5))
            ),
            Connection(
                source="thinker", destination="talker_ar", points=((72, 76), (79, 76))
            ),
            Connection(
                source="talker_ar", destination="code2wav", points=((88, 70), (88, 67))
            ),
            Connection(
                source="code2wav",
                destination="waveform",
                points=((79, 63.5), (72, 63.5)),
            ),
            Connection(
                source="prefix", destination="backbone", points=((20, 24), (27, 24))
            ),
            Connection(
                source="backbone", destination="primary", points=((45, 24), (53, 24))
            ),
            Connection(
                source="primary", destination="residual", points=((71, 24), (80, 24))
            ),
            Connection(
                source="residual",
                destination="feedback",
                points=((89, 20), (89, 10), (74, 10)),
            ),
            Connection(
                source="feedback",
                destination="backbone",
                points=((51, 10), (36, 10), (36, 20)),
            ),
        ]
    )
    node_map = {node.identifier: node for node in nodes}
    for connection in connections:
        if not boundary_contact(
            connection.points[0], node_map[connection.source]
        ) or not boundary_contact(
            connection.points[-1], node_map[connection.destination]
        ):
            raise ValueError(f"Incorrect connector endpoints: {connection}")
        else:
            pass
        for start, end in zip(connection.points, connection.points[1:]):
            for node in nodes:
                if node.identifier not in (
                    connection.source,
                    connection.destination,
                ) and segment_enters_node(start, end, node):
                    raise ValueError(
                        f"Connector crosses unrelated node: {connection} / {node.identifier}"
                    )
                else:
                    pass
    figure = Figure(
        figsize=(CANVAS_WIDTH_PIXELS / CANVAS_DPI, CANVAS_HEIGHT_PIXELS / CANVAS_DPI),
        dpi=CANVAS_DPI,
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.035, 0.02, 0.93, 0.96), xlim=(0, 100), ylim=(0, 100))
    axes.axis("off")
    labels = []
    for node in nodes:
        left, bottom, width, height = node.bounds
        axes.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor=node.color,
                edgecolor="#243449",
                linewidth=1.1,
                zorder=2,
            )
        )
        label = axes.text(
            left + width / 2,
            bottom + height / 2,
            node.label,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#172535",
            zorder=3,
        )
        labels.append((label, node))
    for connection in connections:
        for start, end in zip(connection.points, connection.points[1:-1]):
            axes.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                color="#42566e",
                linewidth=1.2,
                zorder=1,
            )
        axes.add_patch(
            FancyArrowPatch(
                connection.points[-2],
                connection.points[-1],
                arrowstyle="-|>",
                mutation_scale=13,
                shrinkA=0,
                shrinkB=0,
                linewidth=1.2,
                color="#42566e",
                zorder=1,
            )
        )
    axes.text(
        0,
        98,
        "Qwen3-Omni model paths and audit boundaries",
        fontsize=20,
        fontweight="bold",
        color="#172535",
    )
    axes.text(
        0,
        95,
        "Source-derived speech graph. Arrows show data dependencies, not execution timing.",
        fontsize=11,
        color="#42566e",
    )
    axes.text(
        0,
        64.5,
        "Talker registers early and receives encoder geometry, not encoder vectors.",
        fontsize=9.5,
        color="#42566e",
    )
    axes.text(
        0,
        62.5,
        "Session: PCM16 + camera frames → dialog → preprocessing → vision → audio",
        fontsize=9.5,
        color="#42566e",
    )
    axes.text(
        0,
        60.5,
        "→ Thinker → Talker → Code2Wav. Controls and retained KV differ from HTTP.",
        fontsize=9.5,
        color="#42566e",
    )
    axes.text(
        0,
        59,
        "All non-empty content combinations",
        fontsize=11,
        fontweight="bold",
        color="#172535",
    )
    combinations_seen = []
    for case in report["cases"]:
        modalities = tuple(case["input_modalities"])
        if modalities not in combinations_seen:
            combinations_seen.append(modalities)
        else:
            pass
    if len(combinations_seen) != 15:
        raise ValueError("Expected all fifteen subsets of four input modalities")
    else:
        pass
    axes.text(0, 56.5, "Content", fontsize=9.5, fontweight="bold")
    axes.text(46, 56.5, "Vision encoder", fontsize=9.5, fontweight="bold")
    axes.text(70, 56.5, "Audio encoder", fontsize=9.5, fontweight="bold")
    for row, modalities in enumerate(combinations_seen):
        y = 54.5 - row * 1.25
        matching = [
            case
            for case in report["cases"]
            if tuple(case["input_modalities"]) == modalities
        ]
        vision_states = {"image_encoder" in case["join_sources"] for case in matching}
        audio_states = {"audio_encoder" in case["join_sources"] for case in matching}
        vision_label = "yes" if vision_states == {True} else "no"
        audio_label = (
            "yes"
            if audio_states == {True}
            else (
                "conditional on video audio" if audio_states == {False, True} else "no"
            )
        )
        axes.text(0, y, " + ".join(modalities), fontsize=9.5)
        axes.text(46, y, vision_label, fontsize=9.5)
        axes.text(70, y, audio_label, fontsize=9.5)
    public_default = ", ".join(report["chat_api_default_output_modalities"])
    axes.text(
        0,
        34,
        f"Output branches: text → Decode. Audio requested → Talker + Code2Wav, with Decode still active. Chat API default: {public_default}.",
        fontsize=9.5,
        color="#42566e",
    )
    axes.text(
        0,
        32,
        f"{report['case_count']} internal route/join cases measured across output selections, streaming flags and video-audio presence. Not task-quality coverage.",
        fontsize=9.5,
        color="#42566e",
    )
    axes.text(
        0,
        30,
        "Autoregressive speech recurrence",
        fontsize=11,
        fontweight="bold",
        color="#172535",
    )
    axes.text(
        0,
        2,
        "Text profile joins at mm_aggregate. Routing, numerics, generation quality, sessions and H100 performance are separate audit scopes.",
        fontsize=9.5,
        color="#42566e",
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    segments = [
        (connection, start, end)
        for connection in connections
        for start, end in zip(connection.points, connection.points[1:])
    ]
    for index, (connection, start, end) in enumerate(segments):
        for other, other_start, other_end in segments[index + 1 :]:
            if connection is other:
                continue
            else:
                first_horizontal = math.isclose(start[1], end[1])
                second_horizontal = math.isclose(other_start[1], other_end[1])
            if first_horizontal == second_horizontal:
                fixed_axis = 1 if first_horizontal else 0
                moving_axis = 1 - fixed_axis
                intersects = math.isclose(
                    start[fixed_axis], other_start[fixed_axis]
                ) and max(
                    min(start[moving_axis], end[moving_axis]),
                    min(other_start[moving_axis], other_end[moving_axis]),
                ) <= min(
                    max(start[moving_axis], end[moving_axis]),
                    max(other_start[moving_axis], other_end[moving_axis]),
                )
            else:
                horizontal_start, horizontal_end = (
                    (start, end) if first_horizontal else (other_start, other_end)
                )
                vertical_start, vertical_end = (
                    (other_start, other_end) if first_horizontal else (start, end)
                )
                intersects = min(
                    horizontal_start[0], horizontal_end[0]
                ) <= vertical_start[0] <= max(
                    horizontal_start[0], horizontal_end[0]
                ) and min(
                    vertical_start[1], vertical_end[1]
                ) <= horizontal_start[
                    1
                ] <= max(
                    vertical_start[1], vertical_end[1]
                )
            if intersects:
                raise ValueError(
                    f"Undeclared connector intersection: {connection} / {other}"
                )
            else:
                pass
    for label, node in labels:
        extent = label.get_window_extent(renderer)
        left, bottom, width, height = node.bounds
        lower = axes.transData.transform((left, bottom))
        upper = axes.transData.transform((left + width, bottom + height))
        if (
            extent.x0 < lower[0]
            or extent.x1 > upper[0]
            or extent.y0 < lower[1]
            or extent.y1 > upper[1]
        ):
            raise ValueError(f"Label exceeds its node: {node.identifier}")
        else:
            pass
    for index, label in enumerate(axes.texts):
        extent = label.get_window_extent(renderer)
        if extent.height * PREVIEW_WIDTH_PIXELS / CANVAS_WIDTH_PIXELS < 12:
            raise ValueError(
                f"Text is too small at destination width: {label.get_text()}"
            )
        else:
            pass
        for other in axes.texts[index + 1 :]:
            if extent.overlaps(other.get_window_extent(renderer)):
                raise ValueError(
                    f"Labels overlap: {label.get_text()} / {other.get_text()}"
                )
            else:
                pass
        for connection, start, end in segments:
            start_pixels = axes.transData.transform(start)
            end_pixels = axes.transData.transform(end)
            if math.isclose(start[1], end[1]):
                intersects = extent.y0 - 3 < start_pixels[1] < extent.y1 + 3 and max(
                    min(start_pixels[0], end_pixels[0]), extent.x0 - 3
                ) < min(max(start_pixels[0], end_pixels[0]), extent.x1 + 3)
            else:
                intersects = extent.x0 - 3 < start_pixels[0] < extent.x1 + 3 and max(
                    min(start_pixels[1], end_pixels[1]), extent.y0 - 3
                ) < min(max(start_pixels[1], end_pixels[1]), extent.y1 + 3)
            if intersects:
                raise ValueError(
                    f"Connector intersects a label: {connection} / {label.get_text()}"
                )
            else:
                pass
        for node in nodes:
            if any(
                label is node_label and node.identifier == owner.identifier
                for node_label, owner in labels
            ):
                continue
            else:
                left, bottom, width, height = node.bounds
                lower = axes.transData.transform((left, bottom))
                upper = axes.transData.transform((left + width, bottom + height))
            if (
                extent.x0 < upper[0]
                and extent.x1 > lower[0]
                and extent.y0 < upper[1]
                and extent.y1 > lower[1]
            ):
                raise ValueError(
                    f"Label intersects unrelated node: {label.get_text()} / {node.identifier}"
                )
            else:
                pass
        if (
            extent.x0 < 0
            or extent.y0 < 0
            or extent.x1 > CANVAS_WIDTH_PIXELS
            or extent.y1 > CANVAS_HEIGHT_PIXELS
        ):
            raise ValueError(f"Label exceeds canvas: {label.get_text()}")
        else:
            pass
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        arguments.output,
        dpi=CANVAS_DPI,
        facecolor="white",
        transparent=False,
        bbox_inches=None,
    )
    figure.clear()
    with Image.open(arguments.output) as rendered:
        if rendered.size != (
            CANVAS_WIDTH_PIXELS,
            CANVAS_HEIGHT_PIXELS,
        ) or rendered.convert("RGBA").getextrema()[3] != (255, 255):
            raise ValueError("Incorrect image dimensions or transparency")
        else:
            pass
    print(arguments.output)


if __name__ == "__main__":
    main()
