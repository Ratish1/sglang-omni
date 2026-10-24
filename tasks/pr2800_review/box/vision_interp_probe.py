"""The Qwen3-Omni vision position interpolation in every candidate form, on the vision encoder calls a served run made.

Run on the box from main's tree (cwd and PYTHONPATH the tree):
    python3 vision_interp_probe.py --calls <name>=<dir with grids_*.jsonl> [<name>=<dir> ...]
The directories come from videoamme_pair.py --encoder-grids (encoder_grids/sitecustomize.py): one line per vision
encoder call with its image and video grid rows. Each call's image grids and video grids are one interpolation each,
as the encoder runs them. The served encoder is built (build_visual, the checkpoint's dtype, card 0) for its pos_embed
table, and every interpolation runs in each form:
  legacy      the encoder's legacy_pos_embed_interpolate (Transformers 5.6 arithmetic, lists built on the host)
  loop        SGLang 0.5.22 Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list
  vectorized  SGLang 0.5.22 Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_vectorized
  dispatch    SGLang's choice between the two (vectorized at _VECTORIZED_VL_POS_EMBED_MIN_IMAGES grids or more)
  hf519       Transformers 5.19: get_vision_interpolation_indices_and_weights, the weighted embedding sum in fp32,
              one cast to the table dtype
Truth is the legacy arithmetic on the table upcast to float64. Per run: the calls, grids per call, the calls at or
over SGLang's threshold, then per form the relative L2 to truth (median and largest over interpolations), the
interpolations bit-equal to the loop form, and the summed time of one warmup and three timed sweeps, each
interpolation synchronized before and after.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from sglang.srt.models import qwen3_vl
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.qwen3_omni.components.image_encoder import build_visual

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TIMED_SWEEPS = 3


def load_interpolations(directory: Path) -> tuple[int, list[list[list[int]]]]:
    calls = 0
    interpolations = []
    for path in sorted(directory.glob("grids_*.jsonl")):
        for line in path.read_text().splitlines():
            call = json.loads(line)
            calls += 1
            interpolations.extend(call[key] for key in ("image", "video") if call[key])
    return calls, interpolations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", nargs="+", required=True, help="name=directory")
    args = parser.parse_args()
    path = snapshot_download(MODEL)
    visual = build_visual(path, thinker_cfg=load_thinker_config(path), torch_dtype=None, device="cuda:0").eval()
    device = visual.pos_embed.weight.device
    dtype = visual.pos_embed.weight.dtype
    stub = SimpleNamespace(
        num_grid_per_side=visual.num_grid_per_side,
        spatial_merge_size=visual.spatial_merge_size,
        pos_embed=visual.pos_embed,
        device=device,
        dtype=dtype,
    )
    truth_stub = SimpleNamespace(
        num_grid_per_side=visual.num_grid_per_side,
        config=visual.config,
        pos_embed=torch.nn.Embedding.from_pretrained(visual.pos_embed.weight.detach().double()),
    )
    threshold = qwen3_vl._VECTORIZED_VL_POS_EMBED_MIN_IMAGES  # noqa: leading-underscore  # SGLang's constant
    model = qwen3_vl.Qwen3VLMoeVisionModel
    print(f"table {tuple(visual.pos_embed.weight.shape)} {dtype}, side {visual.num_grid_per_side}, merge "
          f"{visual.spatial_merge_size}, SGLang threshold {threshold}")

    def hf519(grid: list[list[int]]) -> torch.Tensor:
        indices, weights = get_vision_interpolation_indices_and_weights(
            torch.tensor(grid, device=device), num_grid_per_side=visual.num_grid_per_side, mode="bilinear",
            align_corners=True, spatial_merge_size=visual.spatial_merge_size,
        )
        return (visual.pos_embed(indices) * weights[:, :, None]).sum(1).to(dtype)

    forms = {
        "legacy": lambda grid: visual.legacy_pos_embed_interpolate(torch.tensor(grid)),
        "loop": lambda grid: model.fast_pos_embed_interpolate_from_list(stub, grid),
        "vectorized": lambda grid: model.fast_pos_embed_interpolate_vectorized(stub, grid),
        "dispatch": lambda grid: (
            model.fast_pos_embed_interpolate_vectorized(stub, grid)
            if len(grid) >= threshold
            else model.fast_pos_embed_interpolate_from_list(stub, grid)
        ),
        "hf519": hf519,
    }
    with torch.no_grad():
        for entry in args.calls:
            name, directory = entry.split("=", 1)
            calls, interpolations = load_interpolations(Path(directory))
            sizes = [len(grid) for grid in interpolations]
            patches = [sum(t * h * w for t, h, w in grid) for grid in interpolations]
            print(f"== {name}: {calls} calls, {len(interpolations)} interpolations, grids per interpolation "
                  f"median {statistics.median(sizes)} largest {max(sizes)}, at or over {threshold}: "
                  f"{sum(size >= threshold for size in sizes)}; patches median {statistics.median(patches)} "
                  f"largest {max(patches)}", flush=True)
            distances: dict[str, list[float]] = {form: [] for form in forms}
            equal = dict.fromkeys(forms, 0)
            for grid in interpolations:
                truth = type(visual).legacy_pos_embed_interpolate(truth_stub, torch.tensor(grid))
                loop = forms["loop"](grid)
                for form, call in forms.items():
                    output = call(grid)
                    distances[form].append(float((output.double() - truth).norm() / truth.norm()))
                    equal[form] += int(torch.equal(output, loop))
            for form, call in forms.items():
                times = []
                for sweep in range(TIMED_SWEEPS + 1):
                    total = 0.0
                    for grid in interpolations:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        call(grid)
                        torch.cuda.synchronize()
                        total += time.perf_counter() - start
                    if sweep > 0:
                        times.append(total * 1000)
                    else:
                        pass
                print(f"{form:10s} relative L2 to truth median {statistics.median(distances[form]):.3e} largest "
                      f"{max(distances[form]):.3e}; equal to loop {equal[form]} of {len(interpolations)}; time "
                      f"{', '.join(f'{value:.1f}' for value in times)} ms", flush=True)


if __name__ == "__main__":
    main()
