"""Where the image A/B's answers moved: every MMMU item split by what the fix changed in its model input.

The two arms differ only in what reaches the HF image processor, so an item whose pixel_values are bit identical across
the arms has the same input on both, and any flip on it is the band of repeat runs. Each item's grid delta is checked
against the prompt_tokens both arms' servers recorded.

usage: PYTHONPATH=IMAGE_TREE python3 mmmu_pixel_split.py QUEUE_DIR OUT_JSON
"""

import base64
import json
import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

from benchmarks.dataset.mmmu import load_mmmu_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import reference_prepared_image
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MMMU_TTS_PROMPT

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
ARMS = ("main_a", "main_b", "image_a", "image_b")


def decoded(uri: str) -> Image.Image:
    image = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    image.load()
    return image


def sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    else:
        tail = sum(math.comb(total, k) for k in range(0, min(wins, losses) + 1)) / 2**total
        return min(1.0, 2 * tail)


def split_row(label: str, items: list[dict]) -> dict:
    main = sum(item["main"] for item in items) / 2
    image = sum(item["image"] for item in items) / 2
    wins = sum(item["image"] > item["main"] for item in items)
    losses = sum(item["image"] < item["main"] for item in items)
    deltas = np.array([(item["image"] - item["main"]) / 2 for item in items], dtype=np.float64)
    half_width = 1.96 * deltas.std(ddof=1) * math.sqrt(len(items)) if len(items) > 1 else 0.0
    row = {
        "split": label,
        "items": len(items),
        "main_correct": main,
        "image_correct": image,
        "delta": image - main,
        "delta_95": [image - main - half_width, image - main + half_width],
        "better_image": wins,
        "better_main": losses,
        "sign_p": sign_test(wins, losses),
        "main_boots_disagree": sum(item["main"] == 1 for item in items),
        "image_boots_disagree": sum(item["image"] == 1 for item in items),
    }
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    root = Path(sys.argv[1])
    arms = {
        name: {row["sample_id"]: row for row in json.loads((root / name / "c32" / "mmmu" / "mmmu_results.json").read_text())["per_sample"]}
        for name in ARMS
    }
    image_processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL).image_processor
    merge = image_processor.merge_size**2
    items = []
    token_mismatches = []
    for sample in load_mmmu_samples(instruction_override=MMMU_TTS_PROMPT):
        images = [decoded(uri) for uri in sample.image_data_uris]
        main_media = image_processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
        image_media = image_processor(images=[reference_prepared_image(image) for image in images], return_tensors="pt")
        main_pixels = main_media["pixel_values"].float()
        image_pixels = image_media["pixel_values"].float()
        same_grid = bool((main_media["image_grid_thw"] == image_media["image_grid_thw"]).all())
        same_pixels = same_grid and bool((main_pixels == image_pixels).all())
        match = None
        if same_grid:
            match = 1 - float((image_pixels - main_pixels).norm() / main_pixels.norm())
        else:
            pass
        grid_delta = int((image_media["image_grid_thw"].prod(dim=1) - main_media["image_grid_thw"].prod(dim=1)).sum()) // merge
        token_delta = arms["image_a"][sample.sample_id]["prompt_tokens"] - arms["main_a"][sample.sample_id]["prompt_tokens"]
        if grid_delta != token_delta:
            token_mismatches.append([sample.sample_id, grid_delta, token_delta])
        else:
            pass
        items.append(
            {
                "sample_id": sample.sample_id,
                "images": len(images),
                "sizes": [list(image.size) for image in images],
                "modes": [image.mode for image in images],
                "same_grid": same_grid,
                "same_pixels": same_pixels,
                "pixel_match": match,
                "token_delta": grid_delta,
                "main": int(arms["main_a"][sample.sample_id]["is_correct"]) + int(arms["main_b"][sample.sample_id]["is_correct"]),
                "image": int(arms["image_a"][sample.sample_id]["is_correct"]) + int(arms["image_b"][sample.sample_id]["is_correct"]),
                "capped": [int(arms[name][sample.sample_id]["completion_tokens"] >= 2048) for name in ARMS],
                "unparseable": [int(arms[name][sample.sample_id]["is_unparseable"]) for name in ARMS],
            }
        )
    print(f"items {len(items)}, grid delta against recorded prompt tokens: {len(token_mismatches)} mismatches {token_mismatches[:5]}", flush=True)
    resized = sorted(item["pixel_match"] for item in items if item["same_grid"] and not item["same_pixels"])
    print(f"pixel match on same-grid changed items: worst {resized[0]:.5f} median {resized[len(resized) // 2]:.5f} best {resized[-1]:.5f}", flush=True)
    rows = [
        split_row("all", items),
        split_row("pixels bit identical", [item for item in items if item["same_pixels"]]),
        split_row("same grid, pixels differ", [item for item in items if item["same_grid"] and not item["same_pixels"]]),
        split_row("grid differs", [item for item in items if not item["same_grid"]]),
        split_row("grid differs, fewer tokens", [item for item in items if item["token_delta"] < 0]),
        split_row("grid differs, more tokens", [item for item in items if item["token_delta"] > 0]),
        split_row("grid differs, same tokens", [item for item in items if not item["same_grid"] and item["token_delta"] == 0]),
    ]
    Path(sys.argv[2]).write_text(json.dumps({"rows": rows, "token_mismatches": token_mismatches, "items": items}))


if __name__ == "__main__":
    main()
