"""Does the card loader's second resample blur MMMU's images, and do the answers lean where it blurs most?

HF's processor resizes to multiples of patch 16 times merge 2 = 32. The card's process_mm_info defaults to
image_patch_size=14, so it first resizes to multiples of 28, and the processor then resamples that. Arms on every item:
main (the processor alone), branch (reference_prepared_image, then the processor), card14 (process_mm_info's fetch_image
at its default, then the processor; the branch must equal it), card16 (fetch_image at the model's own patch 16, then the
processor). Sharpness is the mean squared difference between neighbouring pixels inside each 16-pixel patch of the
pixel_values the model reads, so it needs no patch reassembly and compares grids of different sizes.

usage: PYTHONPATH=IMAGE_TREE python3 mmmu_resample_sharpness.py QUEUE_DIR OUT_JSON
"""

import base64
import json
import math
import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from qwen_omni_utils.v2_5.vision_process import fetch_image
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


def sharpness(pixel_values: torch.Tensor, patch: int, temporal: int) -> float:
    patches = pixel_values.float().reshape(-1, 3, temporal, patch, patch)[:, :, 0]
    across = patches[..., 1:] - patches[..., :-1]
    down = patches[..., 1:, :] - patches[..., :-1, :]
    return 0.5 * (across.pow(2).mean().item() + down.pow(2).mean().item())


def match(left: dict, right: dict) -> float | None:
    if torch.equal(left["image_grid_thw"], right["image_grid_thw"]):
        return 1 - float((left["pixel_values"].float() - right["pixel_values"].float()).norm() / right["pixel_values"].float().norm())
    else:
        return None


def sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    else:
        tail = sum(math.comb(total, k) for k in range(0, min(wins, losses) + 1)) / 2**total
        return min(1.0, 2 * tail)


def quantile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(fraction * len(values)))]


def leaning(label: str, items: list[dict]) -> dict:
    wins = sum(item["image"] > item["main"] for item in items)
    losses = sum(item["image"] < item["main"] for item in items)
    row = {
        "split": label,
        "items": len(items),
        "sharpness_ratio_median": quantile([item["sharpness_branch"] / item["sharpness_main"] for item in items], 0.5),
        "main_correct": sum(item["main"] for item in items) / 2,
        "image_correct": sum(item["image"] for item in items) / 2,
        "better_image": wins,
        "better_main": losses,
        "sign_p": sign_test(wins, losses),
    }
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    root = Path(sys.argv[1])
    arms = {
        name: {row["sample_id"]: bool(row["is_correct"]) for row in json.loads((root / name / "c32" / "mmmu" / "mmmu_results.json").read_text())["per_sample"]}
        for name in ARMS
    }
    image_processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL).image_processor
    patch = image_processor.patch_size
    temporal = image_processor.temporal_patch_size
    items = []
    for sample in load_mmmu_samples(instruction_override=MMMU_TTS_PROMPT):
        images = [decoded(uri) for uri in sample.image_data_uris]
        media = {
            "main": image_processor(images=[image.convert("RGB") for image in images], return_tensors="pt"),
            "branch": image_processor(images=[reference_prepared_image(image) for image in images], return_tensors="pt"),
            "card14": image_processor(images=[fetch_image({"image": image}) for image in images], return_tensors="pt"),
            "card16": image_processor(images=[fetch_image({"image": image}, image_patch_size=patch) for image in images], return_tensors="pt"),
        }
        items.append(
            {
                "sample_id": sample.sample_id,
                "same_grid": bool(torch.equal(media["main"]["image_grid_thw"], media["branch"]["image_grid_thw"])),
                "branch_equals_card14": bool(
                    torch.equal(media["branch"]["image_grid_thw"], media["card14"]["image_grid_thw"])
                    and torch.equal(media["branch"]["pixel_values"], media["card14"]["pixel_values"])
                ),
                "card16_same_grid_as_main": bool(torch.equal(media["main"]["image_grid_thw"], media["card16"]["image_grid_thw"])),
                "card16_match_main": match(media["card16"], media["main"]),
                "branch_match_main": match(media["branch"], media["main"]),
                "sharpness_main": sharpness(media["main"]["pixel_values"], patch, temporal),
                "sharpness_branch": sharpness(media["branch"]["pixel_values"], patch, temporal),
                "sharpness_card16": sharpness(media["card16"]["pixel_values"], patch, temporal),
                "main": int(arms["main_a"][sample.sample_id]) + int(arms["main_b"][sample.sample_id]),
                "image": int(arms["image_a"][sample.sample_id]) + int(arms["image_b"][sample.sample_id]),
            }
        )
    usable = [item for item in items if item["sharpness_main"] > 0]
    ratios = [item["sharpness_branch"] / item["sharpness_main"] for item in usable]
    card16_ratios = [item["sharpness_card16"] / item["sharpness_main"] for item in usable]
    card16_matches = [item["card16_match_main"] for item in items if item["card16_match_main"] is not None]
    summary = {
        "items": len(items),
        "branch_equals_card14": sum(item["branch_equals_card14"] for item in items),
        "card16_same_grid_as_main": sum(item["card16_same_grid_as_main"] for item in items),
        "card16_match_main_worst_median": [min(card16_matches), quantile(card16_matches, 0.5)],
        "branch_over_main_sharpness_p10_median_p90": [quantile(ratios, 0.1), quantile(ratios, 0.5), quantile(ratios, 0.9)],
        "branch_sharper_than_main_items": sum(ratio > 1 for ratio in ratios),
        "card16_over_main_sharpness_p10_median_p90": [quantile(card16_ratios, 0.1), quantile(card16_ratios, 0.5), quantile(card16_ratios, 0.9)],
    }
    print(json.dumps(summary), flush=True)
    changed = sorted((item for item in usable if item["same_grid"] and item["branch_match_main"] < 1), key=lambda item: item["sharpness_branch"] / item["sharpness_main"])
    third = len(changed) // 3
    rows = [
        leaning("same grid, most blurred third", changed[:third]),
        leaning("same grid, middle third", changed[third : 2 * third]),
        leaning("same grid, least blurred third", changed[2 * third :]),
    ]
    Path(sys.argv[2]).write_text(json.dumps({"summary": summary, "rows": rows, "items": items}))


if __name__ == "__main__":
    main()
