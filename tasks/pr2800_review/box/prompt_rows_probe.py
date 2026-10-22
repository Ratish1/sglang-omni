"""The talker's prompt rows through the served pipeline against HF's thinker, on stage 6's MMSU requests.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/prompt_rows_probe.py --arm <tree> --out /data/pr2800/<run>
Card 0 serves the arm with prompt_rows/sitecustomize.py on PYTHONPATH, which records what the talker's prompt prefill
receives. The first SAMPLES mmsu-ci-2000 requests go one at a time with the CI's talker prompt, text and audio,
greedy, MAX_TOKENS tokens (the talker's prompt prefill comes with the first). Card 1 then runs HF's thinker in fp32
(the truth) and in bf16 (the floor) on each request's own prompt ids with HF's audio features, and reads
hidden_states[accept_hidden_layer] at the multimodal positions. Per request: the positions the talker got against the
prompt's audio tokens, and the relative L2 to fp32 of the served rows and of HF bf16's, with their ratio. Writes
<out>/prompt_rows.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import torch

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REPO_ID = "zhaochenyang20/mmsu-ci-2000"
SAMPLES = 40
MAX_TOKENS = 8
PORT = 26000
MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)


def relative_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    return float((value.double() - truth.double()).norm() / truth.double().norm())


def serve_and_record(arm_tree: Path, out: Path) -> list[dict]:
    dump = out / "dump"
    dump.mkdir(parents=True, exist_ok=True)
    card_env = pair_check.card_env
    pair_check.card_env = lambda card, tree_path: {
        **card_env(card, tree_path),
        "PYTHONPATH": f"{BOX / 'prompt_rows'}:{tree_path}",
        "PROMPT_ROWS_DUMP": str(dump),
    }
    server = pair_check.serve(0, arm_tree, MODEL, PORT, ["--config", pair_check.CONFIG, "--colocate"],
                              out / "server.log")
    try:
        rc = subprocess.run(
            [sys.executable, "-m", "benchmarks.eval.benchmark_omni_mmsu", "--host", "127.0.0.1", "--port", str(PORT),
             "--model", "qwen3-omni", "--modalities", "text+audio", "--repo-id", REPO_ID, "--max-samples",
             str(SAMPLES), "--prompt", MMSU_TTS_PROMPT, "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
             "--max-concurrency", "1", "--warmup", "0", "--output-dir", str(out / "client"), "--disable-tqdm"],
            cwd=arm_tree, env={**pair_check.card_env(0, arm_tree), "PYTHONPATH": str(arm_tree)},
            stdout=open(out / "client.log", "w"), stderr=subprocess.STDOUT).returncode
        print(f"client rc {rc}", flush=True)
    finally:
        pair_check.stop(0, server)
    return [torch.load(path) for path in sorted(dump.glob("*.pt"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(args.arm)
    records = serve_and_record(arm_tree, args.out)
    print(f"{len(records)} talker prompt prefills recorded", flush=True)

    sys.path.insert(0, str(arm_tree))
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

    samples = load_mmsu_samples(max_samples=SAMPLES, repo_id=REPO_ID)
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cuda:1", enable_audio_output=False,
        experts_implementation="eager").eval()
    accept_layer = model.config.talker_config.accept_hidden_layer
    audio_token_id = model.config.thinker_config.audio_token_id
    multimodal_ids = torch.tensor([audio_token_id, model.config.thinker_config.image_token_id,
                                   model.config.thinker_config.video_token_id])

    def reference_rows(record: dict, sample, dtype: torch.dtype) -> torch.Tensor:
        audio = librosa.load(sample.audio_path, sr=16000)[0]
        features = processor.feature_extractor(audio, sampling_rate=16000, padding=True, truncation=False,
                                               return_attention_mask=True, return_tensors="pt")
        prompt_ids = record["prompt_ids"][None].to("cuda:1")
        with torch.inference_mode():
            output = model.thinker(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                input_features=features["input_features"].to("cuda:1", dtype=dtype),
                feature_attention_mask=features["attention_mask"].to("cuda:1"),
                output_hidden_states=True, use_cache=False)
        return output.hidden_states[accept_layer][0, record["positions"]].float().cpu()

    results = []
    for record, sample in zip(records, samples):
        expected = torch.isin(record["prompt_ids"], multimodal_ids).nonzero(as_tuple=True)[0].tolist()
        results.append(dict(sample_id=sample.sample_id, request_id=record["request_id"],
                            positions_equal=record["positions"] == expected, rows=len(expected),
                            has_rows=record["rows"] is not None))
    truths = [reference_rows(record, sample, torch.float32) if result["has_rows"] and result["positions_equal"]
              else None for record, sample, result in zip(records, samples, results)]
    model.to(torch.bfloat16)
    for record, sample, result, truth in zip(records, samples, results, truths):
        if truth is None:
            continue
        else:
            pass
        floor = reference_rows(record, sample, torch.bfloat16)
        result.update(served_rel_l2=relative_l2(record["rows"].float(), truth),
                      hf_bf16_rel_l2=relative_l2(floor, truth),
                      zeros_rel_l2=relative_l2(torch.zeros_like(truth), truth))
        result["ratio"] = result["served_rel_l2"] / result["hf_bf16_rel_l2"]
        print(f"{result['sample_id']}: {result['rows']} rows, served {result['served_rel_l2']:.4f}, "
              f"hf bf16 {result['hf_bf16_rel_l2']:.4f}, ratio {result['ratio']:.2f}", flush=True)
    measured = [result for result in results if "ratio" in result]
    ratios = torch.tensor([result["ratio"] for result in measured])
    summary = dict(
        arm=args.arm, recorded=len(records), measured=len(measured),
        positions_equal=sum(result["positions_equal"] for result in results),
        has_rows=sum(result["has_rows"] for result in results),
        ratio_geometric_mean=float(ratios.log().mean().exp()) if len(ratios) else None,
        ratio_max=float(ratios.max()) if len(ratios) else None,
        served_rel_l2_mean=sum(result["served_rel_l2"] for result in measured) / max(len(measured), 1))
    print(f"summary {summary}", flush=True)
    (args.out / "prompt_rows.json").write_text(json.dumps(dict(summary=summary, results=results), indent=1))


if __name__ == "__main__":
    main()
