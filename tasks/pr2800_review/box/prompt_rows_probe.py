"""The talker's prompt rows through the served pipeline against HF's thinker, on stage 6's MMSU requests.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/prompt_rows_probe.py --arm <tree> --out /data/pr2800/<run>
    [--records <an earlier run's dump>]
Card 0 serves the arm with prompt_rows/sitecustomize.py on PYTHONPATH, which records what the talker's prompt prefill
receives. The first SAMPLES mmsu-ci-2000 requests go one at a time with the CI's talker prompt, text and audio,
greedy, MAX_TOKENS tokens (the talker's prompt prefill comes with the first), then the first --repeats of them again: a
repeat matches its cached prompt, so its rows come from the slots the first request wrote and must equal them. With
--records the serving is skipped and that dump is read. Card 1 then runs HF's thinker in fp32 (the truth) and in bf16 (the floor) on each request's own
prompt ids with HF's audio features (the audio loaded by librosa, or with --omni-audio by the server's own loader), and
reads hidden_states[accept_hidden_layer] at the multimodal positions. Per
request: the prompt's audio tokens against HF's feature length, the positions the talker got against the prompt's
multimodal tokens, and the relative L2 to fp32 of the served rows and of HF bf16's, with their ratio. Writes
<out>/prompt_rows.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
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


def client(port: int, out: Path, repeats: int) -> None:
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.eval.benchmark_omni_mmsu import run

    args = argparse.Namespace(
        base_url=None,
        host="127.0.0.1",
        port=port,
        model="qwen3-omni",
        modalities="text+audio",
        output_dir=str(out),
        max_samples=SAMPLES,
        task_names=None,
        categories=None,
        prompt=MMSU_TTS_PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        warmup=0,
        max_concurrency=1,
        request_rate=float("inf"),
        save_audio=False,
        disable_tqdm=True,
        seed=None,
        fingerprint=False,
        lang="en",
        asr_device="cuda:0",
        asr_concurrency=1,
        timeout_s=500,
    )
    samples = load_mmsu_samples(max_samples=SAMPLES, repo_id=REPO_ID)
    results = asyncio.run(run(args, samples=samples, compute_wer=False))
    print(
        f"{results['speed'].get('audio_returned')} of {len(samples)} requests returned audio",
        flush=True,
    )
    # note (ratish): the same requests again match their cached prompts, so the talker's rows
    # come from the slots the first requests wrote.
    if repeats:
        args.output_dir = str(out / "repeats")
        results = asyncio.run(run(args, samples=samples[:repeats], compute_wer=False))
        print(
            f"{results['speed'].get('audio_returned')} of {repeats} repeats returned audio",
            flush=True,
        )
    else:
        pass


def serve_and_record(arm_tree: Path, out: Path, repeats: int) -> Path:
    dump = out / "dump"
    dump.mkdir(parents=True, exist_ok=True)
    card_env = pair_check.card_env
    pair_check.card_env = lambda card, tree_path: {
        **card_env(card, tree_path),
        "PYTHONPATH": f"{BOX / 'prompt_rows'}:{tree_path}",
        "PROMPT_ROWS_DUMP": str(dump),
    }
    server = pair_check.serve(
        0,
        arm_tree,
        MODEL,
        PORT,
        ["--config", pair_check.CONFIG, "--colocate"],
        out / "server.log",
    )
    try:
        rc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--client",
                "--port",
                str(PORT),
                "--out",
                str(out / "client"),
                "--repeats",
                str(repeats),
            ],
            cwd=arm_tree,
            env={**card_env(0, arm_tree), "PYTHONPATH": str(arm_tree)},
            stdout=open(out / "client.log", "w"),
            stderr=subprocess.STDOUT,
        ).returncode
        print(f"client rc {rc}", flush=True)
    finally:
        pair_check.stop(0, server)
    return dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--omni-audio", action="store_true")
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.client:
        client(args.port, args.out, args.repeats)
        return
    else:
        pass
    arm_tree = pair_check.tree(args.arm)
    dump = args.records or serve_and_record(arm_tree, args.out, args.repeats)
    # note (ratish): the server's startup warmup request reaches the talker first; the client's are the last ones.
    all_records = [torch.load(path) for path in sorted(dump.glob("*.pt"))]
    records = all_records[-(SAMPLES + args.repeats) :][:SAMPLES]
    repeat_records = all_records[len(all_records) - args.repeats :]
    print(f"{len(records)} talker prompt prefills recorded", flush=True)
    repeats_equal = sum(
        torch.equal(repeat["prompt_ids"], first["prompt_ids"])
        and repeat["positions"] == first["positions"]
        and torch.equal(repeat["rows"], first["rows"])
        for repeat, first in zip(repeat_records, records)
    )
    server_log = args.out / "server.log"
    # note (ratish): a repeat's thinker prefill computes only its last prompt token.
    repeat_prefix_hits = (
        len(
            re.findall(
                r"#new-seq: 1, #new-token: 1, #cached-token: [1-9]",
                server_log.read_text(errors="replace"),
            )
        )
        if server_log.exists()
        else 0
    )
    print(
        f"repeats with the first rows: {repeats_equal} of {args.repeats}, "
        f"one-token prefills on a cached prompt: {repeat_prefix_hits}",
        flush=True,
    )

    sys.path.insert(0, str(arm_tree))
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )

    from benchmarks.dataset.mmsu import load_mmsu_samples

    samples = load_mmsu_samples(max_samples=SAMPLES, repo_id=REPO_ID)
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL,
        dtype=torch.float32,
        device_map="cuda:1",
        enable_audio_output=False,
        experts_implementation="eager",
    ).eval()
    accept_layer = model.config.talker_config.accept_hidden_layer
    audio_token_id = model.config.thinker_config.audio_token_id
    multimodal_ids = torch.tensor(
        [
            audio_token_id,
            model.config.thinker_config.image_token_id,
            model.config.thinker_config.video_token_id,
        ]
    )
    from sglang_omni.preprocessing.audio import load_audio_path

    def load_audio(path: str):
        return load_audio_path(path) if args.omni_audio else librosa.load(path, sr=16000)[0]

    features = [
        processor.feature_extractor(
            load_audio(sample.audio_path),
            sampling_rate=16000,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        for sample in samples
    ]

    def reference_rows(
        record: dict, sample_features: dict, dtype: torch.dtype
    ) -> torch.Tensor:
        prompt_ids = record["prompt_ids"][None].to("cuda:1")
        with torch.inference_mode():
            output = model.thinker(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                input_features=sample_features["input_features"].to(
                    "cuda:1", dtype=dtype
                ),
                feature_attention_mask=sample_features["attention_mask"].to("cuda:1"),
                output_hidden_states=True,
                use_cache=False,
            )
        return output.hidden_states[accept_layer][0, record["positions"]].float().cpu()

    results = []
    for record, sample, sample_features in zip(records, samples, features):
        expected = (
            torch.isin(record["prompt_ids"], multimodal_ids)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        audio_tokens = int(
            _get_feat_extract_output_lengths(sample_features["attention_mask"].sum(-1))
        )
        results.append(
            dict(
                sample_id=sample.sample_id,
                request_id=record["request_id"],
                audio_tokens_equal=int((record["prompt_ids"] == audio_token_id).sum())
                == audio_tokens,
                positions_equal=record["positions"] == expected,
                rows=len(expected),
                has_rows=record["rows"] is not None,
            )
        )
    truths = [
        (
            reference_rows(record, sample_features, torch.float32)
            if result["audio_tokens_equal"]
            and result["positions_equal"]
            and result["has_rows"]
            else None
        )
        for record, sample_features, result in zip(records, features, results)
    ]
    model.to(torch.bfloat16)
    for record, sample_features, result, truth in zip(
        records, features, results, truths
    ):
        if truth is None:
            continue
        else:
            pass
        floor = reference_rows(record, sample_features, torch.bfloat16)
        result.update(
            served_rel_l2=relative_l2(record["rows"].float(), truth),
            hf_bf16_rel_l2=relative_l2(floor, truth),
            zeros_rel_l2=relative_l2(torch.zeros_like(truth), truth),
        )
        result["ratio"] = result["served_rel_l2"] / result["hf_bf16_rel_l2"]
        print(
            f"{result['sample_id']}: {result['rows']} rows, served {result['served_rel_l2']:.4f}, "
            f"hf bf16 {result['hf_bf16_rel_l2']:.4f}, ratio {result['ratio']:.2f}",
            flush=True,
        )
    measured = [result for result in results if "ratio" in result]
    ratios = torch.tensor([result["ratio"] for result in measured])
    summary = dict(
        arm=args.arm,
        audio_loader="omni" if args.omni_audio else "librosa",
        recorded=len(records),
        repeats=args.repeats,
        repeats_equal=repeats_equal,
        repeat_prefix_hits=repeat_prefix_hits,
        measured=len(measured),
        audio_tokens_equal=sum(result["audio_tokens_equal"] for result in results),
        positions_equal=sum(result["positions_equal"] for result in results),
        has_rows=sum(result["has_rows"] for result in results),
        ratio_geometric_mean=float(ratios.log().mean().exp()) if len(ratios) else None,
        ratio_max=float(ratios.max()) if len(ratios) else None,
        served_rel_l2_mean=sum(result["served_rel_l2"] for result in measured)
        / max(len(measured), 1),
    )
    print(f"summary {summary}", flush=True)
    (args.out / "prompt_rows.json").write_text(
        json.dumps(dict(summary=summary, results=results), indent=1)
    )


if __name__ == "__main__":
    main()
