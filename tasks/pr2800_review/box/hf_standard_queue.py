"""HF's standard Qwen3-Omni (hf_standard_reference.py, transformers 4.57.1) against omni on stage 10's speech and on
MMSU's answers.

Run on the box: python3 /workspace/box9/tasks/pr2800_review/box/hf_standard_queue.py --out /data/pr2800/<run>
1. Sample lists: stage 10's 10 Video-AMME requests (/data/pr2800/videoamme10.json) and MMSU's clips not stored at
   16 kHz plus the first 200 at 16 kHz, with the benchmark's question text.
2. The reference: speech on cards 0 to 2, MMSU on card 3, in the isolated environment.
3. One Qwen3-ASR server on card 0 scores every reference speech output against its text as CI does; the MMSU
   responses are parsed by the benchmark's parser and compared, clip by clip, with omni's main (run239 ups, linear
   resampling) and #2861 (run239 phx, band limited). Writes <out>/speech_wer.json, <out>/mmsu_analysis.json and
   <out>/summary.txt.
--speech-cards and --mmsu-card place the jobs (--mmsu-card -1 runs speech only, scored on the first speech card);
--zero-multimodal-hidden and --fp8-talker pass on to the reference's speech jobs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

REFERENCE_PATH = "/data/pr2800/pylib/nokernels:/data/pr2800/pylib/hf457:/data/pr2800/pylib/qou"
SPEECH_SAMPLES = Path("/data/pr2800/videoamme10.json")
SPEECH_CARDS = (0, 1, 2)
MMSU_CARD = 3
OMNI_MMSU = {"main linear": "/data/pr2800/run239/ups/mmsu.json", "#2861 band limited": "/data/pr2800/run239/phx/mmsu.json"}
CONTROL_COUNT = 200
PORT = 37000


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def write_mmsu_samples(path: Path) -> None:
    import soundfile
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.tasks.audio_understanding import DEFAULT_PROMPT, _build_question_text

    samples = load_mmsu_samples(repo_id="zhaochenyang20/mmsu-ci-2000")
    rates = {sample.sample_id: soundfile.info(io.BytesIO(Path(sample.audio_path).read_bytes())).samplerate
             for sample in samples}
    chosen = [sample for sample in samples if rates[sample.sample_id] != 16000]
    chosen += [sample for sample in samples if rates[sample.sample_id] == 16000][:CONTROL_COUNT]
    path.write_text(json.dumps({"samples": [
        dict(sample_id=sample.sample_id, audio_path=sample.audio_path, native_rate=rates[sample.sample_id],
             question=_build_question_text(sample, DEFAULT_PROMPT), choices=sample.choices,
             answer_index=sample.answer_index, answer_text=sample.answer_text)
        for sample in chosen]}, indent=1))


def run_reference(out: Path, task: str, samples: Path, shard: int, shards: int, card: int,
                  reference_args: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": REFERENCE_PATH, "CUDA_VISIBLE_DEVICES": str(card)}
    log(out, f"{task} shard {shard} card {card} start")
    rc = subprocess.run(
        [sys.executable, str(BOX / "hf_standard_reference.py"), "--task", task, "--samples", str(samples), "--shard",
         str(shard), "--shards", str(shards), "--out", str(out), *reference_args],
        cwd="/tmp", env=env, stdout=open(out / f"{task}_{shard}.log", "w"), stderr=subprocess.STDOUT).returncode
    log(out, f"{task} shard {shard} card {card} rc {rc}")
    return rc


def score_speech(port: int, out: Path) -> None:
    from benchmarks.tasks.asr import compute_text_audio_consistency_from_records

    records = [record for path in sorted(out.glob("speech_*.json")) for record in json.loads(path.read_text())]
    wer = compute_text_audio_consistency_from_records(
        records, "en", "cuda:0", audio_dir=str(out / "audio"), asr_router_port=port, asr_concurrency=4)
    (out / "speech_wer.json").write_text(json.dumps(dict(records=records, wer=wer), indent=1))


def analyze_mmsu(out: Path) -> list[str]:
    from benchmarks.dataset.mmsu import normalize_text
    from benchmarks.metrics.accuracy import INDEX_TO_LETTER
    from benchmarks.tasks.audio_understanding import _extract_prediction

    spec = {sample["sample_id"]: sample for sample in json.loads((out / "mmsu_samples.json").read_text())["samples"]}
    reference = {record["sample_id"]: record for record in json.loads((out / "mmsu_0.json").read_text())}
    hf = {}
    for sample_id, record in reference.items():
        sample = spec[sample_id]
        index, answer = _extract_prediction(record["response"], sample["choices"])
        hf[sample_id] = dict(choice=INDEX_TO_LETTER.get(index, ""), is_correct=index == sample["answer_index"] or (
            bool(answer) and normalize_text(answer) == normalize_text(sample["answer_text"])))
    lines = []
    analysis: dict[str, object] = {}
    for group, keep in (("resampled", lambda rate: rate != 16000), ("16 kHz", lambda rate: rate == 16000)):
        ids = [sample_id for sample_id in hf if keep(spec[sample_id]["native_rate"])]
        line = (f"{group}: {len(ids)} clips, HF correct {sum(hf[i]['is_correct'] for i in ids)}, "
                f"HF unparsed {sum(not hf[i]['choice'] for i in ids)}")
        for label, path in OMNI_MMSU.items():
            omni = {item["sample_id"]: item for item in json.loads(Path(path).read_text())["per_sample"]}
            agree = sum(omni[i]["predicted_choice"] == hf[i]["choice"] for i in ids)
            correct = sum(bool(omni[i]["is_correct"]) for i in ids)
            line += f"; {label} agrees {agree} ({100 * agree / len(ids):.1f} %), correct {correct}"
            analysis[f"{group} {label}"] = dict(agree=agree, correct=correct, clips=len(ids))
        lines.append(line)
    (out / "mmsu_analysis.json").write_text(json.dumps(analysis, indent=1))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-speech", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--speech-cards", default=",".join(str(card) for card in SPEECH_CARDS))
    parser.add_argument("--mmsu-card", type=int, default=MMSU_CARD, help="-1 runs speech only")
    parser.add_argument("--zero-multimodal-hidden", action="store_true",
                        help="speech: zeros at the talker's multimodal prompt rows, as omni's main builds them")
    parser.add_argument("--fp8-talker", help="speech: the talker on this FP8 checkpoint's weights and W8A8 inputs")
    args = parser.parse_args()
    if args.score_speech:
        score_speech(args.port, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    speech_cards = [int(card) for card in args.speech_cards.split(",")]
    speech_args = [
        *(["--zero-multimodal-hidden"] if args.zero_multimodal_hidden else []),
        *(["--fp8-talker", args.fp8_talker] if args.fp8_talker else []),
    ]
    is_mmsu = args.mmsu_card >= 0
    if is_mmsu:
        write_mmsu_samples(args.out / "mmsu_samples.json")
        log(args.out, "samples written")
    else:
        pass
    with ThreadPoolExecutor(max_workers=len(speech_cards) + 1) as pool:
        jobs = [pool.submit(run_reference, args.out, "speech", SPEECH_SAMPLES, shard, len(speech_cards), card,
                            speech_args)
                for shard, card in enumerate(speech_cards)]
        if is_mmsu:
            jobs.append(pool.submit(run_reference, args.out, "mmsu", args.out / "mmsu_samples.json", 0, 1,
                                    args.mmsu_card, []))
        else:
            pass
        codes = [job.result() for job in jobs]
    if any(codes):
        log(args.out, f"stopped: rcs {codes}")
        return
    else:
        pass
    lines = analyze_mmsu(args.out) if is_mmsu else []
    score_card = speech_cards[0]
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr = pair_check.serve(score_card, asr_tree, pair_check.ASR_MODEL, PORT, [], args.out / "asr_server.log")
    try:
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--score-speech", "--port", str(PORT), "--out",
             str(args.out)],
            cwd=asr_tree, env=pair_check.card_env(score_card, asr_tree), stdout=open(args.out / "score.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(score_card, asr)
    if rc == 0:
        wer = json.loads((args.out / "speech_wer.json").read_text())["wer"]
        lines.append(f"speech: {wer['summary']}")
    else:
        lines.append(f"speech scoring rc {rc}")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        log(args.out, line)
    log(args.out, "done")


if __name__ == "__main__":
    main()
