"""The codes a tree's reference batcher gives chosen references, as served: the tokenizer loaded as the server loads it,
the batcher with its graphs (and an eager batcher), each reference encoded once; codes saved to OUT.pt. compare prints,
per reference, whether two trees' codes are byte identical and which frames and codebooks differ.

usage: PYTHONPATH=TREE python3 reference_codes_one.py encode CASES.jsonl --ids ID [ID ...] --model M --out OUT.pt
       python3 reference_codes_one.py compare FIRST.pt SECOND.pt
"""

import argparse
import json
from pathlib import Path

import torch


def encode(cases_path: Path, ids: list[str], model: str, out: Path) -> None:
    import librosa

    from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    cases = {case["id"]: case for case in (json.loads(line) for line in cases_path.read_text().splitlines() if line)}
    device = "cuda:0"
    torch.cuda.set_device(device)
    tokenizer = load_qwen3_tts_tokenizer(model, device=device, dtype="bfloat16", attn_implementation=None)
    graphs = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    eager = Qwen3TTSRefCodeBatcher(tokenizer)
    saved = {}
    with torch.inference_mode():
        for case_id in ids:
            waveform, rate = librosa.load(str(cases[case_id]["ref_audio"]), sr=None, mono=True)
            saved[case_id] = {
                "graphs": graphs.encode_waveform(waveform, rate).cpu(),
                "eager": eager.encode_waveform(waveform, rate).cpu(),
            }
            print(f"{case_id}: {tuple(saved[case_id]['graphs'].shape)} frames x codebooks", flush=True)
    graphs.close()
    eager.close()
    torch.save(saved, out)


def compare(first_path: Path, second_path: Path) -> None:
    first, second = torch.load(first_path), torch.load(second_path)
    for case_id, codes in first.items():
        for path in ("graphs", "eager"):
            left, right = codes[path], second[case_id][path]
            differ = (left != right).nonzero().tolist() if left.shape == right.shape else "shape"
            verdict = "identical" if differ == [] else f"differ at (frame, codebook) {differ[:12]}"
            print(f"{case_id} {path}: {verdict}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    encode_parser = sub.add_parser("encode")
    encode_parser.add_argument("cases", type=Path)
    encode_parser.add_argument("--ids", nargs="+", required=True)
    encode_parser.add_argument("--model", required=True)
    encode_parser.add_argument("--out", type=Path, required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)
    args = parser.parse_args()
    if args.command == "encode":
        encode(args.cases, args.ids, args.model, args.out)
    else:
        compare(args.first, args.second)


if __name__ == "__main__":
    main()
