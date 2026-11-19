"""Whether a video's vision-encoder output depends on the other videos encoded in the same forward, module by module.

The target video and its partners are preprocessed by the stage's own preprocessor at the CI video settings. The
image encoder (the stage's Qwen3OmniImageEncoder) encodes the target alone, as the reference, with a forward hook on
every vision module that keeps the target's rows. It then encodes the target again alone, and inside batches with
partners (the stage's batched call concatenates every request's pixel_values_videos and grid rows), with the target
first and with the target last. For every module, in the reference's execution order: whether the target's rows are
bit-identical to the reference, their max abs difference and relative L2 difference. Prints the first differing
module per run and the final video_embeds and DeepStack rows, and writes every row to --out.

usage: PYTHONPATH=<tree> python3 encoder_batch_identity.py --target 005-3 --partners 001-1,008-2,014-3,020-1,010-2,017-1,019-2 --out enc.json
"""

import argparse
import asyncio
import json

import torch

from preprocess_stage_bench import MODEL, VIDEO_SETTINGS, VIDEOMME_CI


def preprocess(video_paths: dict[str, str]) -> dict[str, dict[str, torch.Tensor]]:
    from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
    from sglang_omni.proto.request import OmniRequest, StagePayload

    preprocessor = Qwen3OmniPreprocessor(model_path=MODEL, max_seq_len=32768, video_decode_threads=8)
    inputs = {}
    for name, path in video_paths.items():
        payload = StagePayload(
            request_id=name,
            request=OmniRequest(
                inputs={
                    "messages": [{"role": "user", "content": "Describe the video."}],
                    "videos": [path],
                    **VIDEO_SETTINGS,
                },
                params={},
            ),
            data={},
        )
        result = asyncio.run(preprocessor(payload))
        encoder = result.data["encoder_inputs"]["image_encoder"]
        inputs[name] = {
            "pixel_values_videos": encoder["pixel_values_videos"],
            "video_grid_thw": encoder["video_grid_thw"],
        }
    return inputs


class Recorder:
    """Forward hooks on every vision module, keeping the target's rows of each output whose rows are patches or merged tokens."""

    def __init__(self, visual: torch.nn.Module) -> None:
        self.reference: dict[str, torch.Tensor] | None = None
        self.order: list[str] = []
        self.rows: tuple[int, int, int, int, int] | None = None
        self.results: dict[str, dict[str, object]] = {}
        for name, module in visual.named_modules():
            if name:
                module.register_forward_hook(self.hook(name))
            else:
                pass

    def hook(self, name: str):
        def record(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor) or self.rows is None:
                return
            else:
                pass
            patch_total, token_total, patch_start, token_start, patch_count = self.rows
            token_count = patch_count // 4
            if tensor.shape[0] == patch_total:
                rows = tensor[patch_start : patch_start + patch_count]
            elif tensor.shape[0] == token_total:
                rows = tensor[token_start : token_start + token_count]
            else:
                return
            if self.reference is None:
                self.order.append(name)
                self.results[name] = {}
                self.current[name] = rows.detach().clone()
            elif name in self.reference:
                want = self.reference[name]
                difference = (rows.float() - want.float()).flatten()
                self.results[name] = {
                    "equal": bool(torch.equal(rows, want)),
                    "max_abs": float(difference.abs().max()),
                    "rel_l2": float(difference.norm() / want.float().flatten().norm().clamp_min(1e-30)),
                }
            else:
                pass

        return record

    current: dict[str, torch.Tensor] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="005-3")
    parser.add_argument("--partners", default="001-1,008-2,014-3,020-1,010-2,017-1,019-2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from benchmarks.dataset.videomme import load_videomme_samples
    from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder

    samples = {sample.question_id: sample for sample in load_videomme_samples(repo_id=VIDEOMME_CI)}
    names = [args.target, *args.partners.split(",")]
    inputs = preprocess({name: samples[name].video_path for name in names})
    model = Qwen3OmniImageEncoder(model_path=MODEL, device="cuda:0")
    recorder = Recorder(model.visual)

    def encode(order: list[str]) -> dict[str, object]:
        pixels = torch.cat([inputs[name]["pixel_values_videos"] for name in order], dim=0)
        grids = torch.cat([inputs[name]["video_grid_thw"] for name in order], dim=0)
        counts = [int(inputs[name]["video_grid_thw"].prod(-1).sum()) for name in order]
        start = sum(counts[: order.index(args.target)])
        count = counts[order.index(args.target)]
        recorder.rows = (sum(counts), sum(counts) // 4, start, start // 4, count)
        with torch.no_grad():
            output = model(pixel_values_videos=pixels, video_grid_thw=grids)
        torch.cuda.synchronize()
        token_start, token_count = start // 4, count // 4
        return {
            "video_embeds": output["video_embeds"][token_start : token_start + token_count],
            "deepstack": [layer[token_start : token_start + token_count] for layer in output["deepstack_visual_embeds_video"]],
        }

    reference = encode([args.target])
    recorder.reference = recorder.current
    runs = {
        "alone again": [args.target],
        "with 1, first": names[:2],
        "with 3, first": names[:4],
        "with 7, first": names[:8],
        "with 1, last": [names[1], args.target],
        "with 3, last": [*names[1:4], args.target],
        "with 7, last": [*names[1:8], args.target],
    }
    report = {"target": args.target, "patch_rows": int(inputs[args.target]["video_grid_thw"].prod(-1).sum()), "runs": {}}
    for label, order in runs.items():
        recorder.results = {}
        output = encode(order)
        first = next((name for name in recorder.order if not recorder.results.get(name, {}).get("equal", True)), None)
        final = {
            "video_embeds_equal": bool(torch.equal(output["video_embeds"], reference["video_embeds"])),
            "video_embeds_max_abs": float((output["video_embeds"].float() - reference["video_embeds"].float()).abs().max()),
            "deepstack_equal": [bool(torch.equal(a, b)) for a, b in zip(output["deepstack"], reference["deepstack"])],
        }
        report["runs"][label] = {"order": order, "first_differing_module": first, "final": final,
                                 "modules": {name: recorder.results.get(name) for name in recorder.order}}
        print(f"{label}: first differing module {first}, final {final}", flush=True)
        if first is not None:
            index = recorder.order.index(first)
            for name in recorder.order[index : index + 6]:
                print(f"    {name}: {recorder.results.get(name)}", flush=True)
            last = recorder.order[-1]
            print(f"    ... {last}: {recorder.results.get(last)}", flush=True)
        else:
            pass
    with open(args.out, "w") as sink:
        json.dump(report, sink, indent=1)


if __name__ == "__main__":
    main()
