"""Check every thinker prefill chunk's multimodal rows against the encoder output, in the serving process.

Enabled by OMNI_SLICE_PROBE=<output prefix>. Wraps ThinkerModelRunner.inject_multimodal_embeds read-only. For each request
in a prefill batch with video or image rows, the expected encoder rows are fixed by the prompt alone: the chunk
[prefix, prefix + length) holds the placeholders at positions p, and placeholder p takes encoder row
(placeholders before p). The check compares the step's input embedding rows with those encoder rows and, when the step
carries DeepStack features, each DeepStack layer's rows with the same encoder rows of that layer, bit for bit. One JSON
line per request per step to <prefix>.<pid>.jsonl.
"""

import json
import os
import sys
import threading
import time

PREFIX = os.environ.get("OMNI_SLICE_PROBE")
MODULE = "sglang_omni.model_runner.thinker_model_runner"


def install(module) -> None:
    torch = sys.modules["torch"]
    lock = threading.Lock()
    path = f"{PREFIX}.{os.getpid()}.jsonl"
    runner_class = module.ThinkerModelRunner
    original = runner_class.inject_multimodal_embeds

    def inject_multimodal_embeds(self, forward_batch, schedule_batch):
        snapshot = []
        for req in schedule_batch.reqs:
            inputs = req.omni_model_inputs
            if not inputs:
                snapshot.append(None)
                continue
            else:
                pass
            pad_values = inputs.get("pad_values", {})
            positions = self.req_mm_token_positions(req, pad_values)
            snapshot.append(
                {
                    "positions": {k: v.detach().clone() for k, v in positions.items()},
                    "video": inputs.get("video_embeds"),
                    "image": inputs.get("image_embeds"),
                    "video_ds": inputs.get("video_deepstack_visual_embeds"),
                    "image_ds": inputs.get("image_deepstack_visual_embeds"),
                    "ds": inputs.get("deepstack_visual_embeds"),
                }
            )
        result = original(self, forward_batch, schedule_batch)
        if result is None or result[0] is None:
            return result
        else:
            pass
        input_embeds, ds_out, vis_mask = result
        full_ds = None
        if ds_out is not None and vis_mask is not None:
            stacked = torch.cat([t.to(input_embeds.device) for t in ds_out], dim=-1)
            full_ds = torch.zeros(
                input_embeds.shape[0],
                stacked.shape[-1],
                device=input_embeds.device,
                dtype=stacked.dtype,
            )
            full_ds[vis_mask] = stacked.to(full_ds.dtype)
        else:
            pass
        extend = forward_batch.extend_seq_lens_cpu
        prefixes = forward_batch.extend_prefix_lens_cpu
        extend = extend.tolist() if isinstance(extend, torch.Tensor) else list(extend)
        prefixes = (
            prefixes.tolist() if isinstance(prefixes, torch.Tensor) else list(prefixes)
        )
        offset = 0
        lines = []
        for index, req in enumerate(schedule_batch.reqs):
            length, prefix = int(extend[index]), int(prefixes[index])
            entry = snapshot[index]
            if entry is not None:
                record = {
                    "time": time.time(),
                    "rid": req.rid,
                    "prompt_len": len(req.origin_input_ids),
                    "prefix": prefix,
                    "length": length,
                    "batch_reqs": len(schedule_batch.reqs),
                    "batch_rows": int(input_embeds.shape[0]),
                }
                for modality in ("video", "image"):
                    encoder = entry[modality]
                    places = entry["positions"][modality].to("cpu")
                    inside = places[(places >= prefix) & (places < prefix + length)]
                    if encoder is None or inside.numel() == 0:
                        continue
                    else:
                        pass
                    first = int((places < prefix).sum())
                    rows = (inside - prefix + offset).to(input_embeds.device)
                    want = encoder[first : first + inside.numel()].to(
                        device=input_embeds.device, dtype=input_embeds.dtype
                    )
                    got = input_embeds.index_select(0, rows)
                    record[f"{modality}_rows"] = int(inside.numel())
                    record[f"{modality}_first_encoder_row"] = first
                    record[f"{modality}_embed_equal"] = bool(torch.equal(got, want))
                    layers = entry[f"{modality}_ds"] or entry["ds"]
                    if layers is not None and full_ds is not None:
                        want_ds = torch.cat(
                            [
                                layer[first : first + inside.numel()].to(
                                    device=full_ds.device, dtype=full_ds.dtype
                                )
                                for layer in layers
                            ],
                            dim=-1,
                        )
                        got_ds = full_ds.index_select(0, rows)
                        record[f"{modality}_deepstack_equal"] = bool(
                            torch.equal(got_ds, want_ds)
                        )
                    else:
                        record[f"{modality}_deepstack_equal"] = None
                lines.append(json.dumps(record))
            else:
                pass
            offset += length
        if lines:
            with lock:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
        else:
            pass
        return result

    runner_class.inject_multimodal_embeds = inject_multimodal_embeds


def watch() -> None:
    while MODULE not in sys.modules or not hasattr(
        sys.modules[MODULE], "ThinkerModelRunner"
    ):
        time.sleep(0.2)
    install(sys.modules[MODULE])


if PREFIX:
    threading.Thread(target=watch, name="thinker-slice-probe", daemon=True).start()
else:
    pass
