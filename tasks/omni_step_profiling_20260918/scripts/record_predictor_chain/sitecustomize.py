"""Recorder of the served predictor chain: inputs and the codes it produced.

Put this directory first on PYTHONPATH of a recording boot only, with
PREDICTOR_CHAIN_OUT=<file.pt> (and optionally PREDICTOR_CHAIN_ROWS, default 4096). Every
Qwen3TTSTalker.code_predictor_forward call appends, per row of its last position, the talker
hidden, the layer 0 code, the 15 codes the predictor sampled after it and the call's batch
size, until the row budget is reached; the file is then written once. The host copies sync
the stream, so this boot is for recording, not timing.
"""

import os

if os.environ.get("PREDICTOR_CHAIN_OUT"):
    import torch

    from sglang_omni.models.qwen3_tts import sglang_model

    OUT = os.environ["PREDICTOR_CHAIN_OUT"]
    BUDGET = int(os.environ.get("PREDICTOR_CHAIN_ROWS", "4096"))
    recorded: dict[str, list[torch.Tensor]] = {
        "talker_hidden": [],
        "layer0_codes": [],
        "codes": [],
        "batch_size": [],
    }
    original = sglang_model.Qwen3TTSTalker.code_predictor_forward

    def recording_forward(self, layer0_codes, talker_hidden, semantic_positions=None):
        result = original(self, layer0_codes, talker_hidden, semantic_positions)
        rows = sum(item.shape[0] for item in recorded["layer0_codes"])
        if rows < BUDGET:
            batch_size = layer0_codes.shape[0]
            codes, _ = result
            hidden = talker_hidden.reshape(batch_size, -1, talker_hidden.shape[-1])
            recorded["talker_hidden"].append(hidden[:, -1].detach().to("cpu"))
            recorded["layer0_codes"].append(
                layer0_codes.reshape(batch_size, -1)[:, -1]
                .detach()
                .to("cpu", torch.long)
            )
            recorded["codes"].append(
                codes.reshape(batch_size, codes.shape[1], -1)[:, 1:, -1]
                .detach()
                .to("cpu", torch.long)
            )
            recorded["batch_size"].append(torch.full((batch_size,), batch_size))
            if rows + batch_size >= BUDGET:
                torch.save({k: torch.cat(v) for k, v in recorded.items()}, OUT)
                print(
                    f"predictor chain recorder: {rows + batch_size} rows to {OUT}",
                    flush=True,
                )
            else:
                pass
        else:
            pass
        return result

    sglang_model.Qwen3TTSTalker.code_predictor_forward = recording_forward
