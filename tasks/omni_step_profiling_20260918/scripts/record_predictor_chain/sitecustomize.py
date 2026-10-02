"""Recorder of the served predictor chain: inputs and the codes it produced.

Put this directory first on PYTHONPATH of a recording boot only, with
PREDICTOR_CHAIN_OUT=<file.pt> (and optionally PREDICTOR_CHAIN_ROWS, default 4096). Every
Qwen3TTSTalker.code_predictor_forward call appends, per row of its last position, the talker
hidden, the layer 0 code, the 15 codes the predictor sampled after it and the call's batch
size, until the row budget is reached; the file is then written once. The host copies sync
the stream, so this boot is for recording, not timing.

PREDICTOR_CHAIN_LOGITS=1 also records each pass's logits (hooks on the lm_heads, so it needs
the eager predictor: SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0) and the call's sampling buffers
(temperature, top k, top p, do sample) as the sampler read them.
"""

import os

if os.environ.get("PREDICTOR_CHAIN_OUT"):
    import torch

    from sglang_omni.models.qwen3_tts import sglang_model

    OUT = os.environ["PREDICTOR_CHAIN_OUT"]
    BUDGET = int(os.environ.get("PREDICTOR_CHAIN_ROWS", "4096"))
    WITH_LOGITS = os.environ.get("PREDICTOR_CHAIN_LOGITS") == "1"
    recorded: dict[str, list[torch.Tensor]] = {
        "talker_hidden": [],
        "layer0_codes": [],
        "codes": [],
        "batch_size": [],
    }
    if WITH_LOGITS:
        for key in ("logits", "temperature", "top_k", "top_p", "do_sample"):
            recorded[key] = []
    else:
        pass
    pass_logits: list[torch.Tensor] = []
    hooked: set[int] = set()
    original = sglang_model.Qwen3TTSTalker.code_predictor_forward

    def keep_logits(module, inputs, output):
        pass_logits.append(output[0][:, -1, :].detach().to("cpu"))

    def recording_forward(self, layer0_codes, talker_hidden, semantic_positions=None):
        if WITH_LOGITS and id(self) not in hooked:
            for head in self.code_predictor.lm_head:
                head.register_forward_hook(keep_logits)
            hooked.add(id(self))
        else:
            pass
        pass_logits.clear()
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
            if WITH_LOGITS:
                passes = codes.shape[1] - 1
                recorded["logits"].append(torch.stack(pass_logits[-passes:], dim=1))
                recorded["temperature"].append(
                    self.sub_temperature_tensor[:batch_size].detach().to("cpu")
                )
                recorded["top_k"].append(
                    self.sub_top_k_tensor[:batch_size].detach().to("cpu")
                )
                recorded["top_p"].append(
                    self.sub_top_p_tensor[:batch_size].detach().to("cpu")
                )
                recorded["do_sample"].append(
                    self.sub_do_sample_tensor[:batch_size].detach().to("cpu")
                )
            else:
                pass
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
