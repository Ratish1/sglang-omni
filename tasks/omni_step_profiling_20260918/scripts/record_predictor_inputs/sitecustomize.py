"""P1-e1 recorder: the served talker's predictor inputs, captured from a live server.

Put this directory first on PYTHONPATH of a recording boot only, with
PREDICTOR_RECORD_OUT=<file.pt> (and optionally PREDICTOR_RECORD_ROWS, default 1600).
Every Qwen3TTSTalker.code_predictor_forward call appends its rows (layer-0 code, talker
hidden) until the row budget is reached, then the file is written once. The host copy
syncs the stream; this boot is for recording, not timing.
"""

import os

if os.environ.get("PREDICTOR_RECORD_OUT"):
    import torch

    from sglang_omni.models.qwen3_tts import sglang_model

    _OUT = os.environ["PREDICTOR_RECORD_OUT"]
    _BUDGET = int(os.environ.get("PREDICTOR_RECORD_ROWS", "1600"))
    _codes: list[torch.Tensor] = []
    _hidden: list[torch.Tensor] = []
    _original = sglang_model.Qwen3TTSTalker.code_predictor_forward

    def _recording_forward(self, layer0_codes, talker_hidden, semantic_positions=None):
        rows = sum(c.shape[0] for c in _codes)
        if rows < _BUDGET:
            codes = layer0_codes.reshape(layer0_codes.shape[0], -1)[:, -1]
            hidden = talker_hidden.reshape(
                talker_hidden.shape[0], -1, talker_hidden.shape[-1]
            )[:, -1]
            _codes.append(codes.detach().to("cpu", torch.long))
            _hidden.append(hidden.detach().to("cpu"))
            if rows + codes.shape[0] >= _BUDGET:
                torch.save(
                    {
                        "talker_hidden": torch.cat(_hidden),
                        "layer0_codes": torch.cat(_codes),
                    },
                    _OUT,
                )
                print(
                    f"predictor recorder: saved {rows + codes.shape[0]} rows to {_OUT}",
                    flush=True,
                )
        return _original(self, layer0_codes, talker_hidden, semantic_positions)

    sglang_model.Qwen3TTSTalker.code_predictor_forward = _recording_forward
