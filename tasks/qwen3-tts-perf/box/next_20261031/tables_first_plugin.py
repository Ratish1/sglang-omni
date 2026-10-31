"""Pytest plugin for block 41: build every codebook table before a reference encoder graph runner is made, as the
loader does for Base checkpoints (stages.py load_float32_encoder_weights), so the runner's compile traces a stored table.

usage: PYTHONPATH=TREE:THIS_DIR python3 -m pytest -p tables_first_plugin ...
"""

from transformers.models.mimi.modeling_mimi import MimiEuclideanCodebook

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import Qwen3TTSReferenceEncoderCudaGraphRunner

ORIGINAL_INIT = Qwen3TTSReferenceEncoderCudaGraphRunner.__init__


def init_with_tables(self, encoder, **kwargs) -> None:
    for module in encoder.modules():
        if isinstance(module, MimiEuclideanCodebook):
            module.embed
        else:
            pass
    ORIGINAL_INIT(self, encoder, **kwargs)


Qwen3TTSReferenceEncoderCudaGraphRunner.__init__ = init_with_tables
