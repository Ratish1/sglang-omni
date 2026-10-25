"""Review-only fixture correction to validate the proposed test strengthening."""

import numpy as np
from transformers.models.mimi.modeling_mimi import MimiEuclideanCodebook


def pytest_runtest_setup(item):
    module = item.module
    if item.name != 'test_reference_code_batcher_replays_captured_buckets_and_encodes_the_rest':
        return
    np.random.seed(3177)
    original = module.MimiModel

    def initialized_model(*args, **kwargs):
        model = original(*args, **kwargs)
        for codebook in model.modules():
            if isinstance(codebook, MimiEuclideanCodebook):
                codebook.embed_sum.normal_()
        return model

    module.MimiModel = initialized_model
