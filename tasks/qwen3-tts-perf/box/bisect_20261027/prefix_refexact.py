"""Calls the branch's exact-length reference test against another tree's source: that tree comes first on the path,
and a name the branch's test imports but the tree lacks is stood in, so the test reads the tree's own batcher codes.

usage: PYTHONPATH=TREE python3 prefix_refexact.py BRANCH_TEST_FILE
"""

import importlib.util
import sys

import sglang_omni
from sglang_omni.models.qwen3_tts import reference_encoder_cuda_graph

print(f"sglang_omni from {sglang_omni.__file__}")
if not hasattr(reference_encoder_cuda_graph, "exact_length_encode"):
    reference_encoder_cuda_graph.exact_length_encode = None
    print("stood in exact_length_encode")
spec = importlib.util.spec_from_file_location("branch_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.test_reference_codes_match_the_exact_length_encode()
except AssertionError as exc:
    print(f"FAILED: {exc!r}")
else:
    print("PASSED")
