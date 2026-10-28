"""#3284 redesign facts for the fused predictor's launches (the code predictor's shapes), on the tree in PYTHONPATH:

bytes:  per launch, row block, tokens per row and pipeline stages, the shared memory Triton's compile for --arch needs
        (compile only, no launch, so a smaller target compiles on an H100; below sm90 without dependent launch).
bits:   on this GPU, each launch at 2 to 8 stages: the stages that loaded (the tree steps down where Triton refuses)
        and whether the outputs equal the 3-stage outputs bit for bit.
hashes: on this GPU's target, each launch's compiled kernel hash and bytes at the tree's default stages (main against
        ours: equal hashes are the same binaries).
host:   host time of one pass of the four launches at 16 rows, eager, after warmup (the step-down's per-launch cost).

usage: PYTHONPATH=TREE python3 predictor_fit.py bytes --arch 86
       PYTHONPATH=TREE python3 predictor_fit.py bits | hashes | host
"""

import argparse
import hashlib
import inspect
import time
from types import SimpleNamespace

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import jit

from sglang_omni.utils import predictor_layers
from sglang_omni.utils.predictor_layers import PredictorLayerShape

HIDDEN = 1024
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
INTERMEDIATE = 3072
SLOTS = 32
STAGES = (2, 3, 4, 5, 6, 8)
HAS_FITTED_STAGES = "fitted_stages" in inspect.signature(predictor_layers.mlp_up).parameters


class Launches:
    """The four launches over one row count with random weights and inputs, one K split."""

    def __init__(self, rows: int, tokens_per_row: int, device: torch.device) -> None:
        torch.manual_seed(rows * 10 + tokens_per_row)
        bf16 = dict(device=device, dtype=torch.bfloat16)
        heads = NUM_HEADS + 2 * NUM_KV_HEADS
        self.rows = rows
        self.tokens_per_row = tokens_per_row
        self.shape = PredictorLayerShape(
            hidden_size=HIDDEN, head_dim=HEAD_DIM, num_q_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
            intermediate_size=INTERMEDIATE, split_qkv=1, split_hidden=1,
        )
        self.norm = SimpleNamespace(weight=torch.rand(HIDDEN, **bf16) + 0.5, variance_epsilon=1e-6)
        self.attention = SimpleNamespace(
            qkv_proj=SimpleNamespace(weight=torch.randn(heads * HEAD_DIM, HIDDEN, **bf16) * 0.02),
            q_norm=SimpleNamespace(weight=torch.rand(HEAD_DIM, **bf16) + 0.5, variance_epsilon=1e-6),
            k_norm=SimpleNamespace(weight=torch.rand(HEAD_DIM, **bf16) + 0.5, variance_epsilon=1e-6),
            rotary_emb=SimpleNamespace(cos_sin_cache=torch.rand(SLOTS, HEAD_DIM, device=device, dtype=torch.float32)),
        )
        self.o_weight = torch.randn(HIDDEN, NUM_HEADS * HEAD_DIM, **bf16) * 0.02
        self.gate_up_weight = torch.randn(2 * INTERMEDIATE, HIDDEN, **bf16) * 0.02
        self.down_weight = torch.randn(HIDDEN, INTERMEDIATE, **bf16) * 0.02
        self.x = torch.randn(rows, HIDDEN, **bf16)
        self.attention_output = torch.randn(rows // tokens_per_row, tokens_per_row, NUM_HEADS, HEAD_DIM, **bf16)
        self.activated_in = torch.randn(rows, INTERMEDIATE, **bf16)
        self.positions = torch.randint(0, SLOTS, (rows,), device=device)
        self.partials, self.sum_sq_partials, self.counters = predictor_layers.allocate_split_scratch(
            self.shape, rows, device
        )
        self.q_out = torch.zeros(rows, NUM_HEADS * HEAD_DIM, **bf16)
        self.k_cache = torch.zeros(rows // tokens_per_row, SLOTS, NUM_KV_HEADS, HEAD_DIM, **bf16)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.residual_out = torch.zeros(rows, HIDDEN, **bf16)
        self.activated = torch.zeros(rows, INTERMEDIATE, **bf16)
        self.down_residual = self.x.clone()

    def run(self, name: str, fitted_stages: dict) -> tuple[torch.Tensor, ...]:
        extra = {"fitted_stages": fitted_stages} if HAS_FITTED_STAGES else {}
        if name == "attention_inputs":
            predictor_layers.attention_inputs(
                x=self.x, tokens_per_row=self.tokens_per_row, norm=self.norm, attention=self.attention,
                q_out=self.q_out, positions=self.positions, k_cache=self.k_cache, v_cache=self.v_cache,
                partials=self.partials, sum_sq_partials=self.sum_sq_partials, counters=self.counters,
                shape=self.shape, **extra,
            )
            return (self.q_out, self.k_cache, self.v_cache)
        elif name == "o_proj_add":
            predictor_layers.o_proj_add(
                attention_output=self.attention_output.transpose(1, 2), tokens_per_row=self.tokens_per_row,
                weight=self.o_weight, residual_in=self.x, residual_out=self.residual_out, partials=self.partials,
                counters=self.counters, shape=self.shape, **extra,
            )
            return (self.residual_out,)
        elif name == "mlp_up":
            predictor_layers.mlp_up(
                residual=self.x, norm=self.norm, weight=self.gate_up_weight, activated=self.activated,
                shape=self.shape, **extra,
            )
            return (self.activated,)
        else:
            self.down_residual.copy_(self.x)
            predictor_layers.down_add(
                activated=self.activated_in, weight=self.down_weight, residual=self.down_residual,
                partials=self.partials, counters=self.counters, shape=self.shape, **extra,
            )
            return (self.down_residual,)

    def run_decode_attention(self) -> None:
        predictor_layers.decode_attention(
            q=self.q_out, tokens_per_row=self.tokens_per_row, k_cache=self.k_cache, v_cache=self.v_cache,
            end=SLOTS, attention_output=self.residual_out.new_zeros(self.rows, NUM_HEADS * HEAD_DIM),
        )


NAMES = ("attention_inputs", "o_proj_add", "mlp_up", "down_add")


def compile_only(records: list) -> None:
    original_run = jit.JITFunction.run

    def run(self, *run_args, grid, warmup, **kwargs):
        kernel = original_run(self, *run_args, grid=grid, warmup=True, **kwargs)
        ptx_lines = kernel.asm["ptx"].split(".section\t.debug")[0].splitlines()
        ptx = "\n".join(line for line in ptx_lines if not line.strip().startswith((".loc", ".file", "$L__")))
        records.append((self.fn.__name__, kwargs.get("BLOCK_M", kwargs.get("BLOCK_Q")), kwargs.get("T"),
                        kwargs.get("num_stages"), kernel.metadata.shared,
                        f"{kernel.hash[:12]} ptx-without-lines {hashlib.sha256(ptx.encode()).hexdigest()[:16]}"))
        return kernel

    jit.JITFunction.run = run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bytes", "bits", "hashes", "host"))
    parser.add_argument("--arch", type=int, default=0)
    options = parser.parse_args()
    device = torch.device("cuda")
    print(f"triton {triton.__version__}, {torch.cuda.get_device_name()}, fitted_stages API {HAS_FITTED_STAGES}", flush=True)
    if options.mode == "bytes":
        target = GPUTarget("cuda", options.arch, 32)
        triton.runtime.driver.active.get_current_target = lambda: target
        predictor_layers.is_arch_support_pdl = lambda: options.arch >= 90
        records = []
        compile_only(records)
        for num_stages in STAGES:
            predictor_layers.NUM_STAGES = num_stages
            for rows in (16, 32, 64):
                launches = Launches(rows, 1, device)
                for name in NAMES:
                    launches.run(name, {})
                if num_stages == STAGES[0]:
                    launches.run_decode_attention()
                else:
                    pass
        seen = set()
        for kernel_name, block_m, tokens_per_row, num_stages, shared, _ in records:
            line = f"sm{options.arch} {kernel_name} BLOCK_M {block_m} T {tokens_per_row} stages {num_stages}: {shared} bytes"
            if line not in seen:
                seen.add(line)
                print(line, flush=True)
            else:
                pass
    elif options.mode == "hashes":
        records = []
        compile_only(records)
        for rows in (16, 32, 64):
            for tokens_per_row in (1, 2):
                launches = Launches(rows, tokens_per_row, device)
                for name in NAMES:
                    launches.run(name, {})
                launches.run_decode_attention()
        for kernel_name, block_m, tokens_per_row, num_stages, shared, kernel_hash in sorted(set(records), key=str):
            print(f"{kernel_name} BLOCK_M {block_m} T {tokens_per_row} stages {num_stages}: {shared} bytes {kernel_hash}", flush=True)
    elif options.mode == "bits":
        for rows in (16, 32, 64):
            for tokens_per_row in (1, 2):
                launches = Launches(rows, tokens_per_row, device)
                for name in NAMES:
                    outputs = {}
                    for num_stages in STAGES:
                        predictor_layers.NUM_STAGES = num_stages
                        fitted_stages = {}
                        try:
                            result = launches.run(name, fitted_stages)
                            torch.cuda.synchronize()
                            outputs[num_stages] = ([output.clone() for output in result], fitted_stages.get((name, max(16, triton.next_power_of_2(rows))), num_stages))
                        except triton.runtime.OutOfResources as error:
                            outputs[num_stages] = (None, f"OutOfResources {error}")
                    reference = outputs[3][0]
                    cells = []
                    for num_stages in STAGES:
                        result, loaded = outputs[num_stages]
                        identical = result is not None and all(torch.equal(a, b) for a, b in zip(result, reference))
                        cells.append(f"{num_stages}: loaded {loaded} identical {identical}")
                    print(f"rows {rows} T {tokens_per_row} {name} | " + " | ".join(cells), flush=True)
        predictor_layers.NUM_STAGES = 3
    else:
        launches = Launches(16, 1, device)
        fitted_stages = {}
        for _ in range(200):
            for name in NAMES:
                launches.run(name, fitted_stages)
        torch.cuda.synchronize()
        for repeat in range(5):
            start = time.perf_counter()
            for _ in range(2000):
                for name in NAMES:
                    launches.run(name, fitted_stages)
            torch.cuda.synchronize()
            print(f"host repeat {repeat}: {1e6 * (time.perf_counter() - start) / 2000:.2f} us per pass of four launches", flush=True)


if __name__ == "__main__":
    main()
