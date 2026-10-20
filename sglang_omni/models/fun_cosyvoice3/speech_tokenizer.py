# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c)  (Mddct: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: the V3 encoder and FSQ code only, masked for padded widths, weights
# read from the checkpoint's ONNX graph, replayed from CUDA graphs.
"""The Fun-CosyVoice3 speech tokenizer on CUDA, replayed from CUDA graphs."""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass

import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_prefill_cuda_graph_batch_sizes,
)

# note (ratish): 30 s of whisper's mel at 100 frames a second, the longest input the
# tokenizer takes.
MAX_FRAMES = 3000
MEL_CHANNELS = 128
STATE_CHANNELS = 1280
HEAD_COUNT = 20
FSMN_KERNEL = 31
ROTARY_POSITIONS = 2048
ROTARY_THETA = 10000.0
# note (ratish): constants of speech_tokenizer_v3.onnx's graph, not its initializers.
LAYER_NORM_EPS = 1e-5
FSQ_SCALE = 0.9990000128746033
FSQ_LEVELS = 3
FSQ_DIMS = 8
MASK_BIAS = -1.0e10
BLOCK_NODE_PARAMETERS = {
    "attn_ln/Mul": "attn_ln.weight",
    "attn_ln/Add_1": "attn_ln.bias",
    "attn/query/MatMul": "query.weight",
    "attn/query/Add": "query.bias",
    "attn/key/MatMul": "key.weight",
    "attn/value/MatMul": "value.weight",
    "attn/value/Add": "value.bias",
    "attn/out/MatMul": "out.weight",
    "attn/out/Add": "out.bias",
    "attn/fsmn_block/Conv": "fsmn.weight",
    "mlp_ln/Mul": "mlp_ln.weight",
    "mlp_ln/Add_1": "mlp_ln.bias",
    "mlp/mlp.0/MatMul": "mlp_in.weight",
    "mlp/mlp.0/Add": "mlp_in.bias",
    "mlp/mlp.2/MatMul": "mlp_out.weight",
    "mlp/mlp.2/Add": "mlp_out.bias",
}
NODE_PARAMETERS = {
    "/conv1/Conv": ("conv1.weight", "conv1.bias"),
    "/conv2/Conv": ("conv2.weight", "conv2.bias"),
    "/quantizer/project_in/MatMul": ("project_down.weight",),
    "/quantizer/project_in/Add": ("project_down.bias",),
}


def conv_frames(frames: int | torch.Tensor) -> int | torch.Tensor:
    """Frames out of a conv of kernel 3, stride 2 and padding 1."""
    return (frames - 1) // 2 + 1


def rotary_tables(head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The cosine and sine (positions, head_dim) the checkpoint was exported with."""
    frequencies = 1.0 / (
        ROTARY_THETA ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    angles = torch.outer(torch.arange(ROTARY_POSITIONS), frequencies).float()
    rotations = torch.polar(torch.ones_like(angles), angles)
    rotations = torch.view_as_real(torch.cat((rotations, rotations), dim=-1))
    return rotations[..., 0].contiguous(), rotations[..., 1].contiguous()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class SpeechTokenizerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_ln = torch.nn.LayerNorm(STATE_CHANNELS, eps=LAYER_NORM_EPS)
        self.query = torch.nn.Linear(STATE_CHANNELS, STATE_CHANNELS)
        self.key = torch.nn.Linear(STATE_CHANNELS, STATE_CHANNELS, bias=False)
        self.value = torch.nn.Linear(STATE_CHANNELS, STATE_CHANNELS)
        self.out = torch.nn.Linear(STATE_CHANNELS, STATE_CHANNELS)
        self.fsmn = torch.nn.Conv1d(
            STATE_CHANNELS,
            STATE_CHANNELS,
            FSMN_KERNEL,
            groups=STATE_CHANNELS,
            bias=False,
        )
        self.mlp_ln = torch.nn.LayerNorm(STATE_CHANNELS, eps=LAYER_NORM_EPS)
        self.mlp_in = torch.nn.Linear(STATE_CHANNELS, 4 * STATE_CHANNELS)
        self.mlp_out = torch.nn.Linear(4 * STATE_CHANNELS, STATE_CHANNELS)

    def forward(
        self,
        x: torch.Tensor,
        position_mask: torch.Tensor,
        attention_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        positions = x.shape[1]
        hidden = self.attn_ln(x)
        query = rotate(self.query(hidden).view(1, positions, HEAD_COUNT, -1), cos, sin)
        key = rotate(self.key(hidden).view(1, positions, HEAD_COUNT, -1), cos, sin)
        value = self.value(hidden)
        memory_input = value * position_mask
        memory = self.fsmn(
            F.pad(memory_input.transpose(1, 2), (FSMN_KERNEL // 2, FSMN_KERNEL // 2))
        ).transpose(1, 2)
        memory = (memory + memory_input) * position_mask
        scale = query.shape[-1] ** -0.25
        scores = (query.permute(0, 2, 1, 3) * scale) @ (key.permute(0, 2, 3, 1) * scale)
        weights = torch.softmax(scores + attention_bias, dim=-1)
        attended = weights @ value.view(1, positions, HEAD_COUNT, -1).transpose(1, 2)
        x = x + (self.out(attended.transpose(1, 2).reshape_as(x)) + memory)
        return x + self.mlp_out(F.gelu(self.mlp_in(self.mlp_ln(x))))


class SpeechTokenizerV3Model(torch.nn.Module):
    """The tokenizer's encoder and FSQ code, exact for the first frames of a mel
    whatever fills the rest of its width."""

    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            MEL_CHANNELS, STATE_CHANNELS, 3, stride=2, padding=1
        )
        self.conv2 = torch.nn.Conv1d(
            STATE_CHANNELS, STATE_CHANNELS, 3, stride=2, padding=1
        )
        self.blocks = torch.nn.ModuleList(
            SpeechTokenizerBlock() for _ in range(layer_count)
        )
        self.project_down = torch.nn.Linear(STATE_CHANNELS, FSQ_DIMS)
        cos, sin = rotary_tables(STATE_CHANNELS // HEAD_COUNT)
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self.register_buffer(
            "fsq_powers",
            FSQ_LEVELS ** torch.arange(FSQ_DIMS, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, mel: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Codes (1, positions) of mel (1, mels, width) whose first frames frames
        are the audio's."""
        device = mel.device
        x = F.gelu(
            self.conv1(mel * (torch.arange(mel.shape[2], device=device) < frames))
        )
        conv1_frames = conv_frames(frames)
        x = F.gelu(
            self.conv2(x * (torch.arange(x.shape[2], device=device) < conv1_frames))
        )
        positions = x.shape[2]
        is_code = torch.arange(positions, device=device) < conv_frames(conv1_frames)
        position_mask = is_code.to(x.dtype)[None, :, None]
        attention_bias = (~is_code).to(x.dtype)[None, None, None, :] * MASK_BIAS
        cos = self.rotary_cos[:positions, None]
        sin = self.rotary_sin[:positions, None]
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, position_mask, attention_bias, cos, sin)
        levels = torch.round(torch.tanh(self.project_down(x)) * FSQ_SCALE) + 1
        return (levels * self.fsq_powers).sum(dim=-1).to(torch.int32)


def load_speech_tokenizer(
    model_path: str, device: torch.device
) -> SpeechTokenizerV3Model:
    """The model with the parameters of its ONNX graph's initializers."""
    graph = onnx.load(model_path).graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    state_dict: dict[str, torch.Tensor] = {}
    layer_count = 0
    for node in graph.node:
        if node.name.startswith("/blocks."):
            block, suffix = node.name[len("/blocks.") :].split("/", 1)
            names = (
                (f"blocks.{block}.{BLOCK_NODE_PARAMETERS[suffix]}",)
                if suffix in BLOCK_NODE_PARAMETERS
                else ()
            )
            layer_count = max(layer_count, int(block) + 1)
        else:
            names = NODE_PARAMETERS.get(node.name, ())
        parameters = [name for name in node.input if name in initializers]
        for name, parameter in zip(names, parameters, strict=True):
            tensor = torch.from_numpy(
                numpy_helper.to_array(initializers[parameter]).copy()
            )
            # note (ratish): an ONNX MatMul's weight is (in, out), a Linear's (out, in).
            state_dict[name] = tensor.t() if node.op_type == "MatMul" else tensor
    model = SpeechTokenizerV3Model(layer_count)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


@dataclass(kw_only=True)
class CapturedTokenizerGraph:
    graph: torch.cuda.CUDAGraph
    mel: torch.Tensor
    frames: torch.Tensor
    codes: torch.Tensor


class SpeechTokenizerCudaGraphs:
    """One graph per mel width on the prefill ladder up to the tokenizer's 30 s
    limit, in one pool: replays run one at a time on the callers' stream and copy
    their codes out before the next."""

    def __init__(self, model: SpeechTokenizerV3Model, device: torch.device) -> None:
        self.widths = [
            *generate_prefill_cuda_graph_batch_sizes(MAX_FRAMES - 1),
            MAX_FRAMES,
        ]
        self.lock = threading.Lock()
        self.graphs: dict[int, CapturedTokenizerGraph] = {}
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device)
        with torch.inference_mode():
            for width in reversed(self.widths):
                mel = torch.zeros(1, MEL_CHANNELS, width, device=device)
                frames = torch.full((1,), width, dtype=torch.int64, device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        model(mel, frames)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    codes = model(mel, frames)
                self.graphs[width] = CapturedTokenizerGraph(
                    graph=graph, mel=mel, frames=frames, codes=codes
                )
        torch.cuda.synchronize(device)

    def tokenize(self, mel: torch.Tensor) -> list[int]:
        """Codes of a CPU mel (1, mels, frames) of at most MAX_FRAMES frames."""
        frames = mel.shape[2]
        captured = self.graphs[self.widths[bisect.bisect_left(self.widths, frames)]]
        with self.lock, torch.inference_mode():
            captured.mel[:, :, :frames].copy_(mel)
            captured.frames.fill_(frames)
            captured.graph.replay()
            codes = captured.codes[0, : conv_frames(conv_frames(frames))].clone()
        return codes.tolist()
