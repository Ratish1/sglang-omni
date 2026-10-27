#!/usr/bin/env bash
# The official qwen-tts 0.1.1 venv at /data/refenv: its declared pins (transformers 4.57.3, accelerate 1.12.0) on
# torch 2.10.0 built for CUDA 12.8, beside the serving image's python3 and without touching it.
set -eu
L=${L:-/data/refenv_build.log}
python3 -m venv /data/refenv >> $L 2>&1
/data/refenv/bin/python -m pip install -q --upgrade pip >> $L 2>&1
/data/refenv/bin/python -m pip install -q torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 >> $L 2>&1
/data/refenv/bin/python -m pip install -q qwen-tts==0.1.1 >> $L 2>&1
/data/refenv/bin/python -c "import torch, transformers, accelerate, qwen_tts; print('refenv', torch.__version__, transformers.__version__, accelerate.__version__)" >> $L 2>&1
echo "refenv_done $(date +%T)" >> $L
