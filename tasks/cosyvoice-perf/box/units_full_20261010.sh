#!/usr/bin/env bash
# The whole unit suite on PR #2632's head in both CI lanes, as .github/workflows/test.yaml runs them:
# the CPU lane with CUDA hidden, then the accelerator lane with the cards visible. CI's environment
# from .github/actions/omni-setup (no torch compile by default, no flashinfer JIT debug, its
# flashinfer workspace). No -x and one pass each, so every failure shows; CI stops at the first and
# retries up to 3 times.
# usage (in the container): bash units_full_20261010.sh
R=/data/units20261010
mkdir -p $R
cd /workspace/sglang-omni
export PYTHONPATH=$PWD
export SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0 FLASHINFER_JIT_DEBUG=0 FLASHINFER_WORKSPACE_BASE=/root
{
  echo "head $(git rev-parse HEAD)"
  python3 -c "import sglang, sglang_omni; print('sglang', sglang.__version__, sglang_omni.__file__)"
  echo "cpu start $(date +%T)"
} >> $R/progress.txt
CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/ -v -p no:cacheprovider \
  -m "not benchmark and not accelerator" -rfEs --durations=25 > $R/cpu.log 2>&1
echo "cpu rc $? $(date +%T)" >> $R/progress.txt
echo "accelerator start $(date +%T)" >> $R/progress.txt
python3 -m pytest tests/ -v -p no:cacheprovider \
  -m "accelerator and not benchmark" -rfEs --durations=25 > $R/accelerator.log 2>&1
echo "accelerator rc $? $(date +%T)" >> $R/progress.txt
echo "queue done $(date +%T)" >> $R/progress.txt
