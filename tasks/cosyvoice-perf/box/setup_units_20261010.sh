#!/usr/bin/env bash
# Prepares the container for the full unit suite on PR #2632's head the way CI's
# .github/scripts/prepare_omni_venv.sh prepares its venv, on the image's python3 (/opt/sglang, the
# environment CI's venv sits on): the project's dependencies the image lacks or pins differently
# (minicpm-o and fun-cosyvoice3 extras), the uv overrides without deps, omni editable without deps,
# CosyVoice and Matcha-TTS at CI's commit, then CI's own import probe and pin check.
# usage (in the container): bash setup_units_20261010.sh <expected head sha>
set -euo pipefail
HEAD_SHA=$1
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-conv-pos-embed-kernel
git checkout -q FETCH_HEAD
[ "$(git rev-parse HEAD)" = "$HEAD_SHA" ] || { echo "head $(git rev-parse HEAD) is not $HEAD_SHA"; exit 1; }
echo "head $(git log --oneline -1)"
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')

mapfile -t MISSING < <(python3 .github/scripts/omni_missing_dependencies.py \
  --extra minicpm-o --extra fun-cosyvoice3 pyproject.toml)
echo "missing from the image:"
printf '  %s\n' "${MISSING[@]}"
for requirement in "${MISSING[@]}"; do
  python3 -m pip install -q "$requirement"
done
mapfile -t OVERRIDES < <(python3 .github/scripts/omni_missing_dependencies.py --overrides pyproject.toml)
python3 -m pip install -q --no-deps "${OVERRIDES[@]}"
uv pip install -q --python "$(command -v python3)" --no-deps -e .

COSYVOICE_COMMIT=$(grep -oP '^COSYVOICE_COMMIT=\K[0-9a-f]+' .github/scripts/prepare_omni_venv.sh)
COSYVOICE=/workspace/CosyVoice
if [ ! -d $COSYVOICE/.git ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE
fi
git -C $COSYVOICE checkout -q --detach "$COSYVOICE_COMMIT"
git -C $COSYVOICE submodule update -q --init --depth=1 third_party/Matcha-TTS
printf '%s\n' $COSYVOICE $COSYVOICE/third_party/Matcha-TTS > "$SITE/cosyvoice.pth"
echo "cosyvoice $(git -C $COSYVOICE rev-parse HEAD) matcha $(git -C $COSYVOICE/third_party/Matcha-TTS rev-parse HEAD)"

OMNI_CI_HOME=/opt bash .github/scripts/validate_omni_venv_imports.sh sglang
OMNI_CI_HOME=/opt bash .github/scripts/verify_omni_installed_pins.sh sglang
echo "still missing: [$(python3 .github/scripts/omni_missing_dependencies.py --check \
  --extra minicpm-o --extra fun-cosyvoice3 pyproject.toml | tr '\n' ' ')]"
python3 -c "
import importlib.metadata as m, sglang, sglang_omni
print('sglang', sglang.__version__, sglang.__file__)
print('sglang_omni', sglang_omni.__file__)
for name in ('torch', 'triton', 'flashinfer-python', 'transformers', 'sgl-kernel'):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, 'absent')
"
echo "setup done $(date +%T)"
