#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="${PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
else
    PYTHON="python3"
fi
MODE="${1:-correctness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/artifacts/communication-h200}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_ROOT}/${STAMP}-${MODE}"
HARNESS="${ROOT}/benchmarks/communication/stage_transport.py"

mkdir -p "${OUTPUT_DIR}"
export TMPDIR="${TMPDIR:-/tmp}"

run_case() {
    local case_name="$1"
    local dst_gpus="$2"
    local window="$3"
    local count="$4"
    shift 4
    "${PYTHON}" "${HARNESS}" \
        --case "${case_name}" \
        --src-gpu 0 \
        --dst-gpus "${dst_gpus}" \
        --window "${window}" \
        --count "${count}" \
        --output "${OUTPUT_DIR}/${case_name}-w${window}.json" \
        "$@" 2>&1 | tee "${OUTPUT_DIR}/${case_name}-w${window}.log"
}

nvidia-smi \
    --query-gpu=uuid,pci.bus_id,name,driver_version,memory.used \
    --format=csv,noheader | tee "${OUTPUT_DIR}/nvidia-smi-before.txt"
nvidia-smi topo -m | tee "${OUTPUT_DIR}/topology.txt"
git -C "${ROOT}" rev-parse HEAD | tee "${OUTPUT_DIR}/revision.txt"
"${PYTHON}" -c 'import torch; print(torch.__version__, torch.version.cuda)' \
    | tee "${OUTPUT_DIR}/torch.txt"

case "${MODE}" in
    correctness)
        failures=0
        run_case direct-payload 0 1 20 || failures=1
        run_case direct-stream 0 1 20 || failures=1
        run_case direct-payload-metadata 0 1 20 \
            --header-bytes 131072 --cpu-view-backing-bytes 4000000 || failures=1
        run_case direct-stream-metadata 0 1 20 \
            --header-bytes 131072 --cpu-view-backing-bytes 4000000 || failures=1
        run_case pooled-payload 1 1 20 || failures=1
        run_case pooled-stream 1 1 20 || failures=1
        run_case fanout 1,1 1 20 || failures=1
        ;;
    stress)
        failures=0
        for window in 1 8 16; do
            run_case direct-payload 0 "${window}" 5000 --warmups 10 || failures=1
            run_case direct-stream 0 "${window}" 5000 --warmups 10 || failures=1
            run_case pooled-payload 1 "${window}" 5000 --warmups 10 || failures=1
            run_case pooled-stream 1 "${window}" 5000 --warmups 10 || failures=1
        done
        run_case fanout 1,1 8 5000 --warmups 10 || failures=1
        ;;
    profile)
        command -v nsys >/dev/null 2>&1 || {
            echo "nsys is required for profile mode" >&2
            exit 2
        }
        for case_name in direct-payload direct-stream pooled-payload pooled-stream; do
            if [[ "${case_name}" == direct-* ]]; then
                dst_gpus=0
            else
                dst_gpus=1
            fi
            nsys profile \
                --trace=cuda,nvtx,osrt \
                --sample=none \
                --cpuctxsw=none \
                --force-overwrite=true \
                --output "${OUTPUT_DIR}/${case_name}" \
                "${PYTHON}" "${HARNESS}" \
                --case "${case_name}" \
                --src-gpu 0 \
                --dst-gpus "${dst_gpus}" \
                --warmups 1 \
                --count 5 \
                --profile \
                --output "${OUTPUT_DIR}/${case_name}.json" \
                2>&1 | tee "${OUTPUT_DIR}/${case_name}.log"
            nsys stats \
                --force-export=true \
                --report cuda_gpu_mem_size_sum,cuda_gpu_kern_sum \
                --format csv \
                --output "${OUTPUT_DIR}/${case_name}-stats" \
                "${OUTPUT_DIR}/${case_name}.nsys-rep"
        done
        failures=0
        ;;
    *)
        echo "usage: $0 {correctness|stress|profile}" >&2
        exit 2
        ;;
esac

nvidia-smi \
    --query-gpu=uuid,pci.bus_id,name,driver_version,memory.used \
    --format=csv,noheader | tee "${OUTPUT_DIR}/nvidia-smi-after.txt"

if grep -R -n -E \
    --include='*.log' --include='*.json' --include='*.txt' \
    "CudaIPCTypes.cpp|Producer process has been terminated|Traceback" \
    "${OUTPUT_DIR}"; then
    echo "unexpected CUDA IPC warning or traceback found" >&2
    exit 1
fi

if [[ "${failures}" -ne 0 ]]; then
    echo "one or more H200 ${MODE} cases failed" >&2
    exit 1
fi

echo "H200 ${MODE} artifacts: ${OUTPUT_DIR}"
