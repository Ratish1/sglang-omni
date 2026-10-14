#!/usr/bin/env bash
set -euo pipefail

run_root=/data/perf2798-2799-2804
source_root=/workspace/sglang-omni

until grep -q 'completed fish head c1 r1' "$run_root/fish-head-driver.log" 2>/dev/null; do
    sleep 10
done
while curl -fsS http://127.0.0.1:28981/health >/dev/null 2>&1; do
    sleep 5
done
sleep 8

cd "$run_root/base2798"
python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B \
    --asr.gpu 1 --host 127.0.0.1 --port 29020 > "$run_root/quality-asr.log" 2>&1 &
asr_process=$!
trap 'kill -TERM "$asr_process" 2>/dev/null || true' EXIT
until curl -fsS http://127.0.0.1:29020/health >/dev/null 2>&1; do
    kill -0 "$asr_process"
    sleep 5
done

for model in higgs fish; do
    if [ "$model" = higgs ]; then
        model_path=bosonai/higgs-audio-v3-tts-4b
    else
        model_path=fishaudio/s2-pro
    fi
    for arm in base head; do
        until [ -f "$run_root/$model-$arm/c16-r1/generated.json" ]; do
            sleep 5
        done
        python3 -m benchmarks.eval.benchmark_tts_seedtts --use-existing-server \
            --transcribe-only --stream --model "$model_path" \
            --asr-model-path Qwen/Qwen3-ASR-1.7B --port 29020 \
            --output-dir "$run_root/$model-$arm/c16-r1" \
            > "$run_root/$model-$arm-wer.log" 2>&1
    done
done
kill -TERM "$asr_process"
wait "$asr_process" || true
trap - EXIT

until grep -q 'completed fish base c1 r1' "$run_root/fish-base-driver.log"; do
    sleep 10
done
while curl -fsS http://127.0.0.1:28980/health >/dev/null 2>&1; do
    sleep 5
done
sleep 8
cd "$source_root"
python3 tasks/pr2798_2799_2804_review/run_pairs.py --model moss --arm head \
    --source "$run_root/head2799" --output-root "$run_root/moss-head-swapped" \
    --gpu 0 --concurrencies 64,1,32 --repeats 3 > "$run_root/moss-head-swapped-driver.log" 2>&1 &
head_process=$!
python3 tasks/pr2798_2799_2804_review/run_pairs.py --model moss --arm base \
    --source "$run_root/base2798" --output-root "$run_root/moss-base-swapped" \
    --gpu 1 --concurrencies 64,1,32 --repeats 3 > "$run_root/moss-base-swapped-driver.log" 2>&1 &
base_process=$!
wait "$head_process"
wait "$base_process"
printf 'finished quality and GPU-swapped MOSS checks\n'
