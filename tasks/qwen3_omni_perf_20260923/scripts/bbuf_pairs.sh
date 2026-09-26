#!/usr/bin/env bash
# The BBuf three tables (kernel, overlap, fuse) for every mapping and formal pair of a shapes
# census (run_census_boot.sh MODE shapes-*): one report per capture, numbers from the formal
# trace, lines from the mapping trace. usage: bbuf_pairs.sh <shapes out root> <analyzer scripts dir>
set -u
R=$1 A=$2
pair() {
  local name=$1 sub=$2
  python3 "$A/analyze_llm_torch_profile.py" --framework sglang \
    --mapping-input "$R/mapping_c0/mapping$sub" --formal-input "$R/formal_c1/formal$sub" \
    --kernel-table-limit 40 > "$R/report_$name.md" 2> "$R/report_$name.err"
  echo "$name rc=$? $(wc -l < "$R/report_$name.md") lines"
}
pair tp_min _min/thinker_prefill_text_text/b1
pair tp_2k _2k/thinker_prefill_text_text/b1
pair tp_8k _8k/thinker_prefill_text_text/b1
pair td_b1 /thinker_decode_text_text/b1
pair td_b64 /thinker_decode_text_text/b64
pair kp_min _min/talker_ar_prefill_text_speech/b1
pair kp_2k _2k/talker_ar_prefill_text_speech/b1
pair kp_8k _8k/talker_ar_prefill_text_speech/b1
pair kd_b1 /talker_ar_decode_text_speech/b1
pair kd_b32 /talker_ar_decode_text_speech/b32
