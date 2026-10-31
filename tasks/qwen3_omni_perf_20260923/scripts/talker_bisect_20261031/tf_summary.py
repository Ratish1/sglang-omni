"""queue104's teacher-forced talker parity, bad requests against their same-sample controls.

Per group: omni's mean per-frame hidden deviation from HF FP32 over HF BF16's (geometric mean and max over requests),
the largest decode input deviation, first-code top-1 agreement, the mean KL from HF FP32's served distribution to
omni's, residual code departures (omni against HF BF16), and P(EOS) after the text ends under each. Then one line per
bad request.

usage: python3 tf_summary.py /data/q104
"""

import json
import math
import sys
from pathlib import Path

ROOT = sys.argv[1]
results = {r["path"]: r for r in json.loads(Path(ROOT, "tf.json").read_text())}
selection = json.loads(Path(ROOT, "selection.json").read_text())
groups = {"bad": [], "control": []}
for item in selection:
    if item["bad"] in results:
        groups["bad"].append((item, results[item["bad"]]))
    else:
        pass
    for path in item["controls"]:
        if path in results:
            groups["control"].append((item, results[path]))
        else:
            pass

for name, rows in groups.items():
    if not rows:
        print(f"{name}: none")
        continue
    else:
        pass
    ratios = [r["hidden_dev_omni_mean"] / r["hidden_dev_floor_mean"] for _, r in rows]
    print(
        f"{name}: {len(rows)} requests, {sum(r['frames'] for _, r in rows)} frames; hidden deviation omni/BF16 "
        f"geometric mean {math.exp(sum(map(math.log, ratios)) / len(ratios)):.3f} max {max(ratios):.3f}; "
        f"decode input deviation max {max(r['decode_input_dev_max'] for _, r in rows):.4f}; "
        f"code0 top-1 agreement {sum(r['code0_top1_agreement'] for _, r in rows) / len(rows):.4f}; "
        f"KL mean {sum(r['kl_mean'] for _, r in rows) / len(rows):.4f}; residual departures omni "
        f"{sum(r['predictor_departures_omni'] for _, r in rows)} HF BF16 "
        f"{sum(r['predictor_departures_floor'] for _, r in rows)} of {sum(r['predictor_codes'] for _, r in rows)}"
    )

print()
for item, r in sorted(groups["bad"], key=lambda pair: -pair[1]["frames"]):
    after_truth = r["p_eos_truth_after_end_mean"]
    after_omni = r["p_eos_omni_after_end_mean"]
    print(
        f"{item['pass']} {item['sample'][-28:]} wer {item['wer']:.2f} {item['audio']:.1f} s frames {r['frames']} "
        f"text end {r['eos_frame_text_end']} | hidden omni {r['hidden_dev_omni_mean']:.4f} BF16 "
        f"{r['hidden_dev_floor_mean']:.4f} | top-1 {r['code0_top1_agreement']:.3f} KL {r['kl_mean']:.4f} | "
        f"sampled log p HF FP32 {r['sampled_logp_truth']:.1f} omni {r['sampled_logp_omni']:.1f} | P(EOS) after end "
        f"HF FP32 {after_truth if after_truth is None else round(after_truth, 4)} omni "
        f"{after_omni if after_omni is None else round(after_omni, 4)} | residual departures "
        f"{r['predictor_departures_omni']}/{r['predictor_departures_floor']}"
    )
