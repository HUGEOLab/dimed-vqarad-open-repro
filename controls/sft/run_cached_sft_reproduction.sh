#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../.." && pwd)"
RESULTS="${HERE}/results"
OUT="${ROOT}/work/cached_sft"
mkdir -p "${OUT}"

python3 "${ROOT}/scripts/verify_inputs.py" --data "${ROOT}/data/vqarad"

score_exact() {
  local stem="$1"
  python3 "${ROOT}/scripts/score_classic_vqarad.py" \
    "${RESULTS}/${stem}.json" --label-policy classic \
    --testset "${ROOT}/data/vqarad/testset.json" \
    --out "${OUT}/${stem}_exact.json"
  cmp "${OUT}/${stem}_exact.json" "${RESULTS}/${stem}_exact.json"
}

score_semantic() {
  local stem="$1"
  python3 "${ROOT}/scripts/score_semantic_vqarad.py" \
    "${RESULTS}/${stem}.json" "${RESULTS}/${stem}_semantic_audit.json" \
    --testset "${ROOT}/data/vqarad/testset.json" \
    --out "${OUT}/${stem}_semantic_metrics.json"
  cmp "${OUT}/${stem}_semantic_metrics.json" \
    "${RESULTS}/${stem}_semantic_metrics.json"
}

score_exact conventional_sft_model_only
score_exact matched_sft_stage1_model_only
score_exact matched_sft_model_only
score_semantic conventional_sft_model_only
score_semantic matched_sft_model_only

python3 "${HERE}/analyze_sft_gap.py" > "${OUT}/analysis_stdout.txt"
python3 - <<'PY' "${HERE}/sft_gap_analysis.json" "${HERE}/EXPECTED_SFT_METRICS.json"
import json, sys
analysis = json.load(open(sys.argv[1]))
expected = json.load(open(sys.argv[2]))
checks = {
    "conventional_sft": (
        analysis["metrics"]["conventional_sft_exact"]["overall_exact"],
        analysis["metrics"]["conventional_sft_semantic"]["overall_semantic"],
    ),
    "matched_sft_final": (
        analysis["metrics"]["matched_sft_exact"]["overall_exact"],
        analysis["metrics"]["matched_sft_semantic"]["overall_semantic"],
    ),
    "method_model_only": (
        analysis["metrics"]["method_model_only_exact"]["overall_exact"],
        analysis["metrics"]["method_model_only_semantic"]["overall_semantic"],
    ),
}
for name, (exact, semantic) in checks.items():
    assert exact == expected[name]["exact"]["overall"], (name, exact)
    assert semantic == expected[name]["semantic"]["overall"], (name, semantic)
print("PASS: frozen SFT predictions, metrics, and paired analysis reproduce.")
PY
