#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ROOT}/work/cached"

python3 "${ROOT}/scripts/verify_inputs.py" --data "${ROOT}/data/vqarad"
python3 "${ROOT}/scripts/apply_visual_verifier_v3_frozen.py" \
  --features "${ROOT}/artifacts/verifier/verifier_openfocusv2_test_features.json" \
  --baseline "${ROOT}/artifacts/results/results_openfocusv2_baseline.json" \
  --model "${ROOT}/artifacts/verifier/verifier_v3_selector.joblib" \
  --calibration "${ROOT}/artifacts/verifier/verifier_v3_train_calibration.json" \
  --output "${ROOT}/work/cached/verifier_v3.json"
python3 "${ROOT}/scripts/apply_open_retrieval_verifier_v7.py" \
  --train "${ROOT}/data/vqarad/trainset.json" \
  --test "${ROOT}/data/vqarad/testset.json" \
  --baseline "${ROOT}/work/cached/verifier_v3.json" \
  --output "${ROOT}/work/cached/final.json" \
  --calibration "${ROOT}/work/cached/verifier_v7_calibration.json"
python3 "${ROOT}/scripts/compare_responses.py" \
  "${ROOT}/work/cached/final.json" \
  "${ROOT}/artifacts/results/results_openfocusv2_verifier_v3_v7.json"
python3 "${ROOT}/scripts/score_classic_vqarad.py" \
  "${ROOT}/work/cached/final.json" --label-policy classic \
  --testset "${ROOT}/data/vqarad/testset.json" \
  --out "${ROOT}/work/cached/exact_metrics.json"
python3 "${ROOT}/scripts/score_semantic_vqarad.py" \
  "${ROOT}/work/cached/final.json" \
  "${ROOT}/artifacts/results/results_openfocusv2_verifier_v3_v7_semantic_qwen3.json" \
  --testset "${ROOT}/data/vqarad/testset.json" \
  --out "${ROOT}/work/cached/semantic_metrics.json"
echo "PASS: cached reproduction matches the frozen responses and metrics."
