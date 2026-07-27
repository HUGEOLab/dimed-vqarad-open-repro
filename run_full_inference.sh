#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${1:?usage: $0 /path/to/Lingshu-7B /path/to/VQA-RAD-dir [cuda:0]}"
VQARAD_DIR="${2:?VQA-RAD directory must contain trainset.json, testset.json, images/}"
DEVICE="${3:-cuda:0}"
OUT="${ROOT}/work/full"
mkdir -p "${OUT}"
bash "${ROOT}/verify_upstream.sh" "${BASE_MODEL}" "${VQARAD_DIR}"
export VQARAD_DIR DIMED_ARTIFACT_DIR="${ROOT}/artifacts/verifier"
python3 "${ROOT}/scripts/infer_counterfactual_verifier.py" \
  --base "${BASE_MODEL}" --adapter "${ROOT}/artifacts/adapters/openfocusv2_final" \
  --output "${OUT}/baseline.json" --tag selectivev5 --baseline_only --device "${DEVICE}"
python3 "${ROOT}/scripts/extract_visual_verifier_v2.py" \
  --split test --base "${BASE_MODEL}" \
  --adapter "${ROOT}/artifacts/adapters/openfocusv2_final" \
  --baseline "${OUT}/baseline.json" --output "${OUT}/features.json" \
  --device "${DEVICE}" --resume
python3 "${ROOT}/scripts/apply_visual_verifier_v3_frozen.py" \
  --features "${OUT}/features.json" --baseline "${OUT}/baseline.json" \
  --model "${ROOT}/artifacts/verifier/verifier_v3_selector.joblib" \
  --calibration "${ROOT}/artifacts/verifier/verifier_v3_train_calibration.json" \
  --output "${OUT}/verifier_v3.json"
python3 "${ROOT}/scripts/apply_open_retrieval_verifier_v7.py" \
  --train "${VQARAD_DIR}/trainset.json" --test "${VQARAD_DIR}/testset.json" \
  --baseline "${OUT}/verifier_v3.json" --output "${OUT}/final.json" \
  --calibration "${OUT}/verifier_v7_calibration.json"
python3 "${ROOT}/scripts/score_classic_vqarad.py" "${OUT}/final.json" \
  --label-policy classic --testset "${VQARAD_DIR}/testset.json" \
  --out "${OUT}/exact_metrics.json"
echo "Full inference finished. Run run_semantic_judge.sh for semantic metrics."
