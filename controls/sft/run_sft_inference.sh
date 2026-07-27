#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../.." && pwd)"
BASE_MODEL="${1:?usage: $0 /path/to/Lingshu-7B /path/to/VQA-RAD-dir [cuda:0]}"
VQARAD_DIR="${2:?VQA-RAD directory must contain trainset.json, testset.json, images/}"
DEVICE="${3:-cuda:0}"
OUT="${ROOT}/work/sft_inference"
mkdir -p "${OUT}"

bash "${ROOT}/verify_upstream.sh" "${BASE_MODEL}" "${VQARAD_DIR}"
export VQARAD_DIR DIMED_REPRO_ROOT="${ROOT}" \
  DIMED_ARTIFACT_DIR="${ROOT}/artifacts/verifier"

infer_score() {
  local name="$1"
  local adapter="$2"
  python3 "${ROOT}/scripts/infer_counterfactual_verifier.py" \
    --base "${BASE_MODEL}" --adapter "${adapter}" \
    --output "${OUT}/${name}.json" --tag selectivev5 \
    --baseline_only --device "${DEVICE}"
  python3 "${ROOT}/scripts/score_classic_vqarad.py" \
    "${OUT}/${name}.json" --label-policy classic \
    --testset "${VQARAD_DIR}/testset.json" \
    --out "${OUT}/${name}_exact.json"
}

infer_score conventional_sft_model_only \
  "${HERE}/adapters/conventional_sft_1epoch"
infer_score matched_sft_stage1_model_only \
  "${HERE}/adapters/matched_sft_stage1"
infer_score matched_sft_model_only \
  "${HERE}/adapters/matched_sft_stage2"
echo "PASS: shipped SFT adapter inference and exact scoring completed."
