#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../.." && pwd)"
BASE_MODEL="${1:?usage: $0 /path/to/Lingshu-7B /path/to/VQA-RAD-dir [GPU-index]}"
VQARAD_DIR="${2:?VQA-RAD directory must contain trainset.json, testset.json, images/}"
GPU="${3:-0}"
OUT="${ROOT}/work/retrained_sft"
mkdir -p "${OUT}"

bash "${ROOT}/verify_upstream.sh" "${BASE_MODEL}" "${VQARAD_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}" VQARAD_DIR DIMED_REPRO_ROOT="${ROOT}" \
  DIMED_ARTIFACT_DIR="${ROOT}/artifacts/train"

python3 "${ROOT}/scripts/train_dimed.py" \
  --model "${BASE_MODEL}" --out "${OUT}/conventional_sft_1epoch" \
  --classic_mode all --no_pattern --no_neg --load_in_4bit \
  --epochs 1 --lr 1e-5 --img_side 224 --grad_accum 16

python3 "${ROOT}/scripts/train_dimed.py" \
  --model "${BASE_MODEL}" --out "${OUT}/matched_sft_stage1" \
  --classic_mode all --no_pattern --no_neg --load_in_4bit \
  --epochs 100 --max_steps 1182 --lr 2e-4 \
  --img_side 224 --grad_accum 16

python3 "${ROOT}/scripts/train_dimed.py" \
  --model "${BASE_MODEL}" --out "${OUT}/matched_sft_stage2" \
  --init_adapter "${OUT}/matched_sft_stage1/final" --load_in_4bit \
  --faithful_tag openfocusv1 --faithful_open_only \
  --closed_anchor_rate 0.10 --no_pattern --no_neg \
  --epochs 100 --max_steps 250 --lr 2e-6 \
  --img_side 224 --grad_accum 16

infer_score() {
  local name="$1"
  local adapter="$2"
  python3 "${ROOT}/scripts/infer_counterfactual_verifier.py" \
    --base "${BASE_MODEL}" --adapter "${adapter}" \
    --output "${OUT}/${name}.json" --tag selectivev5 \
    --baseline_only --device cuda:0
  python3 "${ROOT}/scripts/score_classic_vqarad.py" \
    "${OUT}/${name}.json" --label-policy classic \
    --testset "${VQARAD_DIR}/testset.json" \
    --out "${OUT}/${name}_exact.json"
}

export DIMED_ARTIFACT_DIR="${ROOT}/artifacts/verifier"
infer_score conventional_sft_model_only \
  "${OUT}/conventional_sft_1epoch/final"
infer_score matched_sft_stage1_model_only \
  "${OUT}/matched_sft_stage1/final"
infer_score matched_sft_model_only \
  "${OUT}/matched_sft_stage2/final"
echo "PASS: all ordinary-SFT controls retrained, inferred, and exact-scored."
