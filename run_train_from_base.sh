#!/usr/bin/env bash
# Train both stages from the bare base model, then score, using no shipped adapter.
#
#   bash run_train_from_base.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir [GPU-index]
#
# Stage 1 (e2)          : 2 epochs, LR 2e-4, on the 9,445-row cleanv3 corpus
# Stage 2 (openfocusv2) : 1 epoch,  LR 2e-6, on the 3,992-row OPEN-focused corpus
# Then: baseline inference -> CTGM features -> verifier v3 -> verifier v7 -> exact score
#
# Both stages use per-device batch size 1 with gradient accumulation 16, matching
# provenance/training_args.bin. This is a method reproduction, not a bitwise one:
# see scripts/build_e2_corpus.py for why stage 1 cannot be bitwise reproduced.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${1:?usage: $0 /path/to/Lingshu-7B /path/to/VQA-RAD-dir [GPU-index]}"
VQARAD_DIR="${2:?VQA-RAD directory must contain trainset.json, testset.json, images/}"
GPU="${3:-0}"
OUT="${ROOT}/work/from_base"
CORPUS="${OUT}/corpus"
mkdir -p "${OUT}/logs"

bash "${ROOT}/verify_upstream.sh" "${BASE_MODEL}" "${VQARAD_DIR}"

python3 "${ROOT}/scripts/build_e2_corpus.py" --out "${CORPUS}" \
  2>&1 | tee "${OUT}/logs/build_corpus.log"

export CUDA_VISIBLE_DEVICES="${GPU}" VQARAD_DIR DIMED_REPRO_ROOT="${ROOT}"

# ---- Stage 1: e2 from the bare base model -----------------------------------
DIMED_ARTIFACT_DIR="${CORPUS}" python3 "${ROOT}/scripts/train_dimed.py" \
  --model "${BASE_MODEL}" --out "${OUT}/e2" --load_in_4bit \
  --faithful_tag cleanv3 --use_ground --ground_repeat 4 \
  --ground_dim 0.6 --ground_mode dim \
  --epochs 2 --lr 2e-4 --img_side 224 --grad_accum 16 \
  2>&1 | tee "${OUT}/logs/e2_train.log"

# ---- Stage 2: OPEN-focused continuation from the freshly trained e2 ---------
DIMED_ARTIFACT_DIR="${CORPUS}" python3 "${ROOT}/scripts/train_dimed.py" \
  --model "${BASE_MODEL}" --out "${OUT}/openfocusv2" \
  --init_adapter "${OUT}/e2/final" --load_in_4bit \
  --faithful_tag openfocusv1 --faithful_open_only --closed_anchor_rate 0.10 \
  --use_ground --use_open_pairs --open_pattern_only --no_neg \
  --ground_repeat 2 --ground_dim 0.6 --ground_mode dim \
  --epochs 1 --lr 2e-6 --img_side 224 --grad_accum 16 \
  2>&1 | tee "${OUT}/logs/openfocusv2_train.log"

# ---- Inference and the two verifier stages ----------------------------------
# Note: these scripts reuse DIMED_ARTIFACT_DIR for the *verifier* artifacts.
export DIMED_ARTIFACT_DIR="${ROOT}/artifacts/verifier"

python3 "${ROOT}/scripts/infer_counterfactual_verifier.py" \
  --base "${BASE_MODEL}" --adapter "${OUT}/openfocusv2/final" \
  --output "${OUT}/baseline.json" --tag selectivev5 \
  --baseline_only --device cuda:0 \
  2>&1 | tee "${OUT}/logs/baseline_inference.log"

python3 "${ROOT}/scripts/extract_visual_verifier_v2.py" \
  --split test --base "${BASE_MODEL}" --adapter "${OUT}/openfocusv2/final" \
  --baseline "${OUT}/baseline.json" --output "${OUT}/features.json" \
  --device cuda:0 \
  2>&1 | tee "${OUT}/logs/feature_extraction.log"

python3 "${ROOT}/scripts/apply_visual_verifier_v3_frozen.py" \
  --features "${OUT}/features.json" --baseline "${OUT}/baseline.json" \
  --model "${ROOT}/artifacts/verifier/verifier_v3_selector.joblib" \
  --calibration "${ROOT}/artifacts/verifier/verifier_v3_train_calibration.json" \
  --output "${OUT}/verifier_v3.json" \
  2>&1 | tee "${OUT}/logs/verifier_v3.log"

python3 "${ROOT}/scripts/apply_open_retrieval_verifier_v7.py" \
  --train "${VQARAD_DIR}/trainset.json" --test "${VQARAD_DIR}/testset.json" \
  --baseline "${OUT}/verifier_v3.json" --output "${OUT}/final.json" \
  --calibration "${OUT}/verifier_v7_calibration.json" \
  2>&1 | tee "${OUT}/logs/verifier_v7.log"

# ---- Exact scoring ----------------------------------------------------------
python3 "${ROOT}/scripts/score_classic_vqarad.py" "${OUT}/baseline.json" \
  --label-policy classic --testset "${VQARAD_DIR}/testset.json" \
  --out "${OUT}/baseline_exact.json"
python3 "${ROOT}/scripts/score_classic_vqarad.py" "${OUT}/final.json" \
  --label-policy classic --testset "${VQARAD_DIR}/testset.json" \
  --out "${OUT}/final_exact.json"

cat <<EOF

TRAIN_FROM_BASE_DONE
  adapters : ${OUT}/e2/final , ${OUT}/openfocusv2/final
  responses: ${OUT}/final.json
  exact    : ${OUT}/final_exact.json

Semantic is the reported metric. Run the judge on the responses above:

  bash run_semantic_judge.sh ${OUT}/final.json \\
    /path/to/Qwen3-VL-8B-Instruct cuda:0 ${VQARAD_DIR}/testset.json
EOF
