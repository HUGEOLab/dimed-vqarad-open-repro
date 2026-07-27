#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${1:?usage: $0 /path/to/Lingshu-7B /path/to/VQA-RAD-dir [GPU-index]}"
VQARAD_DIR="${2:?VQA-RAD directory must contain trainset.json, testset.json, images/}"
GPU="${3:-0}"
bash "${ROOT}/verify_upstream.sh" "${BASE_MODEL}" "${VQARAD_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}" VQARAD_DIR DIMED_ARTIFACT_DIR="${ROOT}/artifacts/train"
python3 "${ROOT}/scripts/train_dimed.py" \
  --model "${BASE_MODEL}" --out "${ROOT}/work/retrained_openfocusv2" \
  --init_adapter "${ROOT}/artifacts/adapters/e2_final" --load_in_4bit \
  --faithful_tag openfocusv1 --faithful_open_only --closed_anchor_rate 0.10 \
  --use_ground --use_open_pairs --open_pattern_only --no_neg \
  --ground_repeat 2 --ground_dim 0.6 --ground_mode dim \
  --epochs 1 --lr 2e-6 --img_side 224 --grad_accum 16
