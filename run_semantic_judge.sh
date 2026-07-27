#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${1:-${ROOT}/work/full/final.json}"
JUDGE_MODEL="${2:?usage: $0 results.json /local/path/to/Qwen3-VL-8B-Instruct [cuda:0]}"
DEVICE="${3:-cuda:0}"
TESTSET="${4:-${ROOT}/data/vqarad/testset.json}"
AUDIT="${RESULTS%.json}_semantic_audit.json"
METRICS="${RESULTS%.json}_semantic_metrics.json"
python3 "${ROOT}/scripts/judge_open_local.py" "${RESULTS}" "${AUDIT}" \
  --model "${JUDGE_MODEL}" --device "${DEVICE}" --batch-size 12
python3 "${ROOT}/scripts/score_semantic_vqarad.py" "${RESULTS}" "${AUDIT}" \
  --testset "${TESTSET}" --out "${METRICS}"
