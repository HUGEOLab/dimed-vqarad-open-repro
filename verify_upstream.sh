#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${1:?usage: $0 /path/to/Lingshu-7B /path/to/VQA-RAD-dir}"
VQARAD_DIR="${2:?VQA-RAD directory must contain trainset.json, testset.json, images/}"
python3 "${ROOT}/scripts/verify_inputs.py" --data "${VQARAD_DIR}" \
  --base "${BASE_MODEL}" --require-images
(cd "${BASE_MODEL}" && sha256sum -c "${ROOT}/provenance/lingshu7b.sha256")
(cd "${VQARAD_DIR}/images" && sha256sum -c "${ROOT}/provenance/vqarad_images.sha256")
echo "PASS: exact base-model, split, and image hashes match."
