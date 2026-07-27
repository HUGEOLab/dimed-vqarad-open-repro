#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
sha256sum -c MANIFEST.sha256
python3 -m py_compile scripts/*.py controls/sft/*.py
bash -n ./*.sh controls/sft/*.sh
python3 scripts/verify_inputs.py --data data/vqarad
bash run_cached_reproduction.sh
bash controls/sft/run_cached_sft_reproduction.sh
echo "PASS: hashes, syntax, split, main result, and SFT controls are valid."
