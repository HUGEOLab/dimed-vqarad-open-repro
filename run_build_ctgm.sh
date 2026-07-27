#!/usr/bin/env bash
# Rebuild the CTGM / de-confounding corpora from the shipped region scores, then
# check the result byte-for-byte against the corpora this package ships.
#
#   bash run_build_ctgm.sh [/path/to/VQA-RAD-dir]
#
# No GPU and no model. Needs only VQA-RAD (images and trainset.json) plus
# artifacts/ctgm/ctgm_{train,test}_min2.jsonl, both shipped here.
#
# Chain:
#   ctgm_{train,test}_min2.jsonl                    (BiomedCLIP region scores, shipped)
#     -> build_clean_v2.py       high-precision filter + seeded cross-organ negatives
#     -> build_clean_v3.py       stricter score/margin gate, 3x box expansion
#     -> build_selective_v5.py   route CTGM only to localizable questions
#     -> build_openfocus_v1.py   OPEN-focused branch + invariant OPEN pairs
#
# The first mile, producing min2 itself, is scripts/ctgm/build_test_regions.py and
# needs BiomedCLIP; scripts/ctgm/rewrite_patterns_lingshu.py builds the pattern
# corpus and needs Lingshu-7B. Both are shipped for audit but are not run here.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VQARAD_DIR="${1:-${ROOT}/data/vqarad}"
WORK="${ROOT}/work/ctgm"
mkdir -p "${WORK}"

if [ ! -d "${VQARAD_DIR}/images" ]; then
  echo "ERROR: ${VQARAD_DIR}/images not found. VQA-RAD images are not bundled;" >&2
  echo "       pass the directory holding trainset.json, testset.json, images/." >&2
  exit 1
fi

cp "${ROOT}/artifacts/ctgm/ctgm_train_min2.jsonl" "${WORK}/"
cp "${ROOT}/artifacts/ctgm/ctgm_test_min2.jsonl" "${WORK}/"

export CTGM_WORK_DIR="${WORK}" VQARAD_DIR PYTHONPATH="${ROOT}/scripts/ctgm"
for stage in build_clean_v2 build_clean_v3 build_selective_v5 build_openfocus_v1; do
  echo "--- ${stage}"
  python3 "${ROOT}/scripts/ctgm/${stage}.py" > "${WORK}/${stage}.log"
done

echo
echo "--- comparing against the shipped corpora"
python3 - "${ROOT}" "${WORK}" <<'PY'
import hashlib, sys
from pathlib import Path

root, work = Path(sys.argv[1]), Path(sys.argv[2])
PAIRS = [
    ("ctgm_train_cleanv3.jsonl", "artifacts/train/ctgm_train_cleanv3_candidates.jsonl"),
    ("semantic_negatives_cleanv3.jsonl", "artifacts/train/semantic_negatives_cleanv3.jsonl"),
    ("ctgm_train_openfocusv1.jsonl", "artifacts/train/ctgm_train_openfocusv1.jsonl"),
    ("open_invariant_pairs_openfocusv1.jsonl", "artifacts/train/open_invariant_pairs_openfocusv1.jsonl"),
    ("ctgm_test_selectivev5.jsonl", "artifacts/verifier/ctgm_test_selectivev5.jsonl"),
]

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

bad = []
for built, shipped in PAIRS:
    a, b = work / built, root / shipped
    if not a.is_file():
        bad.append(f"{built}: not produced"); continue
    ha, hb = sha(a), sha(b)
    rows = sum(1 for _ in a.open())
    if ha == hb:
        print(f"  OK   {built:42s} {rows} rows")
    else:
        bad.append(f"{built}: {ha[:12]} != shipped {hb[:12]}")
        print(f"  DIFF {built:42s} {ha[:12]} != {hb[:12]}")
if bad:
    print("\nFAIL: rebuilt corpora differ from the shipped ones:")
    for line in bad:
        print(f"  {line}")
    raise SystemExit(1)
print("\nPASS: every CTGM corpus rebuilds byte-for-byte from the shipped region scores.")
PY
