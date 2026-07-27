#!/usr/bin/env python3
"""Stage the training corpora for a bare-base (e2 -> openfocusv2) run.

`train_dimed.py` reads every faithful-mode corpus from one directory, given by
`DIMED_ARTIFACT_DIR`. This script assembles that directory from the corpora
shipped under `artifacts/train/`.

One file needs reconstructing. The recorded e2 run consumed a CTGM snapshot that
was not retained after later artifact revisions. The retained source is the
786-row candidate file `ctgm_train_cleanv3_candidates.jsonl`. The historical
builder required the normalized clinical term to occur as a literal substring of
the question before masking; applying that rule yields exactly 741 rows, which
reproduces the recorded 9,445-row corpus and 591 optimizer steps per epoch.

The row counts match the recorded run exactly. Content identity with the
historical snapshot cannot be verified, because the snapshot is gone. Treat a
bare-base run as a method reproduction, not a bitwise one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "artifacts/train"

CANDIDATES = "ctgm_train_cleanv3_candidates.jsonl"
RECOVERED = "ctgm_train_cleanv3.jsonl"
COPIES = (
    "pattern_train_lingshu.jsonl",
    "semantic_negatives_cleanv3.jsonl",
    "ctgm_train_openfocusv1.jsonl",
    "open_invariant_pairs_openfocusv1.jsonl",
)

EXPECTED_CANDIDATES = 786
EXPECTED_RETAINED = 741
E2_COMPOSITION = {
    "original": 3064, "pattern": 2822, "semantic_negative": 595,
    "ctgm_unique": 741, "ctgm_repeat": 4, "total": 9445,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="Directory to stage into; pass it as DIMED_ARTIFACT_DIR when training.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source = TRAIN_DIR / CANDIDATES
    candidates = [json.loads(line) for line in source.read_text().splitlines()]
    retained = [
        row for row in candidates
        if str(row.get("clinical_term", "")).strip().lower()
        in str(row.get("question", "")).lower()
    ]
    if len(candidates) != EXPECTED_CANDIDATES or len(retained) != EXPECTED_RETAINED:
        raise RuntimeError(
            f"CTGM reconstruction invariant failed: candidates={len(candidates)} "
            f"(expected {EXPECTED_CANDIDATES}), retained={len(retained)} "
            f"(expected {EXPECTED_RETAINED})"
        )
    (args.out / RECOVERED).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained)
    )
    for name in COPIES:
        shutil.copyfile(TRAIN_DIR / name, args.out / name)

    manifest = {
        "reconstruction_rule": "lowercase clinical_term is a literal substring of lowercase question",
        "bitwise_reproducible": False,
        "reason": "historical CTGM snapshot not retained; rule inferred from recorded row counts",
        "source_candidates": len(candidates),
        "retained_ctgm_rows": len(retained),
        "excluded_ctgm_rows": len(candidates) - len(retained),
        "e2_composition": E2_COMPOSITION,
        "expected_optimizer_steps": {
            "gradient_accumulation": 16, "steps_per_epoch": 591,
            "epochs": 2, "total_steps": 1182,
        },
        "files": {
            path.name: {"rows": sum(1 for _ in path.open()), "sha256": sha256(path)}
            for path in sorted(args.out.glob("*.jsonl"))
        },
    }
    (args.out / "e2_corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
