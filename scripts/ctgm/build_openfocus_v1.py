#!/usr/bin/env python3
"""Build an OPEN-focused, answer-invariant DiMed training extension.

The extension deliberately avoids answer-flipped OPEN negatives.  It contains:

1. localizable OPEN questions with a reliable clean-v3 CTGM region;
2. global OPEN questions routed to a full-image masked-text branch; and
3. source-validated counterfactual text pairs: different questions attached to
   the same image and the same normalized OPEN gold answer.

Every row is derived from the public training split.  Test questions and test
answers are never consulted.
"""
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_selective_v5 import global_reason


VQARAD = Path(os.environ.get("VQARAD_DIR", str(Path(__file__).resolve().parents[2] / "data/vqarad")))
HERE = Path(os.environ.get("CTGM_WORK_DIR", Path(__file__).resolve().parent))
TRAIN = VQARAD / "trainset.json"


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def main():
    train = json.loads(TRAIN.read_text())
    train_by_qid = {int(row["qid"]): row for row in train}
    clean_rows = [
        json.loads(line)
        for line in (HERE / "ctgm_train_cleanv3.jsonl").read_text().splitlines()
    ]
    min2_rows = [
        json.loads(line)
        for line in (HERE / "ctgm_train_min2.jsonl").read_text().splitlines()
    ]

    # Reliable local regions come only from clean-v3.  Global questions are
    # allowed to use the full image even when their patch scores are diffuse,
    # because no spatial intervention is applied to those rows.
    local_by_qid = {
        int(row["qid"]): row
        for row in clean_rows
        if row.get("answer_type") == "OPEN" and not global_reason(row["question"])
    }
    global_by_qid = {}
    for row in min2_rows:
        if row.get("answer_type") != "OPEN":
            continue
        reason = global_reason(row["question"])
        if reason:
            item = dict(row)
            item["top_patch_indices"] = []
            item["top_patch_boxes"] = []
            item["_openfocus"] = {
                "route": "full_image",
                "reason": reason,
                "answer_policy": "unchanged_open_gold",
            }
            global_by_qid[int(row["qid"])] = item

    ground_rows = []
    route_counts = Counter()
    for qid, row in sorted(local_by_qid.items()):
        item = dict(row)
        item["_openfocus"] = {
            "route": "local_ctgm",
            "reason": "reliable_cleanv3_region",
            "answer_policy": "unchanged_open_gold",
        }
        ground_rows.append(item)
        route_counts["local_ctgm"] += 1
    for qid, row in sorted(global_by_qid.items()):
        # A qid cannot be both local and global under the question gate.
        if qid not in local_by_qid:
            ground_rows.append(row)
            route_counts[f"full_image:{row['_openfocus']['reason']}"] += 1

    with (HERE / "ctgm_train_openfocusv1.jsonl").open("w") as handle:
        for row in ground_rows:
            handle.write(json.dumps(row) + "\n")

    # Build genuine answer-invariant textual counterfactual groups from source
    # annotations.  Repeated duplicate questions are collapsed, while every
    # retained variant keeps its original qid and wording.
    groups = defaultdict(list)
    for row in train:
        if str(row.get("answer_type", "")).upper() != "OPEN":
            continue
        groups[(row["image_name"], norm(row["answer"]))].append(row)

    pair_rows = []
    pair_groups = 0
    for (image_name, answer_norm), members in sorted(groups.items()):
        unique = {}
        for row in members:
            unique.setdefault(norm(row["question"]), row)
        variants = list(unique.values())
        if len(variants) < 2:
            continue
        pair_groups += 1
        pair_id = f"{image_name}::{answer_norm}"
        qids = [int(row["qid"]) for row in variants]
        for row in variants:
            item = dict(row)
            item["_open_pair"] = True
            item["_pair_id"] = pair_id
            item["_paired_qids"] = [qid for qid in qids if qid != int(row["qid"])]
            item["_validity_rule"] = (
                "same source image and identical normalized OPEN gold answer"
            )
            item["_answer_policy"] = "unchanged"
            pair_rows.append(item)

    with (HERE / "open_invariant_pairs_openfocusv1.jsonl").open("w") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row) + "\n")

    # Kept only for CLI compatibility.  The recommended OPEN-focused run uses
    # --no_neg; copying does not silently add these rows to training.
    shutil.copyfile(
        HERE / "semantic_negatives_cleanv3.jsonl",
        HERE / "semantic_negatives_openfocusv1.jsonl",
    )
    # Test routing stays identical to selective-v5.  This builder is train-only
    # and must not select policies using test labels.
    shutil.copyfile(
        HERE / "ctgm_test_selectivev5.jsonl",
        HERE / "ctgm_test_openfocusv1.jsonl",
    )

    open_qids = {
        int(row["qid"]) for row in train if row.get("answer_type") == "OPEN"
    }
    manifest = {
        "version": "openfocusv1",
        "source_split": str(TRAIN),
        "test_labels_used_for_training_policy": False,
        "open_train_rows": len(open_qids),
        "ground_rows": len(ground_rows),
        "ground_unique_qids": len({int(row["qid"]) for row in ground_rows}),
        "ground_route_counts": dict(route_counts),
        "ground_open_coverage": round(
            len({int(row["qid"]) for row in ground_rows}) / len(open_qids), 6
        ),
        "invariant_pair_groups": pair_groups,
        "invariant_pair_rows": len(pair_rows),
        "invariant_pair_unique_qids": len({int(row["qid"]) for row in pair_rows}),
        "invariant_pair_open_coverage": round(
            len({int(row["qid"]) for row in pair_rows}) / len(open_qids), 6
        ),
        "negative_policy": "exclude with --no_neg; no OPEN answer is flipped",
        "recommended_ground_repeat": 2,
        "recommended_pattern_scope": "OPEN only",
    }
    (HERE / "build_manifest_openfocusv1.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
