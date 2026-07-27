#!/usr/bin/env python3
"""Build an equal-loss CTGM branch without admitting diffuse/random regions."""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from build_clean_v2 import BLACKLIST, IMAGES, expand_box

HERE = Path(os.environ.get("CTGM_WORK_DIR", Path(__file__).resolve().parent))
MIN_SCORE = 0.30
MIN_MARGIN = 0.010


def filter_region(row):
    term = str(row.get("clinical_term") or "").strip().lower()
    scores = row.get("top_patch_scores") or []
    reason = None
    if not term or len(scores) < 3:
        reason = "missing_clinical_term"
    elif term in BLACKLIST or term.startswith("is ") or len(term) < 4:
        reason = "generic_or_template_term"
    elif scores[0] < MIN_SCORE:
        reason = "low_alignment_score"
    elif scores[0] - scores[2] < MIN_MARGIN:
        reason = "diffuse_patch_scores"
    item = dict(row)
    if reason:
        item["top_patch_indices"] = []
        item["top_patch_boxes"] = []
        item["_clean_v3"] = {"accepted": False, "reason": reason}
        return item, reason
    with Image.open(IMAGES / row["image_name"]) as image:
        width, height = image.size
    item["top_patch_boxes"] = [
        expand_box(box, width, height, factor=3.0) for box in row["top_patch_boxes"]
    ]
    item["_clean_v3"] = {
        "accepted": True, "top1_score": scores[0],
        "top1_top3_margin": scores[0] - scores[2], "box_expansion": 3.0,
    }
    return item, None


def main():
    manifest = {"min_top1_score": MIN_SCORE, "min_top1_top3_margin": MIN_MARGIN}
    for split in ("train", "test"):
        source = [
            json.loads(x) for x in (HERE / f"ctgm_{split}_min2.jsonl").read_text().splitlines()
        ]
        output, counts = [], Counter()
        for row in source:
            item, reason = filter_region(row)
            if split == "train" and reason:
                counts[reason] += 1
                continue
            output.append(item)
            counts[reason or "accepted"] += 1
        with (HERE / f"ctgm_{split}_cleanv3.jsonl").open("w") as out:
            for row in output:
                out.write(json.dumps(row) + "\n")
        manifest[f"{split}_source_rows"] = len(source)
        manifest[f"{split}_output_rows"] = len(output)
        manifest[f"{split}_policy"] = dict(counts)
    # Negative construction is independent of CTGM thresholds and remains the
    # medically validated cross-organ clean-v2 set.
    shutil.copyfile(
        HERE / "semantic_negatives_cleanv2.jsonl",
        HERE / "semantic_negatives_cleanv3.jsonl",
    )
    manifest["ground_repeat"] = 4
    manifest["effective_ground_rows"] = manifest["train_output_rows"] * 4
    (HERE / "build_manifest_cleanv3.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
