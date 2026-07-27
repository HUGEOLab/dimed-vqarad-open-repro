#!/usr/bin/env python3
"""Route CTGM only to questions for which local top-k evidence is meaningful."""
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


HERE = Path(os.environ.get("CTGM_WORK_DIR", Path(__file__).resolve().parent))
GLOBAL_PATTERNS = {
    "modality": r"\bmodality\b|what type of image|what kind of image|how was .*taken",
    "plane_or_sequence": r"\bplane\b|\bsequence\b",
    "position_or_orientation": r"\bposition(?:ed|ing)?\b|\borientation\b",
    "count_or_measurement": r"how many|how big|how wide|multiple or",
    "global_classification": r"normal or abnormal|organ system",
}


def global_reason(question):
    for reason, pattern in GLOBAL_PATTERNS.items():
        if re.search(pattern, question, re.I):
            return reason
    return None


def main():
    manifest = {"source": "cleanv3", "routing": "question-only localizability gate"}
    for split in ("train", "test"):
        rows = [
            json.loads(x) for x in (HERE / f"ctgm_{split}_cleanv3.jsonl").read_text().splitlines()
        ]
        output, counts = [], Counter()
        for row in rows:
            item = dict(row)
            reason = global_reason(item["question"])
            if reason:
                item.update(top_patch_indices=[], top_patch_boxes=[])
                item["_selective_v5"] = {
                    "ctgm_enabled": False, "reason": reason,
                }
                counts[f"routed_full_image:{reason}"] += 1
                if split == "train":
                    continue
            else:
                item["_selective_v5"] = {
                    "ctgm_enabled": bool(item.get("top_patch_boxes")),
                    "reason": "localizable" if item.get("top_patch_boxes") else "no_reliable_region",
                }
                counts["local_ctgm" if item.get("top_patch_boxes") else "no_reliable_region"] += 1
            output.append(item)
        with (HERE / f"ctgm_{split}_selectivev5.jsonl").open("w") as handle:
            for item in output:
                handle.write(json.dumps(item) + "\n")
        manifest[f"{split}_source_rows"] = len(rows)
        manifest[f"{split}_output_rows"] = len(output)
        manifest[f"{split}_routing_counts"] = dict(counts)
    shutil.copyfile(
        HERE / "semantic_negatives_cleanv3.jsonl",
        HERE / "semantic_negatives_selectivev5.jsonl",
    )
    (HERE / "build_manifest_selectivev5.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
