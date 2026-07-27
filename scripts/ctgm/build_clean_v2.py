#!/usr/bin/env python3
"""Filter noisy CTGM artifacts into a high-precision training branch."""
import json
import os
import random
from collections import Counter
from pathlib import Path
from PIL import Image

VQARAD = Path(os.environ.get("VQARAD_DIR", str(Path(__file__).resolve().parents[2] / "data/vqarad")))
HERE = Path(os.environ.get("CTGM_WORK_DIR", Path(__file__).resolve().parent))
IMAGES = VQARAD / "images"
BLACKLIST = {"is are", "abnormality", "abnormal", "pathology", "plane", "imaging",
             "modality", "image", "film", "organ", "organ system", "located",
             "evidence", "present", "patient", "section", "normal", "normal size",
             "normal appearing", "body", "size", "view", "picture", "finding",
             "findings", "scan", "clinical term", "clinical item"}

ORGAN_NEGATIVES = {
    "HEAD": ["intracranial hemorrhage", "hydrocephalus", "cerebral infarction",
             "brain mass", "skull fracture", "midline shift"],
    "CHEST": ["pneumothorax", "pleural effusion", "cardiomegaly",
              "pulmonary nodule", "rib fracture", "mediastinal mass"],
    "ABD": ["appendicitis", "bowel obstruction", "liver mass",
            "renal calculus", "splenomegaly", "pancreatic mass"],
}


def expand_box(box, width, height, factor=3.0):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    return [max(0, round(cx - bw / 2)), max(0, round(cy - bh / 2)),
            min(width, round(cx + bw / 2)), min(height, round(cy + bh / 2))]


def clean_region(row):
    """Return a high-confidence expanded region, or an empty-region fallback."""
    raw_term = row.get("clinical_term")
    term = str(raw_term or "").strip().lower()
    scores = row.get("top_patch_scores") or []
    margin = scores[0] - scores[2] if len(scores) >= 3 else 0.0
    reason = None
    if not raw_term or len(scores) < 3:
        reason = "missing_clinical_term"
    elif term in BLACKLIST or term.startswith("is ") or len(term) < 4:
        reason = "generic_or_template_term"
    elif scores[0] < 0.35:
        reason = "low_alignment_score"
    elif margin < 0.025:
        reason = "diffuse_patch_scores"
    item = dict(row)
    if reason:
        # Test rows must stay aligned with the 451 QA rows.  An empty box list
        # explicitly disables counterfactual intervention when CTGM is unsafe.
        item["top_patch_indices"] = []
        item["top_patch_boxes"] = []
        item["_clean_v2"] = {"accepted": False, "reason": reason,
                             "top1_score": scores[0] if scores else None,
                             "top1_top3_margin": margin}
        return item, reason
    with Image.open(IMAGES / row["image_name"]) as image:
        width, height = image.size
    item["top_patch_boxes"] = [expand_box(x, width, height) for x in row["top_patch_boxes"]]
    item["_clean_v2"] = {"accepted": True, "top1_score": scores[0],
                         "top1_top3_margin": margin, "box_expansion": 3.0,
                         "background_brightness": 0.6}
    return item, None


def main():
    source = [json.loads(x) for x in (HERE / "ctgm_train_min2.jsonl").read_text().splitlines()]
    kept, reasons = [], Counter()
    for row in source:
        item, reason = clean_region(row)
        if reason:
            reasons[reason] += 1
            continue
        kept.append(item)
    with (HERE / "ctgm_train_cleanv2.jsonl").open("w") as out:
        for row in kept:
            out.write(json.dumps(row) + "\n")
    # Keep every test row in original order; rejected/noisy CTGM rows receive no
    # intervention and therefore fall back to the full-image model response.
    test_source = [json.loads(x) for x in (HERE / "ctgm_test_min2.jsonl").read_text().splitlines()]
    test_clean, test_reasons = [], Counter()
    for row in test_source:
        item, reason = clean_region(row)
        test_clean.append(item)
        test_reasons[reason or "accepted"] += 1
    with (HERE / "ctgm_test_cleanv2.jsonl").open("w") as out:
        for row in test_clean:
            out.write(json.dumps(row) + "\n")
    # Build medically valid answer-flipped negatives. A target finding is drawn
    # only from a different body compartment, so its absence is supported by the
    # image-organ annotation. Canonical presence wording avoids broken syntax.
    train = json.loads((VQARAD / "trainset.json").read_text())
    term_by_qid = {row["qid"]: row["clinical_term"] for row in source}
    rng = random.Random(42)
    negatives, strategies = [], Counter()
    for row in train:
        if str(row["answer"]).strip().lower() != "yes" or row["qid"] not in term_by_qid:
            continue
        other_organs = [x for x in ORGAN_NEGATIVES if x != row["image_organ"]]
        draw = rng.random()
        if draw < 0.6:
            strategy = "distant"
            organ = rng.choice(other_organs)
            target = rng.choice(ORGAN_NEGATIVES[organ][:4])
        elif draw < 0.8:
            strategy = "similar"
            src_term = term_by_qid[row["qid"]].lower()
            organ = rng.choice(other_organs)
            if any(x in src_term for x in ("mass", "tumor", "lesion", "nodule")):
                target = ORGAN_NEGATIVES[organ][3 if organ == "CHEST" else 3]
            elif "fracture" in src_term:
                target = "rib fracture" if organ == "CHEST" else ("skull fracture" if organ == "HEAD" else "renal calculus")
            else:
                target = rng.choice(ORGAN_NEGATIVES[organ][:4])
        else:
            strategy = "low_frequency"
            organ = rng.choice(other_organs)
            target = rng.choice(ORGAN_NEGATIVES[organ][4:])
        strategies[strategy] += 1
        negatives.append({
            **row, "question": f"Is there evidence of {target} in the image?",
            "answer": "no", "answer_type": "CLOSED", "_negative": True,
            "_source_term": term_by_qid[row["qid"]], "_replacement_term": target,
            "_replacement_strategy": strategy, "_target_organ": organ,
            "_validity_rule": "target finding belongs to a different annotated body compartment",
        })
    with (HERE / "semantic_negatives_cleanv2.jsonl").open("w") as out:
        for row in negatives:
            out.write(json.dumps(row) + "\n")
    (HERE / "build_manifest_cleanv2.json").write_text(json.dumps({
        "source_rows": len(source), "kept_rows": len(kept), "rejected": dict(reasons),
        "test_rows": len(test_source), "test_region_policy": dict(test_reasons),
        "negative_rows": len(negatives), "negative_strategy_counts": dict(strategies),
        "policy": "canonical cross-organ answer-flipped negatives",
        "min_top1_score": 0.35, "min_top1_top3_margin": 0.025,
        "box_expansion": 3.0, "background_brightness": 0.6}, indent=2))
    print({"source": len(source), "kept": len(kept), "rejected": dict(reasons)})


if __name__ == "__main__":
    main()
