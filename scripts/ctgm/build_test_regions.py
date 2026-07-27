#!/usr/bin/env python3
"""Select verifier regions from test questions without consulting test answers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from build_biomedclip_grounding import (
    DATA, HERE, IMAGES, encode_terms, extract_entities, load_biomedclip,
    patch_box, patch_features,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=("literal", "min2"), default="min2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()
    rows = json.loads((DATA / "testset.json").read_text())
    # Answers are present in the public test file but are intentionally never read.
    question_only = [{"question": r["question"]} for r in rows]
    entities = extract_entities(question_only)
    confounders = json.loads((HERE / f"confounders_classic_{args.tag}.json").read_text())
    # At inference the answer is unknown. Rank a term by its strongest TRAIN-only PMI,
    # breaking ties by support. The paper does not specify this selection operation.
    term_strength = {}
    for item in confounders:
        key = item["term"]
        value = (item["pmi"], item["joint_count"])
        term_strength[key] = max(term_strength.get(key, (-1.0, -1)), value)
    selected = []
    for terms in entities:
        hits = [t for t in terms if t in term_strength]
        if hits:
            selected.append(max(hits, key=lambda t: term_strength[t]))
        elif terms:
            selected.append(max(terms, key=len))
        else:
            selected.append(None)

    unique_terms = sorted({x for x in selected if x})
    model, preprocess, tokenizer = load_biomedclip(args.device)
    text_features = encode_terms(model, tokenizer, unique_terms, args.device)
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}
    patch_cache = {}
    output = []
    for row, term in zip(rows, selected):
        record = {"qid": row["qid"], "image_name": row["image_name"],
                  "question": row["question"], "clinical_term": term}
        if term is not None:
            if row["image_name"] not in patch_cache:
                patch_cache[row["image_name"]] = patch_features(
                    model, preprocess, IMAGES / row["image_name"], args.device
                )
            scores = patch_cache[row["image_name"]] @ text_features[term_to_idx[term]]
            values, indices = torch.topk(scores, args.top_k)
            from PIL import Image
            with Image.open(IMAGES / row["image_name"]) as image:
                width, height = image.size
            record.update({
                "top_patch_indices": indices.tolist(),
                "top_patch_scores": values.tolist(),
                "top_patch_boxes": [patch_box(i, width, height) for i in indices.tolist()],
            })
        else:
            record.update({"top_patch_indices": [], "top_patch_scores": [], "top_patch_boxes": []})
        output.append(record)
    target = HERE / f"ctgm_test_{args.tag}.jsonl"
    with target.open("w") as out:
        for row in output:
            out.write(json.dumps(row) + "\n")
    print({"rows": len(rows), "with_entity": sum(x is not None for x in selected),
           "unique_terms": len(unique_terms), "output": str(target)})


if __name__ == "__main__":
    main()
