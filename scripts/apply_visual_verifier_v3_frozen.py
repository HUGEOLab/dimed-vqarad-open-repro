#!/usr/bin/env python3
"""Apply the already-frozen v3 selector to another checkpoint's features."""
import argparse
import json
from pathlib import Path

import joblib

from train_apply_visual_verifier_v3 import flatten, select_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.features.read_text())
    baseline = json.loads(args.baseline.read_text())
    model = joblib.load(args.model)
    calibration = json.loads(args.calibration.read_text())
    params = calibration["best"]
    X, _, _, refs = flatten(rows)
    probabilities = model.predict_proba(X)[:, 1]
    selections, _ = select_rows(
        rows, probabilities, refs, params["threshold"], params["margin"]
    )
    by_qid = {int(row["qid"]): (i, row) for i, row in enumerate(rows)}
    output, changes = [], []
    for base in baseline:
        item = dict(base)
        pair = by_qid.get(int(item["qid"]))
        if pair is None:
            item["verifier_v3_frozen_used"] = False
            output.append(item)
            continue
        row_i, row = pair
        selection = selections[row_i]
        candidate = row["candidates"][selection["candidate_i"]]
        item.update({
            "pre_verifier_v3_frozen_response": item["response"],
            "verifier_v3_frozen_used": selection["accepted"],
            "verifier_v3_frozen_candidate": candidate["answer"],
            "verifier_v3_frozen_source": candidate["source"],
            "verifier_v3_frozen_probability": selection["best_probability"],
            "verifier_v3_selector_origin": args.model.name,
        })
        if selection["accepted"]:
            item["response"] = candidate["answer"]
            changes.append({
                "qid": item["qid"], "question": item["question"], "gold": item["answer"],
                "before": item["pre_verifier_v3_frozen_response"],
                "after": item["response"], "source": candidate["source"],
                "probability": selection["best_probability"],
            })
        output.append(item)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    args.output.with_name(args.output.stem + "_audit.json").write_text(
        json.dumps({"selector_retrained": False, "test_labels_used": False,
                    "changes": len(changes), "changed_items": changes}, indent=2) + "\n"
    )
    print(json.dumps({"selector_retrained": False, "changes": len(changes)}, indent=2))


if __name__ == "__main__":
    main()
