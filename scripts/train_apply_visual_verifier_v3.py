#!/usr/bin/env python3
"""Train a source-aware visual candidate selector and apply it to test.

All model fitting and gate calibration use train rows only. Candidate-level
out-of-fold probabilities are grouped by source image to avoid QA duplicates
from the same image leaking across calibration folds.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCES = (
    "baseline", "ground_generation", "crop_generation",
    "contrastive_generation", "same_image_train", "global_train",
)


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def vector(row, candidate):
    base = next(x for x in row["candidates"] if x["source"] == "baseline")
    question_tokens = set(norm(row["question"]).split())
    answer_tokens = norm(candidate["answer"]).split()
    numeric = [
        candidate["full"], candidate["grounded"], candidate["removed"],
        candidate["visual_support"],
        candidate["full"] - base["full"],
        candidate["grounded"] - base["grounded"],
        candidate["removed"] - base["removed"],
        candidate["visual_support"] - base["visual_support"],
        candidate.get("similarity") if candidate.get("similarity") is not None else -1.0,
        len(answer_tokens),
        len(set(answer_tokens) & question_tokens) / max(1, len(set(answer_tokens))),
    ]
    return numeric + [float(candidate["source"] == source) for source in SOURCES]


def flatten(rows, exclude_baseline=True):
    X, y, groups, refs = [], [], [], []
    for row_i, row in enumerate(rows):
        for candidate_i, candidate in enumerate(row["candidates"]):
            if exclude_baseline and candidate["source"] == "baseline":
                continue
            X.append(vector(row, candidate))
            y.append(norm(candidate["answer"]) == norm(row["answer"]))
            groups.append(row["image_name"])
            refs.append((row_i, candidate_i))
    return np.asarray(X), np.asarray(y, dtype=int), np.asarray(groups), refs


def select_rows(rows, probabilities, refs, threshold, margin):
    by_row = defaultdict(list)
    for probability, (row_i, candidate_i) in zip(probabilities, refs):
        by_row[row_i].append((float(probability), candidate_i))
    selections, stats = {}, {"changes": 0, "fixed": 0, "broke": 0, "correct": 0}
    for row_i, row in enumerate(rows):
        ranked = sorted(by_row[row_i], reverse=True)
        base_i = next(i for i, x in enumerate(row["candidates"]) if x["source"] == "baseline")
        base_p = None
        if ranked:
            best_p, best_i = ranked[0]
            accept = best_i != base_i and best_p >= threshold
        else:
            best_p, best_i, accept = 0.0, base_i, False
        chosen_i = best_i if accept else base_i
        before = norm(row["baseline_response"]) == norm(row["answer"])
        after = norm(row["candidates"][chosen_i]["answer"]) == norm(row["answer"])
        stats["changes"] += accept
        stats["fixed"] += accept and after and not before
        stats["broke"] += accept and before and not after
        stats["correct"] += after
        selections[row_i] = {
            "candidate_i": chosen_i, "accepted": bool(accept),
            "best_probability": best_p, "baseline_probability": base_p,
        }
    stats["net"] = stats["fixed"] - stats["broke"]
    return selections, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", type=Path, required=True)
    ap.add_argument("--test-features", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--model-output", type=Path, required=True)
    args = ap.parse_args()
    train_rows = json.loads(args.train_features.read_text())
    test_rows = json.loads(args.test_features.read_text())
    baseline_rows = json.loads(args.baseline.read_text())
    X, y, groups, refs = flatten(train_rows)
    group_folds = GroupKFold(n_splits=5)

    trials = []
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        oof = np.zeros(len(y), dtype=float)
        for train_i, valid_i in group_folds.split(X, y, groups):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, class_weight="balanced", max_iter=3000),
            )
            model.fit(X[train_i], y[train_i])
            oof[valid_i] = model.predict_proba(X[valid_i])[:, 1]
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for margin in (0.0,):
                _, stats = select_rows(train_rows, oof, refs, threshold, margin)
                trials.append({"C": C, "threshold": threshold, "margin": margin, **stats})
    viable = [x for x in trials if x["changes"] >= 5]
    best = max(
        viable,
        key=lambda x: (x["correct"], x["net"], x["fixed"], -x["broke"], -x["changes"]),
    )
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best["C"], class_weight="balanced", max_iter=3000),
    )
    final_model.fit(X, y)
    joblib.dump(final_model, args.model_output)

    test_X, _, _, test_refs = flatten(test_rows)
    test_prob = final_model.predict_proba(test_X)[:, 1]
    selections, _ = select_rows(
        test_rows, test_prob, test_refs, best["threshold"], best["margin"]
    )
    feature_by_qid = {int(row["qid"]): (row_i, row) for row_i, row in enumerate(test_rows)}
    output, changes = [], []
    for base in baseline_rows:
        item = dict(base)
        pair = feature_by_qid.get(int(base["qid"]))
        if pair is None:
            item["verifier_v3_used"] = False
            output.append(item)
            continue
        row_i, row = pair
        selection = selections[row_i]
        candidate = row["candidates"][selection["candidate_i"]]
        item.update({
            "pre_verifier_v3_response": item["response"],
            "verifier_v3_used": selection["accepted"],
            "verifier_v3_candidate": candidate["answer"],
            "verifier_v3_candidate_source": candidate["source"],
            "verifier_v3_probability": selection["best_probability"],
            "verifier_v3_baseline_probability": selection["baseline_probability"],
        })
        if selection["accepted"]:
            item["response"] = candidate["answer"]
            changes.append({
                "qid": item["qid"], "question": item["question"],
                "gold": item["answer"], "before": item["pre_verifier_v3_response"],
                "after": item["response"], "source": candidate["source"],
                "probability": selection["best_probability"],
                "baseline_probability": selection["baseline_probability"],
            })
        output.append(item)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    payload = {
        "train_only": True, "grouping": "5-fold by image_name",
        "candidate_rows": len(train_rows), "candidate_examples": len(y),
        "positive_candidate_rate": float(y.mean()), "best": best,
        "top10": sorted(
            viable,
            key=lambda x: (x["correct"], x["net"], x["fixed"], -x["broke"], -x["changes"]),
            reverse=True,
        )[:10],
        "test_local_rows": len(test_rows), "test_changes_without_gold_selection": len(changes),
    }
    args.calibration.write_text(json.dumps(payload, indent=2) + "\n")
    args.output.with_name(args.output.stem + "_audit.json").write_text(
        json.dumps({"calibration": payload, "changed_items": changes}, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
