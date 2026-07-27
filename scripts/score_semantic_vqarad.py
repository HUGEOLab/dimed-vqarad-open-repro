#!/usr/bin/env python3
"""Combine a non-yes/no semantic audit with official VQA-RAD type labels.

The semantic judge covers all non-yes/no answers. The remaining yes/no rows
use normalized exact match, for which semantic equivalence is unambiguous.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("semantic_audit", type=Path)
    ap.add_argument("--testset", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = json.loads(args.results.read_text())
    audit = json.loads(args.semantic_audit.read_text())
    judged = {int(x["index"]): bool(x["equivalent"]) for x in audit["items"]}
    labels = defaultdict(set)
    for row in json.loads(args.testset.read_text()):
        labels[(norm(row["question"]), norm(row["answer"]))].add(
            str(row["answer_type"]).strip().upper()
        )
    counts, correct = Counter(), Counter()
    semantic_n = yesno_n = 0
    for index, row in enumerate(rows):
        types = labels[(norm(row["question"]), norm(row["answer"]))]
        if len(types) != 1:
            raise RuntimeError(f"Non-unique type at result index {index}: {types}")
        answer_type = next(iter(types)); counts[answer_type] += 1
        if norm(row["answer"]) in {"yes", "no"}:
            equivalent = norm(row["answer"]) == norm(row.get("response", ""))
            yesno_n += 1
        else:
            if index not in judged:
                raise RuntimeError(f"Missing semantic decision for index {index}")
            equivalent = judged[index]; semantic_n += 1
        if equivalent:
            correct[answer_type] += 1
    total_correct = sum(correct.values())
    payload = {
        "protocol": "official VQA-RAD answer_type labels",
        "methods": {"semantic_judged_non_yesno": semantic_n,
                    "normalized_exact_yesno": yesno_n},
        "counts": dict(counts), "correct": dict(correct),
        "accuracy": {key: correct[key] / counts[key] for key in counts},
        "overall_correct": total_correct, "total": len(rows),
        "overall_semantic": total_correct / len(rows),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
