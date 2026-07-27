#!/usr/bin/env python3
"""Score VQA-RAD generations under explicit, reproducible split policies."""

import argparse
import os
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DIMED_REPRO_ROOT", str(HERE.parent)))
CLASSIC_TEST = Path(os.environ.get(
    "VQARAD_TESTSET", str(ROOT / "data/vqarad/testset.json")
))


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--testset", type=Path, default=CLASSIC_TEST)
    parser.add_argument(
        "--label-policy", choices=("classic", "paper"), default="classic",
        help=("classic uses original VQA-RAD answer_type (179/272); paper uses "
              "the common non-yes/no vs yes/no split (200/251), which is much "
              "closer to the weighting implied by DiMed Table I."),
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    classic = json.loads(args.testset.read_text())
    labels = defaultdict(set)
    for row in classic:
        labels[(norm(row["question"]), norm(row["answer"]))].add(
            str(row["answer_type"]).strip().upper()
        )

    scored = []
    for index, row in enumerate(results):
        key = (norm(row["question"]), norm(row["answer"]))
        types = labels.get(key, set())
        if len(types) != 1:
            raise RuntimeError(f"Cannot uniquely align test row {index}: {key}, labels={types}")
        classic_type = next(iter(types))
        answer_type = (
            classic_type if args.label_policy == "classic"
            else ("CLOSED" if norm(row["answer"]) in {"yes", "no"} else "OPEN")
        )
        scored.append({
            "index": index,
            "answer_type": answer_type,
            "correct": norm(row["answer"]) == norm(row["response"]),
        })

    counts = Counter(x["answer_type"] for x in scored)
    correct = Counter(x["answer_type"] for x in scored if x["correct"])
    payload = {
        "protocol": (
            "classic VQA-RAD original answer_type; normalized exact match"
            if args.label_policy == "classic"
            else "DiMed-compatible non-yes/no OPEN split; normalized exact match"
        ),
        "total": len(scored),
        "counts": dict(counts),
        "correct": dict(correct),
        "accuracy": {k: correct[k] / counts[k] for k in counts},
        "overall_exact": sum(x["correct"] for x in scored) / len(scored),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
