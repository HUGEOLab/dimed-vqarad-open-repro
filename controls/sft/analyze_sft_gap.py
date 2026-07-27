#!/usr/bin/env python3
"""Reproduce the ordinary-SFT versus DiMed/openfocusv2 gap audit."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
RESULTS = HERE / "results"
PROVENANCE = HERE / "provenance"
METHOD_RESULTS = PACKAGE / "artifacts" / "results"


def load(path: Path):
    return json.loads(path.read_text())


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def exact_correct(row):
    return norm(row["answer"]) == norm(row["response"])


def exact_binomial_two_sided(a_wins: int, b_wins: int) -> float:
    """Exact two-sided sign/McNemar test on discordant pairs."""
    n = a_wins + b_wins
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_wins, b_wins) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(rows_a, rows_b, correct_a, correct_b):
    output = {}
    for answer_type in ("OPEN", "CLOSED", "ALL"):
        indices = range(len(rows_a)) if answer_type == "ALL" else (
            i for i, row in enumerate(rows_a)
            if str(row["answer_type"]).strip().upper() == answer_type
        )
        cc = ww = a_fix = b_fix = 0
        for i in indices:
            a = correct_a(i)
            b = correct_b(i)
            if a and b:
                cc += 1
            elif not a and not b:
                ww += 1
            elif a:
                a_fix += 1
            else:
                b_fix += 1
        output[answer_type] = {
            "both_correct": cc,
            "both_wrong": ww,
            "a_only_correct": a_fix,
            "b_only_correct": b_fix,
            "two_sided_exact_p": exact_binomial_two_sided(a_fix, b_fix),
        }
    return output


def semantic_corrector(rows, audit):
    judged = {int(x["index"]): bool(x["equivalent"]) for x in audit["items"]}

    def correct(index):
        row = rows[index]
        if norm(row["answer"]) in {"yes", "no"}:
            return exact_correct(row)
        return judged[index]

    return correct


def training_state(path: Path):
    state = load(path)
    losses = [x for x in state["log_history"] if "loss" in x]
    return {
        "global_step": state["global_step"],
        "epoch": state.get("epoch"),
        "first_logged_loss": losses[0]["loss"],
        "last_logged_loss": losses[-1]["loss"],
        "last_logged_grad_norm": losses[-1].get("grad_norm"),
    }


def main():
    result_paths = {
        "conventional_sft": RESULTS / "conventional_sft_model_only.json",
        "matched_sft": RESULTS / "matched_sft_model_only.json",
        "method_model_only": METHOD_RESULTS / "results_openfocusv2_baseline.json",
    }
    rows = {name: load(path) for name, path in result_paths.items()}
    assert all(len(value) == 451 for value in rows.values())

    audits = {
        "conventional_sft": load(RESULTS / "conventional_sft_model_only_semantic_audit.json"),
        "matched_sft": load(RESULTS / "matched_sft_model_only_semantic_audit.json"),
        "method_model_only": load(RESULTS / "results_openfocusv2_baseline_semantic_audit.json"),
    }

    exact_pair = paired(
        rows["matched_sft"], rows["method_model_only"],
        lambda i: exact_correct(rows["matched_sft"][i]),
        lambda i: exact_correct(rows["method_model_only"][i]),
    )
    semantic_pair = paired(
        rows["matched_sft"], rows["method_model_only"],
        semantic_corrector(rows["matched_sft"], audits["matched_sft"]),
        semantic_corrector(rows["method_model_only"], audits["method_model_only"]),
    )
    sft_semantic = semantic_corrector(rows["matched_sft"], audits["matched_sft"])
    method_semantic = semantic_corrector(
        rows["method_model_only"], audits["method_model_only"]
    )
    method_open_exact_fixes = []
    for i, (sft_row, method_row) in enumerate(zip(
        rows["matched_sft"], rows["method_model_only"]
    )):
        if str(sft_row["answer_type"]).strip().upper() != "OPEN":
            continue
        if not exact_correct(sft_row) and exact_correct(method_row):
            method_open_exact_fixes.append({
                "index": i,
                "question": sft_row["question"],
                "reference": sft_row["answer"],
                "matched_sft": sft_row["response"],
                "method": method_row["response"],
                "matched_sft_semantically_correct": sft_semantic(i),
                "method_semantically_correct": method_semantic(i),
            })

    metric_files = {
        "conventional_sft_exact": RESULTS / "conventional_sft_model_only_exact.json",
        "conventional_sft_semantic": RESULTS / "conventional_sft_model_only_semantic_metrics.json",
        "matched_sft_stage1_exact": RESULTS / "matched_sft_stage1_model_only_exact.json",
        "matched_sft_exact": RESULTS / "matched_sft_model_only_exact.json",
        "matched_sft_semantic": RESULTS / "matched_sft_model_only_semantic_metrics.json",
        "method_model_only_exact": METHOD_RESULTS / "results_openfocusv2_baseline_metrics_classic.json",
        "method_model_only_semantic": RESULTS / "results_openfocusv2_baseline_semantic_metrics.json",
        "method_verifier_exact": METHOD_RESULTS / "results_openfocusv2_verifier_v3_v7_metrics_classic.json",
        "method_verifier_semantic": METHOD_RESULTS / "results_openfocusv2_verifier_v3_v7_semantic_official.json",
    }

    payload = {
        "metrics": {name: load(path) for name, path in metric_files.items()},
        "paired_matched_sft_vs_method_model_only": {
            "exact": exact_pair,
            "semantic": semantic_pair,
            "open_exact_fix_decomposition": {
                "method_exact_fixes": len(method_open_exact_fixes),
                "already_semantically_correct_for_sft": sum(
                    x["matched_sft_semantically_correct"]
                    for x in method_open_exact_fixes
                ),
                "semantic_or_factual_fixes": sum(
                    not x["matched_sft_semantically_correct"]
                    and x["method_semantically_correct"]
                    for x in method_open_exact_fixes
                ),
                "items": method_open_exact_fixes,
            },
        },
        "training_diagnostics": {
            "conventional_sft": training_state(
                PROVENANCE / "conventional_sft_trainer_state.json"
            ),
            "matched_sft_stage1": training_state(
                PROVENANCE / "matched_sft_stage1_trainer_state.json"
            ),
            "matched_sft_stage2": training_state(
                PROVENANCE / "matched_sft_stage2_trainer_state.json"
            ),
            "method_e2": training_state(
                PROVENANCE / "method_e2_trainer_state.json"
            ),
            "method_openfocusv2": training_state(
                PROVENANCE / "method_openfocusv2_trainer_state.json"
            ),
        },
        "corpus_audit": {
            "matched_sft_stage1": {
                "rows": 3064,
                "composition": {"original": 3064},
                "effective_epochs_at_1182_steps": 6.157,
                "open_rows_per_epoch": 1241,
                "approx_effective_open_presentations": 7641,
            },
            "method_e2_at_training_time": {
                "rows": 9445,
                "composition": {
                    "original": 3064,
                    "pattern_rewrite": 2822,
                    "ctgm_effective": 2964,
                    "semantic_negative": 595,
                },
                "effective_epochs_at_1182_steps": 2.0,
                "open_rows_per_epoch": 3573,
                "effective_open_presentations": 7146,
                "note": "Historical corpus count; CTGM artifacts were revised after this run.",
            },
            "matched_sft_stage2": {
                "rows": 1412,
                "composition": {"original_open_plus_10pct_closed_anchor": 1412},
                "effective_epochs_at_250_steps": 2.816,
            },
            "method_openfocusv2": {
                "rows": 3992,
                "composition": {
                    "original_open_plus_10pct_closed_anchor": 1412,
                    "open_pattern": 1088,
                    "open_invariant_pair": 746,
                    "ctgm_grounded": 746,
                },
                "effective_epochs_at_250_steps": 1.0,
            },
        },
        "interpretation_limits": [
            "The comparison isolates the full augmented recipe, not CTGM alone.",
            "Exact-match superiority is stronger than semantic superiority.",
            "The paired semantic difference is not statistically significant at alpha=0.05.",
            "The verifier includes same-image train retrieval and must be reported separately.",
        ],
    }
    out = HERE / "sft_gap_analysis.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("WROTE controls/sft/sft_gap_analysis.json")


if __name__ == "__main__":
    main()
