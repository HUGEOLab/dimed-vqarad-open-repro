#!/usr/bin/env python3
"""Generate the paper's pattern branch with local Lingshu-7B (train only)."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

VQARAD = Path(os.environ.get("VQARAD_DIR", str(Path(__file__).resolve().parents[2] / "data/vqarad")))
TRAIN = VQARAD / "trainset.json"
MODEL = ROOT / "projects/causal_grpo/reproduce_dimed/Lingshu-7B"
OUT = Path(__file__).resolve().parent / "pattern_train_lingshu.jsonl"

SYSTEM = """You standardize radiology VQA questions to remove wording shortcuts.
Return exactly one question and no explanation.
For YES/NO questions, prefer the canonical form 'Is there [entity or condition] in the image?' when that preserves the exact meaning and polarity.
For open or multiple-choice questions (what, which, where, how many, modality, plane, size, color, description, or explicit A-or-B choices), that yes/no template cannot preserve the answer. Keep the original question type, every offered choice, and requested attribute, but rewrite it into concise neutral wording.
Never add the known answer, never change negation, laterality, anatomy, modality, or the requested attribute."""


def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def clean(value):
    value = value.strip().splitlines()[0].strip().strip('"')
    value = re.sub(r"^(rewritten question|rewrite|question)\s*:\s*", "", value, flags=re.I)
    return value.strip()


def semantic_type(row):
    return "BINARY" if str(row["answer"]).strip().lower() in {"yes", "no"} else "OPEN_OR_CHOICE"


def validate(original, rewritten, question_kind):
    if len(rewritten) < 6 or norm(rewritten) == norm(original):
        return False, "empty_or_unchanged"
    first = norm(rewritten).split()[0]
    closed_starts = {"is", "are", "does", "do", "can", "has", "have", "was", "were", "could", "would"}
    open_starts = {"what", "which", "where", "how", "why", "who", "when", "describe", "name", "identify"}
    if question_kind == "BINARY" and first not in closed_starts:
        return False, "closed_became_nonbinary"
    # Multiple-choice questions can grammatically start with “is/are”; only reject
    # a conversion of a genuine wh-question into binary form.
    orig_first = norm(original).split()[0]
    if question_kind != "BINARY" and orig_first in open_starts and first in closed_starts:
        return False, "open_type_not_preserved"
    original_semantic = re.sub(r"[-( ]*yes\s*/\s*no[) ]*$", "", original, flags=re.I)
    neg_orig = bool(re.search(r"\b(no|not|without|absent)\b", original_semantic, re.I))
    neg_new = bool(re.search(r"\b(no|not|without|absent)\b", rewritten, re.I))
    if neg_orig != neg_new:
        return False, "negation_changed"
    # A common self-rewrite failure reverses a normality question while leaving
    # its gold yes/no label unchanged.
    normal_orig = bool(re.search(r"\bnormal\b", original, re.I)) and not bool(re.search(r"\babnormal\b", original, re.I))
    normal_new = bool(re.search(r"\bnormal\b", rewritten, re.I)) and not bool(re.search(r"\babnormal\b", rewritten, re.I))
    abnormal_orig = bool(re.search(r"\babnormal\b", original, re.I))
    abnormal_new = bool(re.search(r"\babnormal\b", rewritten, re.I))
    if (normal_orig and abnormal_new) or (abnormal_orig and normal_new):
        return False, "normality_reversed"
    # Do not collapse an explicit alternative into one side of the choice.
    contrast_pairs = [("left", "right"), ("hyperintense", "hypointense"),
                      ("hyperdense", "hypodense"), ("anterior", "posterior"),
                      ("superior", "inferior"), ("normal", "abnormal")]
    for a, b in contrast_pairs:
        if re.search(rf"\b{a}\b.*\b{b}\b|\b{b}\b.*\b{a}\b", original, re.I):
            if not (re.search(rf"\b{a}\b", rewritten, re.I) and re.search(rf"\b{b}\b", rewritten, re.I)):
                return False, "choice_collapsed"
    return True, "accepted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    rows = json.loads(TRAIN.read_text())

    # One deterministic generation per unique question; duplicate QA frames share it.
    unique = {}
    for row in rows:
        key = (norm(row["question"]), semantic_type(row))
        unique.setdefault(key, row)
    items = list(unique.items())
    print(f"loading Lingshu-7B; {len(items)} unique training questions", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device,
        attn_implementation="flash_attention_2",
    ).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    tok.padding_side = "left"
    generated = {}

    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        texts = []
        for _, row in batch:
            user = f"TYPE: {semantic_type(row)}\nORIGINAL: {row['question']}"
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tok(texts, return_tensors="pt", padding=True).to(args.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=48, do_sample=False, use_cache=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for (key, row), output in zip(batch, outputs):
            rewritten = clean(tok.decode(output[prompt_len:], skip_special_tokens=True))
            accepted, reason = validate(row["question"], rewritten, semantic_type(row))
            generated[key] = (rewritten if accepted else row["question"], accepted, reason, rewritten)
        if start == 0 or (start + len(batch)) % 256 == 0:
            print(f"generated {start + len(batch)}/{len(items)}", flush=True)

    accepted_count = 0
    with OUT.open("w") as out:
        for row in rows:
            key = (norm(row["question"]), semantic_type(row))
            rewrite, accepted, reason, raw = generated[key]
            accepted_count += int(accepted)
            out.write(json.dumps({
                **row,
                "q_rewrite": rewrite,
                "_rewrite_accepted": accepted,
                "_rewrite_reason": reason,
                "_rewrite_raw": raw,
            }) + "\n")
    summary = {"rows": len(rows), "unique_questions": len(items), "accepted_rows": accepted_count,
               "fallback_rows": len(rows) - accepted_count, "model": "Lingshu-7B"}
    (OUT.parent / "pattern_manifest.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
