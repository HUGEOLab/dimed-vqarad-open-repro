#!/usr/bin/env python3
"""Audit open-ended VQA-RAD predictions with a local, deterministic LLM judge.

The judge is intentionally strict: a prediction must express the same clinical
fact as the reference.  Abbreviations and harmless article/word-order changes
are accepted; wrong laterality, location, modality/sequence, measurement, or
missing required findings are rejected.  The output keeps every decision for
manual review and never changes model predictions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def is_open(row: dict) -> bool:
    return str(row["answer"]).strip().lower() not in {"yes", "no"}


def make_prompt(row: dict) -> str:
    return f"""You are auditing a medical visual-question-answering benchmark.
Decide whether the candidate answer is semantically equivalent to the reference
answer for the question. Accept standard abbreviations, synonyms, singular/plural,
and harmless omitted articles. Reject wrong laterality, anatomy, location,
modality/sequence, measurement, polarity, or a partial answer that omits a
required finding. Do not infer facts beyond the two answers. Output exactly 1
for equivalent or 0 for not equivalent.

Question: {row['question']}
Reference answer: {row['answer']}
Candidate answer: {row.get('response', '')}
Decision:"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("output_json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = json.loads(Path(args.results_json).read_text())
    open_rows = [(i, r) for i, r in enumerate(rows) if is_open(r)]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()

    audit = []
    for start in range(0, len(open_rows), args.batch_size):
        batch = open_rows[start : start + args.batch_size]
        texts = []
        for _, row in batch:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": make_prompt(row)}],
                }
            ]
            texts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = processor(
            text=texts, padding=True, return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        for (idx, row), raw in zip(batch, decoded):
            decisions = re.findall(r"(?<!\d)[01](?!\d)", raw)
            token = decisions[-1] if decisions else ""
            equivalent = token == "1"
            audit.append(
                {
                    "index": idx,
                    "question": row["question"],
                    "reference": row["answer"],
                    "candidate": row.get("response", ""),
                    "equivalent": equivalent,
                    "judge_raw": raw.strip(),
                }
            )
        print(f"judged {min(start + args.batch_size, len(open_rows))}/{len(open_rows)}", flush=True)

    payload = {
        "judge_model": args.model,
        "open_n": len(audit),
        "open_correct": sum(x["equivalent"] for x in audit),
        "items": audit,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in payload.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
