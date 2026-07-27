#!/usr/bin/env python3
"""Extract answer candidates and visual-support features for verifier v2.

The extractor never uses a gold answer to generate or score a candidate. Gold
is retained in the output only so TRAIN features can calibrate a frozen gate
and TEST results can be evaluated after that gate is frozen.
"""
from __future__ import annotations

import argparse
import os
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForImageTextToText, AutoProcessor


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DIMED_REPRO_ROOT", str(HERE.parent)))
DATA = Path(os.environ.get(
    "VQARAD_DIR", str(ROOT / "data/vqarad")
))
ARTIFACT_DIR = Path(os.environ.get("DIMED_ARTIFACT_DIR", str(HERE)))


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def prompt_for(question):
    return question + "\nPlease answer the question concisely."


def regional_views(image, boxes, dim=0.2):
    grounded = ImageEnhance.Brightness(image).enhance(dim)
    removed = image.copy()
    draw = ImageDraw.Draw(removed)
    width, height = image.size
    valid = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
        y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        valid.append((x1, y1, x2, y2))
        grounded.paste(image.crop((x1, y1, x2, y2)), (x1, y1))
        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
    if not valid:
        return grounded, removed, image
    ux1, uy1 = min(x[0] for x in valid), min(x[1] for x in valid)
    ux2, uy2 = max(x[2] for x in valid), max(x[3] for x in valid)
    pad_x, pad_y = round((ux2 - ux1) * 0.15), round((uy2 - uy1) * 0.15)
    crop = image.crop((max(0, ux1 - pad_x), max(0, uy1 - pad_y),
                       min(width, ux2 + pad_x), min(height, uy2 + pad_y)))
    return grounded, removed, crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline", type=Path,
                    help="Saved full-image baseline; recommended for test.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top_retrieval", type=int, default=3)
    ap.add_argument("--top_global", type=int, default=2)
    ap.add_argument("--dim", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    train = json.loads((DATA / "trainset.json").read_text())
    rows = json.loads((DATA / f"{args.split}set.json").read_text())
    if args.split == "train":
        region_rows = [
            json.loads(line) for line in
            (ARTIFACT_DIR / "ctgm_train_openfocusv1.jsonl").read_text().splitlines()
        ]
    else:
        region_rows = [
            json.loads(line) for line in
            (ARTIFACT_DIR / "ctgm_test_selectivev5.jsonl").read_text().splitlines()
        ]
    regions = {int(row["qid"]): row for row in region_rows if row.get("top_patch_boxes")}
    rows = [row for row in rows if int(row["qid"]) in regions and norm(row["answer"]) not in {"yes", "no"}]
    if args.max_rows:
        rows = rows[:args.max_rows]

    baseline = {}
    if args.baseline:
        baseline = {int(x["qid"]): x for x in json.loads(args.baseline.read_text())}

    open_train = [row for row in train if norm(row["answer"]) not in {"yes", "no"}]
    by_image = defaultdict(list)
    for row in open_train:
        by_image[row["image_name"]].append(row)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1
    ).fit([row["question"] for row in open_train])
    global_question_matrix = vectorizer.transform([row["question"] for row in open_train])

    print("loading model", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=args.device, attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    proc = AutoProcessor.from_pretrained(args.base)
    tok = proc.tokenizer

    def prepare(image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return proc(text=text, images=[image], return_tensors="pt").to(args.device)

    def with_suffix(seed, token_ids):
        suffix = torch.tensor([token_ids], device=args.device)
        fed = dict(seed)
        fed["input_ids"] = torch.cat([seed["input_ids"], suffix], dim=1)
        fed["attention_mask"] = torch.cat(
            [seed["attention_mask"], torch.ones_like(suffix)], dim=1
        )
        prompt_len = seed["input_ids"].shape[1]
        for key, value in seed.items():
            if key in {"input_ids", "attention_mask"} or not torch.is_tensor(value):
                continue
            if value.ndim == 2 and value.shape == seed["input_ids"].shape:
                fed[key] = torch.cat([value, torch.zeros_like(suffix)], dim=1)
        return fed, prompt_len

    @torch.inference_mode()
    def generate(seed):
        output = model.generate(
            **seed, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
        )
        ids = output[0, seed["input_ids"].shape[1]:]
        return tok.decode(ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def contrastive_generate(full_seed, removed_seed, alpha=1.0):
        answer_ids = []
        for _ in range(args.max_new_tokens):
            def next_logits(seed):
                fed = seed if not answer_ids else with_suffix(seed, answer_ids)[0]
                return model(**fed, use_cache=False).logits[:, -1, :]
            full_logits = next_logits(full_seed)
            removed_logits = next_logits(removed_seed)
            guided = (1.0 + alpha) * full_logits - alpha * removed_logits
            token_id = int(guided.argmax(dim=-1))
            if token_id in {tok.eos_token_id, tok.pad_token_id}:
                break
            answer_ids.append(token_id)
        return tok.decode(answer_ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def mean_logp(seed, answer):
        ids = tok.encode(answer, add_special_tokens=False)
        if not ids:
            return -1e9
        fed, prompt_len = with_suffix(seed, ids)
        logits = model(**fed, use_cache=False).logits[
            0, prompt_len - 1: prompt_len - 1 + len(ids)
        ].float()
        target = torch.tensor(ids, device=args.device)
        total = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).sum()
        return float(total / len(ids))

    saved = {}
    if args.resume and args.output.exists():
        saved = {int(x["qid"]): x for x in json.loads(args.output.read_text())}

    for index, row in enumerate(rows, 1):
        qid = int(row["qid"])
        if qid in saved:
            continue
        image = Image.open(DATA / "images" / row["image_name"]).convert("RGB")
        grounded, removed, crop = regional_views(
            image, regions[qid]["top_patch_boxes"], args.dim
        )
        prompt = prompt_for(row["question"])
        full_seed = prepare(image, prompt)
        ground_seed = prepare(grounded, prompt)
        removed_seed = prepare(removed, prompt)
        crop_seed = prepare(crop, prompt)
        focus_prompt = (
            "The image background is suppressed. Focus on the preserved clinical "
            "region and answer the original question concisely.\nOriginal question: "
            + row["question"]
        )
        ground_focus_seed = prepare(grounded, focus_prompt)
        crop_focus_seed = prepare(crop, focus_prompt)
        base_answer = (
            str(baseline[qid]["response"]).strip() if qid in baseline
            else generate(full_seed)
        )
        generated = {
            "baseline": base_answer,
            "ground_generation": generate(ground_focus_seed),
            "crop_generation": generate(crop_focus_seed),
            "contrastive_generation": contrastive_generate(full_seed, removed_seed),
        }

        facts = [x for x in by_image[row["image_name"]] if int(x["qid"]) != qid]
        retrieved = []
        if facts:
            matrix = vectorizer.transform(
                [row["question"]] + [x["question"] for x in facts]
            )
            sims = cosine_similarity(matrix[:1], matrix[1:])[0]
            for fact_i in sims.argsort()[::-1][:args.top_retrieval]:
                fact = facts[int(fact_i)]
                retrieved.append({
                    "answer": str(fact["answer"]),
                    "source": "same_image_train",
                    "similarity": float(sims[int(fact_i)]),
                    "source_qid": int(fact["qid"]),
                    "source_question": fact["question"],
                })
        query = vectorizer.transform([row["question"]])
        global_sims = cosine_similarity(query, global_question_matrix)[0]
        global_added = 0
        for fact_i in global_sims.argsort()[::-1]:
            fact = open_train[int(fact_i)]
            if int(fact["qid"]) == qid or fact["image_name"] == row["image_name"]:
                continue
            retrieved.append({
                "answer": str(fact["answer"]),
                "source": "global_train",
                "similarity": float(global_sims[int(fact_i)]),
                "source_qid": int(fact["qid"]),
                "source_question": fact["question"],
            })
            global_added += 1
            if global_added >= args.top_global:
                break

        candidates, seen = [], set()
        for source, answer in generated.items():
            key = norm(answer)
            if answer and key not in seen:
                candidates.append({"answer": answer, "source": source, "similarity": None})
                seen.add(key)
        for item in retrieved:
            key = norm(item["answer"])
            if key and key not in seen:
                candidates.append(item)
                seen.add(key)

        for item in candidates:
            item["full"] = mean_logp(full_seed, item["answer"])
            item["grounded"] = mean_logp(ground_seed, item["answer"])
            item["removed"] = mean_logp(removed_seed, item["answer"])
            item["visual_support"] = item["grounded"] - item["removed"]
        saved[qid] = {
            "qid": qid, "split": args.split, "image_name": row["image_name"],
            "question": row["question"], "answer": row["answer"],
            "answer_type": row["answer_type"], "baseline_response": base_answer,
            "clinical_term": regions[qid].get("clinical_term"),
            "boxes": regions[qid]["top_patch_boxes"], "candidates": candidates,
            "gold_used_for_candidates_or_features": False,
        }
        if index <= 5 or index % 10 == 0:
            print(f"[{index}/{len(rows)}] qid={qid} base={base_answer!r} "
                  f"ground={generated['ground_generation']!r} n={len(candidates)}", flush=True)
        if index % 5 == 0:
            args.output.write_text(json.dumps(list(saved.values()), indent=2))
    args.output.write_text(json.dumps(list(saved.values()), indent=2))
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
