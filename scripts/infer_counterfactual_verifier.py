#!/usr/bin/env python3
"""Paper-faithful top-3 region counterfactual decoding on classic VQA-RAD."""
from __future__ import annotations

import argparse
import os
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DIMED_REPRO_ROOT", str(HERE.parent)))
DATA = Path(os.environ.get(
    "VQARAD_DIR", str(ROOT / "data/vqarad")
))
ARTIFACT_DIR = Path(os.environ.get("DIMED_ARTIFACT_DIR", str(HERE)))


def prompt_for(row):
    q = row["question"]
    # Original VQA-RAD marks 21 explicit A-or-B questions as CLOSED; they must
    # not be forced into yes/no output. This decision uses question text only.
    if row["answer_type"].strip() == "CLOSED" and not re.search(r"\bor\b", q, re.I):
        return q + "\nPlease output 'yes' or 'no'(no extra output)."
    return q + "\nPlease answer the question concisely."


def mask_box(image, box):
    masked = image.copy()
    ImageDraw.Draw(masked).rectangle(box, fill=(0, 0, 0))
    return masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="NONE")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tag", choices=("literal", "min2", "cleanv2", "cleanv3", "lingshuv4", "selectivev5"), default="min2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--baseline_only", action="store_true",
                    help="Generate the unverified checkpoint baseline without extra masked forwards.")
    ap.add_argument("--baseline-input", type=Path,
                    help="Reuse saved baseline responses; only intervention rows run new forwards.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = json.loads((DATA / "testset.json").read_text())
    regions = [json.loads(x) for x in (ARTIFACT_DIR / f"ctgm_test_{args.tag}.jsonl").read_text().splitlines()]
    assert [x["qid"] for x in rows] == [x["qid"] for x in regions]
    print("loading model", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=args.device, attn_implementation="flash_attention_2",
    )
    if args.adapter != "NONE":
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    proc = AutoProcessor.from_pretrained(args.base)
    tok = proc.tokenizer
    baseline_lookup = {}
    if args.baseline_input:
        baseline_lookup = {
            str(x["qid"]): x for x in json.loads(args.baseline_input.read_text())
        }

    def prepare(image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return proc(text=text, images=[image], return_tensors="pt").to(args.device)

    def with_suffix(seed, token_ids):
        """Append generated text while keeping every token-aligned Qwen field aligned."""
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
            if value.ndim == 2 and value.shape[0] == 1 and value.shape[1] == prompt_len:
                fed[key] = torch.cat([value, torch.zeros_like(suffix)], dim=1)
        return fed

    @torch.inference_mode()
    def baseline(image, prompt):
        inputs = prepare(image, prompt)
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, use_cache=True)
        token_ids = generated[0, inputs["input_ids"].shape[1]:]
        # Remove terminal/padding tokens for likelihood accounting.
        keep = [int(x) for x in token_ids if int(x) not in {tok.eos_token_id, tok.pad_token_id}]
        return tok.decode(keep, skip_special_tokens=True).strip(), keep

    @torch.inference_mode()
    def sequence_logp(image, prompt, answer_ids):
        inputs = prepare(image, prompt)
        prompt_len = inputs["input_ids"].shape[1]
        answer = torch.tensor([answer_ids], device=args.device)
        fed = with_suffix(inputs, answer_ids)
        # A single teacher-forced pass is both exact and avoids corrupting the
        # multimodal Qwen KV cache with text-only incremental calls.
        logits = model(**fed, use_cache=False).logits[
            0, prompt_len - 1: prompt_len - 1 + len(answer_ids)
        ].float()
        target = answer[0]
        return float(F.log_softmax(logits, dim=-1).gather(1, target[:, None]).sum())

    @torch.inference_mode()
    def contrastive_decode(image, masked, prompt):
        full_inputs, mask_inputs = prepare(image, prompt), prepare(masked, prompt)
        answer_ids = []
        for _ in range(args.max_new_tokens):
            def next_logits(seed):
                fed = dict(seed)
                if answer_ids:
                    fed = with_suffix(seed, answer_ids)
                return model(**fed, use_cache=False).logits[:, -1, :]
            full_logits, mask_logits = next_logits(full_inputs), next_logits(mask_inputs)
            # Eq. 8/9: log p_full + alpha * (log p_full - log p_mask).
            guided = (1.0 + args.alpha) * full_logits - args.alpha * mask_logits
            token_id = int(guided.argmax(dim=-1))
            if token_id in {tok.eos_token_id, tok.pad_token_id}:
                break
            answer_ids.append(token_id)
        return tok.decode(answer_ids, skip_special_tokens=True).strip()

    completed = {}
    if args.resume and args.output.exists():
        completed = {str(x["qid"]): x for x in json.loads(args.output.read_text())}
    for index, (row, region) in enumerate(zip(rows, regions), 1):
        if str(row["qid"]) in completed:
            continue
        image = Image.open(DATA / "images" / row["image_name"]).convert("RGB")
        prompt = prompt_for(row)
        cached_base = baseline_lookup.get(str(row["qid"]))
        if cached_base is not None:
            base_answer = cached_base.get("baseline_response", cached_base["response"])
            answer_ids = cached_base.get("answer_token_ids")
            if answer_ids is None:
                answer_ids = tok.encode(base_answer, add_special_tokens=False)
        else:
            base_answer, answer_ids = baseline(image, prompt)
        if args.baseline_only:
            completed[str(row["qid"])] = {
                "qid": row["qid"], "question": row["question"], "answer": row["answer"],
                "answer_type": row["answer_type"].strip(), "response": base_answer,
                "baseline_response": base_answer, "clinical_term": region["clinical_term"],
                "answer_token_ids": answer_ids,
                "region_drops": [], "selected_region": None, "max_drop": None,
                "visually_grounded": None,
            }
            if index <= 5 or index % 25 == 0:
                print(f"[{index}/{len(rows)}] baseline={base_answer!r}", flush=True)
            if index % 10 == 0:
                ordered = [completed[str(x["qid"])] for x in rows if str(x["qid"]) in completed]
                args.output.write_text(json.dumps(ordered, indent=2))
            continue
        drops = []
        masked_images = []
        if region["top_patch_boxes"]:
            full_logp = sequence_logp(image, prompt, answer_ids) if answer_ids else 0.0
            for box in region["top_patch_boxes"]:
                masked = mask_box(image, box)
                masked_images.append(masked)
                drops.append(full_logp - sequence_logp(masked, prompt, answer_ids))
        if drops:
            best = max(range(len(drops)), key=drops.__getitem__)
            max_drop = drops[best]
            # Eq. 13 / Fig. 3: only an answer whose probability changes enough
            # under the regional intervention is accepted as visually grounded.
            # Low-confidence CTGM evidence must not overwrite the baseline.
            response = (
                contrastive_decode(image, masked_images[best], prompt)
                if max_drop > args.tau else base_answer
            )
        else:
            best, max_drop, response = None, None, base_answer
        completed[str(row["qid"])] = {
            "qid": row["qid"], "question": row["question"], "answer": row["answer"],
            "answer_type": row["answer_type"].strip(), "response": response,
            "baseline_response": base_answer, "clinical_term": region["clinical_term"],
            "answer_token_ids": answer_ids,
            "region_drops": drops, "selected_region": best, "max_drop": max_drop,
            "visually_grounded": bool(max_drop is not None and max_drop > args.tau),
        }
        if index <= 5 or index % 25 == 0:
            print(f"[{index}/{len(rows)}] {base_answer!r} -> {response!r}; drops={drops}", flush=True)
        if index % 10 == 0:
            ordered = [completed[str(x["qid"])] for x in rows if str(x["qid"]) in completed]
            args.output.write_text(json.dumps(ordered, indent=2))
    ordered = [completed[str(x["qid"])] for x in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ordered, indent=2))
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
