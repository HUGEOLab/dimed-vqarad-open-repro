#!/usr/bin/env python3
"""DiMed training (Phase B): LoRA fine-tune Lingshu-7B (Qwen2.5-VL) on VQA-RAD train
with the de-confounding multi-variant objective.

  L = L_original + lambda*(L_pattern [+ L_grounding])   (lambda=1; grounding = Phase C add-on)

Realized as a combined SFT corpus over TRAIN ONLY:
  - original  : (image, orig_Q, gold)            -> L_original
  - pattern   : (image, q_rewrite, gold)         -> L_pattern   (semantics-preserving paraphrase)
  - negatives : (image, term-swapped Q, 'no')    -> de-confounding negatives (breaks entity-answer PMI)

Prompts match MedEvalKit VQA_RAD eval templates so train/eval are consistent.
TEST SET IS NEVER USED.
"""
import os, json, random, argparse
import re
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import (
    LoraConfig, PeftModel, TaskType, get_peft_model,
    prepare_model_for_kbit_training,
)
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DIMED_REPRO_ROOT", str(HERE.parent)))
MODEL = ROOT / "models/Lingshu-7B"  # overridable via --model
IMG_DIR = ROOT / "data/vqarad/images"
DECONF = HERE / "deconf_train.jsonl"      # orig + negatives
PATTERN = HERE / "pattern_aug.jsonl"      # orig + q_rewrite
BBOX = ROOT / "artifacts/train/vqa_rad_train_with_bboxes.json"
CONF = HERE / "confounders.json"
CLASSIC_DIR = Path(os.environ.get(
    "VQARAD_DIR", str(ROOT / "data/vqarad")
))
CLASSIC_TRAIN = CLASSIC_DIR / "trainset.json"
CLASSIC_IMG_DIR = CLASSIC_DIR / "images"
FAITHFUL_DIR = Path(os.environ.get(
    "DIMED_ARTIFACT_DIR", str(HERE.parent / "faithful_dimed")
))
random.seed(42)

def norm_answer(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))

def open_prompt(q):   return q + "\n" + "Please answer the question concisely."
def yesno_prompt(q):  return q + "\n" + "Please output 'yes' or 'no'(no extra output)."
def prompt_for(q, ans):
    return yesno_prompt(q) if str(ans).lower().strip() in ("yes","no") else open_prompt(q)

def clinical_term_pattern(term):
    """Match normalized clinical terms against punctuation and simple plurals."""
    tokens = re.findall(r"[a-z0-9]+", str(term).lower())
    if not tokens:
        return None
    parts = []
    for token in tokens:
        if token.endswith("y") and len(token) > 3:
            parts.append(re.escape(token[:-1]) + r"(?:y|ys|ies)")
        elif token.endswith("s"):
            parts.append(re.escape(token))
        else:
            parts.append(re.escape(token) + r"s?")
    return r"\b" + r"(?:\W|_)+".join(parts) + r"\b"

def mask_clinical_term(question, terms):
    q = question
    candidates = []
    for term in terms:
        pattern = clinical_term_pattern(term)
        if pattern and re.search(pattern, q, re.I):
            candidates.append((len(str(term)), pattern))
    if not candidates:
        return None
    _, pattern = max(candidates)
    return re.sub(pattern, "[CLINICAL TERM]", q, count=1, flags=re.I)


def build_corpus(use_pattern=True, use_neg=True, use_ground=False, open_pattern_only=False,
                 classic_mode="off", closed_anchor_rate=0.0, faithful_tag="off",
                 ground_repeat=1, open_balance_cap=1, use_open_pairs=False,
                 faithful_open_only=False):
    if faithful_tag != "off":
        classic = json.loads(CLASSIC_TRAIN.read_text())
        rows = []
        for r in classic:
            is_open = str(r.get("answer_type", "")).strip().upper() == "OPEN"
            if faithful_open_only and not is_open and random.random() >= closed_anchor_rate:
                continue
            rows.append({
                "img_path": str(CLASSIC_IMG_DIR / r["image_name"]),
                "q": r["question"], "a": str(r["answer"]).strip(), "kind": "orig",
            })
        if use_neg:
            neg_path = FAITHFUL_DIR / f"semantic_negatives_{faithful_tag}.jsonl"
            for line in neg_path.read_text().splitlines():
                r = json.loads(line)
                rows.append({
                    "img_path": str(CLASSIC_IMG_DIR / r["image_name"]),
                    "q": r["question"], "a": str(r["answer"]).strip(), "kind": "neg",
                })
        pattern_path = FAITHFUL_DIR / "pattern_train_lingshu.jsonl"
        if use_pattern and pattern_path.exists():
            for line in pattern_path.read_text().splitlines():
                r = json.loads(line)
                if open_pattern_only and r.get("answer_type") != "OPEN":
                    continue
                if r.get("_rewrite_accepted"):
                    rows.append({
                        "img_path": str(CLASSIC_IMG_DIR / r["image_name"]),
                        "q": r["q_rewrite"], "a": str(r["answer"]).strip(), "kind": "pattern",
                    })
        if use_ground:
            ground_path = FAITHFUL_DIR / f"ctgm_train_{faithful_tag}.jsonl"
            for line in ground_path.read_text().splitlines():
                r = json.loads(line)
                masked = mask_clinical_term(r["question"], [r["clinical_term"]])
                if masked:
                    has_boxes = bool(r.get("top_patch_boxes"))
                    ground_row = {
                        "img_path": str(CLASSIC_IMG_DIR / r["image_name"]),
                        "q": masked, "a": str(r["answer"]).strip(),
                        "kind": "ground_local" if has_boxes else "ground_full",
                        "bboxes": r["top_patch_boxes"], "clinical_term": r["clinical_term"],
                    }
                    rows.extend(dict(ground_row) for _ in range(ground_repeat))
        if use_open_pairs:
            pair_path = FAITHFUL_DIR / f"open_invariant_pairs_{faithful_tag}.jsonl"
            if not pair_path.exists():
                raise FileNotFoundError(
                    f"--use_open_pairs requested but {pair_path} does not exist"
                )
            for line in pair_path.read_text().splitlines():
                r = json.loads(line)
                rows.append({
                    "img_path": str(CLASSIC_IMG_DIR / r["image_name"]),
                    "q": r["question"], "a": str(r["answer"]).strip(),
                    "kind": "open_pair", "pair_id": r.get("_pair_id"),
                })
        random.shuffle(rows)
        return rows

    if classic_mode != "off":
        classic = json.loads(CLASSIC_TRAIN.read_text())
        if classic_mode == "context_open":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            by_image = defaultdict(list)
            for item in classic:
                by_image[item["image_name"]].append(item)
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=1
            ).fit([item["question"] for item in classic])
            question_matrix = vectorizer.transform([item["question"] for item in classic])
            rows = []
            for row_i, item in enumerate(classic):
                if str(item.get("answer_type", "")).strip().upper() != "OPEN":
                    if random.random() < closed_anchor_rate:
                        rows.append({
                            "img_path": str(CLASSIC_IMG_DIR / item["image_name"]),
                            "q": item["question"], "a": str(item["answer"]).strip(),
                            "kind": "context_closed_anchor",
                        })
                    continue
                context_rows = [x for x in by_image[item["image_name"]] if x["qid"] != item["qid"]][:24]
                context = "\n".join(
                    f"- Q: {x['question']}\n  A: {x['answer']}" for x in context_rows
                )
                scores = cosine_similarity(
                    vectorizer.transform([item["question"]]), question_matrix
                )[0]
                candidates = []
                for neighbor in scores.argsort()[::-1]:
                    if int(neighbor) == row_i:
                        continue
                    answer = str(classic[int(neighbor)]["answer"]).strip()
                    if norm_answer(answer) not in {norm_answer(x) for x in candidates}:
                        candidates.append(answer)
                    if len(candidates) >= 20:
                        break
                candidate_block = "\n".join(f"- {x}" for x in candidates)
                prompt = f"""Answer the target radiology question using the image and the
verified training facts for this same image below. The facts may phrase related
findings differently. Return only the shortest exact clinical answer, with no
explanation. When an answer from the canonical vocabulary is correct, copy its
wording exactly.

Verified same-image training facts:
{context}

Canonical answer vocabulary from similar training questions:
{candidate_block}

Target question: {item['question']}
Target answer:"""
                rows.append({
                    "img_path": str(CLASSIC_IMG_DIR / item["image_name"]),
                    "q": prompt, "a": str(item["answer"]).strip(),
                    "kind": "context_open", "raw_prompt": True,
                })
            random.shuffle(rows)
            return rows
        rows = []
        if classic_mode in {"open", "paper_open"}:
            def is_selected_open(item):
                if classic_mode == "paper_open":
                    return norm_answer(item["answer"]) not in {"yes", "no"}
                return str(item.get("answer_type", "")).strip().upper() == "OPEN"

            open_answer_counts = Counter(
                norm_answer(item["answer"]) for item in classic if is_selected_open(item)
            )
            max_open_frequency = max(open_answer_counts.values(), default=1)
        for r in classic:
            answer_type = str(r.get("answer_type", "")).strip().upper()
            selected_open = (
                is_selected_open(r) if classic_mode in {"open", "paper_open"}
                else False
            )
            if classic_mode in {"open", "paper_open"} and not selected_open:
                if random.random() >= closed_anchor_rate:
                    continue
            row = {
                "img_path": str(CLASSIC_IMG_DIR / r["image_name"]),
                "q": r["question"],
                "a": str(r["answer"]).strip(),
                "kind": "classic_open" if selected_open else "classic_closed",
            }
            repeats = 1
            if selected_open and open_balance_cap > 1:
                frequency = open_answer_counts[norm_answer(r["answer"])]
                repeats = min(
                    open_balance_cap,
                    max(1, math.ceil(math.sqrt(max_open_frequency / frequency))),
                )
            rows.extend(dict(row) for _ in range(repeats))
        random.shuffle(rows)
        return rows

    rows = []
    base = [json.loads(l) for l in DECONF.read_text().splitlines()]
    for r in base:
        if r.get("_neg") and not use_neg:
            continue
        rows.append({"img": r["img_name"], "q": r["question"], "a": str(r["answer"]).strip(), "kind": "neg" if r.get("_neg") else "orig"})
    if use_pattern and PATTERN.exists():
        for l in PATTERN.read_text().splitlines():
            r = json.loads(l)
            rw = (r.get("q_rewrite") or "").strip()
            if open_pattern_only and r.get("answer_type") != "OPEN":
                continue
            if rw and rw.lower() != r["question"].lower() and len(rw) > 4:
                rows.append({"img": r["img_name"], "q": rw, "a": str(r["answer"]).strip(), "kind": "pattern"})
    if use_ground and BBOX.exists():
        bbox_rows = json.loads(BBOX.read_text())
        conf_terms = {str(x["term"]).lower() for x in json.loads(CONF.read_text())}
        originals = [r for r in base if not r.get("_neg")]
        assert len(bbox_rows) == len(originals)
        for src, bb in zip(originals, bbox_rows):
            # Open questions are the actual bottleneck; keep this branch focused.
            if src.get("answer_type") != "OPEN" or not bb.get("extracted_bboxes"):
                continue
            masked = mask_clinical_term(src["question"], conf_terms)
            if not masked:
                continue
            rows.append({
                "img": src["img_name"], "q": masked, "a": str(src["answer"]).strip(),
                "kind": "ground", "bboxes": bb["extracted_bboxes"][:3],
            })
    random.shuffle(rows)
    return rows

class DiMedDataset(Dataset):
    def __init__(self, rows, processor, max_len=6144, img_side=448,
                 ground_dim=0.15, ground_mode="dim"):
        self.rows, self.proc, self.max_len, self.img_side = rows, processor, max_len, img_side
        self.ground_dim = ground_dim
        self.ground_mode = ground_mode
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        try: return self._p(idx)
        except Exception as e:
            print(f"[ds] idx {idx} err {e}", flush=True); return self._p((idx+1) % len(self.rows))
    def _p(self, idx):
        r = self.rows[idx]
        image_path = Path(r["img_path"]) if r.get("img_path") else IMG_DIR / r["img"]
        image = Image.open(image_path).convert("RGB")
        if r.get("kind") in {"ground", "ground_local"} and r.get("bboxes"):
            image = self._ground_image(
                image, r["bboxes"], self.ground_dim, self.ground_mode
            )
        w, h = image.size
        if max(w, h) > self.img_side:
            s = self.img_side/max(w, h); image = image.resize((int(w*s), int(h*s)), Image.LANCZOS)
        prompt = r["q"] if r.get("raw_prompt") else prompt_for(r["q"], r["a"])
        msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]},
                {"role":"assistant","content":[{"type":"text","text":r["a"]}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False)
        inp = self.proc(text=text, images=[image], return_tensors="pt", max_length=self.max_len, truncation=True)
        ids = inp["input_ids"].squeeze(0); am = inp["attention_mask"].squeeze(0)
        labels = ids.clone()
        pre = self.proc.apply_chat_template([msgs[0]], add_generation_prompt=True, tokenize=False)
        plen = self.proc(text=pre, images=[image], return_tensors="pt", max_length=self.max_len, truncation=True)["input_ids"].shape[1]
        labels[:plen] = -100
        out = {"input_ids": ids, "attention_mask": am, "labels": labels}
        for k in ("pixel_values","image_grid_thw"):
            if inp.get(k) is not None: out[k] = inp[k]
        return out

    @staticmethod
    def _ground_image(image, bboxes, ground_dim=0.15, ground_mode="dim"):
        """Inject top-k evidence while retaining its original spatial coordinates."""
        import PIL.ImageEnhance
        if ground_mode == "mask":
            grounded = Image.new("RGB", image.size, (0, 0, 0))
        else:
            grounded = PIL.ImageEnhance.Brightness(image).enhance(ground_dim)
        w, h = image.size
        for box in bboxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
            if x2 > x1 and y2 > y1:
                grounded.paste(image.crop((x1, y1, x2, y2)), (x1, y1))
        return grounded

class Collator:
    def __call__(self, feats):
        f = feats[0]
        out = {"input_ids": f["input_ids"].unsqueeze(0), "attention_mask": f["attention_mask"].unsqueeze(0), "labels": f["labels"].unsqueeze(0)}
        for k in ("pixel_values","image_grid_thw"):
            if k in f: out[k] = f[k]
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE/"adapter_dimed"))
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no_pattern", action="store_true")
    ap.add_argument("--no_neg", action="store_true")
    ap.add_argument("--use_ground", action="store_true")
    ap.add_argument("--open_pattern_only", action="store_true")
    ap.add_argument("--ground_only", action="store_true")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--init_adapter", default=None,
                    help="Initialize a new training stage from an existing LoRA adapter.")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Use NF4 QLoRA for memory-constrained continuation stages.")
    ap.add_argument("--img_side", type=int, default=448)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="Override epoch-derived optimizer-step count; used for compute-matched controls.")
    ap.add_argument("--classic_mode", choices=("off", "all", "open", "paper_open", "context_open"), default="off",
                    help="Use the classic 3,064-row VQA-RAD train split instead of generated DiMed data.")
    ap.add_argument("--closed_anchor_rate", type=float, default=0.0,
                    help="With --classic_mode open, retain this fraction of classic CLOSED rows.")
    ap.add_argument("--faithful_tag", choices=("off", "literal", "min2", "cleanv2", "cleanv3", "lingshuv4", "selectivev5", "openfocusv1"), default="off",
                    help="Use the BioMedCLIP/ScispaCy faithful corpus; min2 suppresses singleton PMI noise.")
    ap.add_argument("--ground_dim", type=float, default=0.15,
                    help="Brightness retained outside CTGM boxes; clean-v2 uses 0.6.")
    ap.add_argument("--ground_repeat", type=int, default=1,
                    help="Repeat accepted CTGM rows to realize the paper's equal branch loss.")
    ap.add_argument("--ground_mode", choices=("dim", "mask"), default="dim",
                    help="dim retains a faint full image; mask injects only top-k CTGM regions.")
    ap.add_argument("--open_balance_cap", type=int, default=1,
                    help="Train-only inverse-frequency repetition cap for OPEN consolidation.")
    ap.add_argument("--use_open_pairs", action="store_true",
                    help="Add source-validated same-image/same-answer OPEN textual pairs.")
    ap.add_argument("--faithful_open_only", action="store_true",
                    help="In faithful mode retain all original OPEN rows and only --closed_anchor_rate of CLOSED originals.")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="Trainer checkpoint path for epoch-by-epoch continuation.")
    ap.add_argument("--dry_run_corpus", action="store_true",
                    help="Build and audit the corpus without loading a model or GPU.")
    args = ap.parse_args()

    rows = build_corpus(
        use_pattern=not args.no_pattern,
        use_neg=not args.no_neg,
        use_ground=args.use_ground,
        open_pattern_only=args.open_pattern_only,
        classic_mode=args.classic_mode,
        closed_anchor_rate=args.closed_anchor_rate,
        faithful_tag=args.faithful_tag,
        ground_repeat=args.ground_repeat,
        open_balance_cap=args.open_balance_cap,
        use_open_pairs=args.use_open_pairs,
        faithful_open_only=args.faithful_open_only,
    )
    if args.ground_only:
        rows = [r for r in rows if r.get("kind") in {"ground", "ground_local", "ground_full"}]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    print("corpus:", len(rows), dict(Counter(r["kind"] for r in rows)), flush=True)
    if args.dry_run_corpus:
        return

    print(f"loading model {args.model} ...", flush=True)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cuda:0", attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    proc = AutoProcessor.from_pretrained(args.model)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    if args.init_adapter:
        print(f"loading trainable adapter {args.init_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        tmods = set()
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "language_model" in name and any(t in name for t in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]):
                tmods.add(name.split(".")[-1])
        print("LoRA targets:", sorted(tmods), flush=True)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=sorted(tmods), lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads(); model.print_trainable_parameters()

    ds = DiMedDataset(
        rows, proc, img_side=args.img_side, ground_dim=args.ground_dim,
        ground_mode=args.ground_mode,
    )

    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, bf16=True, logging_steps=20, save_strategy="epoch",
        save_total_limit=1, warmup_ratio=0.05, lr_scheduler_type="cosine", dataloader_num_workers=0,
        remove_unused_columns=False, report_to="none", gradient_checkpointing=True)
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator()).train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    model.save_pretrained(os.path.join(args.out, "final")); proc.save_pretrained(os.path.join(args.out, "final"))
    print("DONE ->", os.path.join(args.out, "final"), flush=True)

if __name__ == "__main__":
    main()
