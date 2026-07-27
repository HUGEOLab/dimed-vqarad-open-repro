# DiMed VQA-RAD OPEN reproduction

Reproduces the frozen `openfocusv2 -> visual verifier v3 -> retrieval verifier v7`
result on the 451-question VQA-RAD test split.

OPEN/CLOSED follows the original VQA-RAD `answer_type` label (179 OPEN / 272
CLOSED), not the non-yes/no proxy.

| Metric | OPEN | CLOSED | Overall |
|---|---:|---:|---:|
| **Semantic** (reported) | 132/179 = **73.74%** | 233/272 = 85.66% | 365/451 = **80.93%** |
| Normalized exact | 110/179 = 61.45% | 232/272 = 85.29% | 342/451 = 75.83% |

Semantic is the reported metric: OPEN answers are free text, so equivalence is
judged per row by a local LLM (Qwen3-VL-8B) over all 200 non-yes/no rows, and the
251 yes/no rows use normalized exact match. Exact match is kept as a
deterministic anchor anyone can recompute without a GPU. Model-judge scores are
not official exact-match scores; cite them with the judge identity and prompt.

**Weights are not in this repository** — each LoRA adapter is 182 MB, over
GitHub's per-file limit. Steps 1-3 need no weights and no GPU; steps 4-6 do.

---

## Step 0 — Environment

Python 3.10. Pinned versions and GPU details are in `ENVIRONMENT.md`.

```bash
python3 -m pip install -r requirements-lock.txt
```

## Step 1 — Verify the archive (no GPU)

```bash
bash verify_package.sh
```

Checks every file against `MANIFEST.sha256`, compiles all Python, validates both
split hashes, and reruns the cached result plus the SFT controls.

Expected: it reports **18 missing files and exits 1**. Those 18 are the adapter
files this repository does not ship. Every other hash must pass.

## Step 2 — Reproduce the result (no GPU, no weights, ~1 min)

Fastest end-to-end check. Reruns both frozen verifier stages from the shipped
baseline and features, compares every normalized response against the reference
output, then recomputes both metrics.

```bash
bash run_cached_reproduction.sh
```

Ends with `PASS: cached reproduction matches the frozen responses and metrics.`

| Output | Expected |
|---|---|
| `work/cached/exact_metrics.json` | OPEN 110/179, overall 342/451 |
| `work/cached/semantic_metrics.json` | OPEN 132/179, overall 365/451 |

This verifies the complete candidate-selection and scoring logic. It does not
re-run image-model generation — that is step 4.

## Step 3 — Reproduce the ordinary-SFT comparison (no GPU)

```bash
bash controls/sft/run_cached_sft_reproduction.sh
```

| Model | Overall exact | OPEN exact | Overall semantic | OPEN semantic |
|---|---:|---:|---:|---:|
| Conventional SFT, 1 epoch | 53.66% | 17.88% | 66.74% | 48.60% |
| Compute-matched SFT | 70.07% | 46.93% | 78.05% | 65.36% |
| Method, model only | 73.84% | 58.10% | 79.60% | 70.39% |

The full recipe clearly improves OPEN exact over the compute-matched control; the
semantic difference is smaller and not significant at alpha 0.05 in the paired
test. This package does not claim CTGM alone causes the full gap. Details in
`controls/sft/README.md`.

## Step 4 — Full image-model inference (GPU + weights)

Obtain, under their own licenses:

1. `lingshu-medical-mllm/Lingshu-7B`, revision `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`
2. VQA-RAD, arranged as below
3. The `openfocusv2_final` LoRA adapter, placed at `artifacts/adapters/openfocusv2_final/`

```text
VQA-RAD-dir/
├── trainset.json
├── testset.json
└── images/          # exactly 315 files
```

Optional byte-level check of the upstream inputs (the run scripts call it anyway):

```bash
bash verify_upstream.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir
```

Then, on a GPU with about 46 GB:

```bash
bash run_full_inference.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir cuda:0
```

Regenerates all 451 baseline answers, the 60 local CTGM feature rows, both
verifier stages, and `work/full/exact_metrics.json`. Generation is greedy.
Roughly 15-25 minutes on one NVIDIA A40.

## Step 5 — Re-run the semantic judge (GPU)

Skip this if you only want the reported numbers. The frozen per-row judge
decisions ship in
`artifacts/results/results_openfocusv2_verifier_v3_v7_semantic_qwen3.json`, so
step 2 already reproduces 365/451 without downloading the judge model.

To re-judge, get `Qwen/Qwen3-VL-8B-Instruct` revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`:

```bash
bash run_semantic_judge.sh \
  work/full/final.json \
  /path/to/Qwen3-VL-8B-Instruct \
  cuda:0 \
  /path/to/VQA-RAD-dir/testset.json
```

Greedy decoding, strict fixed prompt, batch size 12. Keep the batch size as
shipped — judge decisions are not bitwise stable across batch sizes. The prompt
accepts standard abbreviations and synonyms, and rejects wrong laterality,
anatomy, location, modality, sequence, measurement, polarity, or an omitted
required finding.

## Step 6 — Re-run OPEN-focused training (GPU + weights)

Needs the `e2_final` adapter at `artifacts/adapters/e2_final/`.

```bash
bash run_training.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir 0
```

Corpus: 3,992 audited rows — 1,412 original (all 1,241 OPEN plus a seeded 10%
CLOSED anchor), 1,088 OPEN pattern rewrites, 746 invariant OPEN pairs, 746
grounded branches. One epoch, LR 2e-6, 4-bit base, batch size 1, gradient
accumulation 16, seed 42. About 65 minutes on an A40.

CUDA kernels are not bitwise deterministic across hardware, so use the shipped
final adapter for strict metric reproduction; retraining is for method
reproducibility. To retrain the SFT controls instead, use
`controls/sft/run_sft_training.sh` — it starts from the bare base model and needs
no shipped adapter.

---

## Scope and known properties

- **Strictly reproducible:** frozen-adapter inference, both verifier stages,
  exact and semantic scoring, the OPEN-focused continuation, and the SFT controls.
- **Outside guaranteed scope:** retraining e2 from the bare base model. The exact
  historical e2 CTGM corpus snapshot was not retained after later artifact
  revisions, so the shipped, hashed e2 adapter is the authoritative starting state.
- Selector v3 was fitted on TRAIN features only, grouped by image in five-fold
  calibration, and frozen before test application. Retrieval v7 picks its
  threshold by TRAIN leave-one-question-out precision. Neither uses test answers
  for selection.
- The verifier retrieves same-image TRAIN QA, and VQA-RAD's supplied split shares
  source images between train and test. This is transductive; report that property
  explicitly when citing these numbers.
- Relative to the openfocusv2 baseline the final verifier changed 15 responses:
  11 exact fixes, 2 exact breaks, 2 wrong-to-wrong (net +9 exact).
- Research benchmark artifacts, not clinical decision software.

See `UPSTREAM_ASSETS.md` before redistributing.
