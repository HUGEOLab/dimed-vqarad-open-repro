# DiMed VQA-RAD OPEN reproduction package (2026-07-24)

This archive reproduces the frozen `openfocusv2 -> visual verifier v3 ->
retrieval verifier v7` result on the 451-question VQA-RAD test split.

## Reported result

The primary OPEN/CLOSED definition is the original VQA-RAD `answer_type`
label, not the non-yes/no proxy.

| Metric | OPEN | CLOSED | Overall |
|---|---:|---:|---:|
| Normalized exact | 110/179 = 61.45% | 232/272 = 85.29% | 342/451 = 75.83% |
| Semantic | 132/179 = 73.74% | 233/272 = 85.66% | 365/451 = 80.93% |

Semantic scoring uses the frozen local Qwen3-VL-8B audit for all 200
non-yes/no answers and normalized exact matching for the remaining 251 yes/no
answers. The 200 non-yes/no rows consist of 179 original OPEN rows and 21
original CLOSED forced-choice rows.

## Archive contents

- `artifacts/adapters/`: initial e2 and final openfocusv2 LoRA adapters.
- `artifacts/train/`: exact OPEN-focused training corpus branches.
- `artifacts/verifier/`: CTGM regions, frozen features, selector, and
  train-only calibration.
- `artifacts/results/`: baseline, final predictions, audits, and metrics.
- `data/vqarad/`: exact train/test split metadata. Images are not bundled.
- `scripts/`: training, inference, verifier, exact scoring, and semantic judge.
- `provenance/`: training state and upstream-file hashes.
- `controls/sft/`: conventional and compute-matched ordinary-SFT controls,
  three frozen LoRA adapters, raw predictions, semantic audits, logs, paired
  statistics, and independent cached/inference/retraining entry points.

## 1. Verify the archive

Use Python 3.10 and install the pinned CPU dependencies first:

```bash
python3 -m pip install -r requirements-lock.txt
bash verify_package.sh
```

`verify_package.sh` checks every bundled file against `MANIFEST.sha256`,
compiles all Python scripts, and validates both split hashes.

## 2. Fast deterministic reproduction

This path needs no GPU or base model. It reruns both frozen verifier stages
from the shipped baseline/features, compares every normalized response against
the reference output, and recomputes exact and semantic metrics:

```bash
bash run_cached_reproduction.sh
```

Expected files:

- `work/cached/exact_metrics.json`: OPEN 110/179, overall 342/451.
- `work/cached/semantic_metrics.json`: OPEN 132/179, overall 365/451.

This verifies the complete selection and scoring logic, but not image-model
generation. Use the next path for that.

## 3. Full image-model inference

Obtain the exact upstream assets under their respective licenses:

1. Lingshu-7B revision `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`.
2. VQA-RAD images arranged as `VQARAD_DIR/images/<image_name>`.
3. The included `trainset.json` and `testset.json`, or files with the same
   SHA-256 values.

The VQA-RAD directory must contain:

```text
VQA-RAD-dir/
├── trainset.json
├── testset.json
└── images/             # exactly 315 files
```

Optionally verify every upstream byte before the expensive run (the full
inference and training scripts also invoke this automatically):

```bash
bash verify_upstream.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir
```

Run on a GPU with approximately 46 GB memory:

```bash
bash run_full_inference.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir cuda:0
```

This regenerates all 451 baseline answers, the 60 local CTGM feature rows,
both verifier stages, and `work/full/exact_metrics.json`. Expected runtime on
one NVIDIA A40 is roughly 15-25 minutes. Generation is greedy. Pinned package
versions and exact model hashes are recorded in `ENVIRONMENT.md` and
`provenance/lingshu7b.sha256`.

## 4. Re-run the semantic judge

Download `Qwen/Qwen3-VL-8B-Instruct` revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, then run:

```bash
bash run_semantic_judge.sh \
  work/full/final.json \
  /path/to/Qwen3-VL-8B-Instruct \
  cuda:0 \
  /path/to/VQA-RAD-dir/testset.json
```

The judge uses greedy decoding and a strict fixed prompt. It accepts standard
abbreviations/synonyms but rejects wrong laterality, anatomy, location,
modality, sequence, measurement, polarity, or omitted required findings.
Model-judge scores are not official exact-match scores and should be reported
with the judge identity and prompt.

## 5. Re-run OPEN-focused training

Training starts from the included e2 LoRA adapter and uses only the included
TRAIN-derived artifacts:

```bash
bash run_training.sh /path/to/Lingshu-7B /path/to/VQA-RAD-dir 0
```

The audited corpus has 3,992 rows: 1,412 original rows (all 1,241 OPEN plus a
seeded 10% CLOSED anchor), 1,088 OPEN pattern rewrites, 746 invariant OPEN
pairs, and 746 grounded branches. It uses one epoch, LR `2e-6`, 4-bit base
loading, batch size 1, gradient accumulation 16, and random seed 42. The
recorded run took about 65 minutes on an NVIDIA A40.

CUDA kernels are not guaranteed bitwise deterministic across hardware. For
strict metric reproduction, use the included final adapter; retraining is for
method reproducibility.

The included method-training entry point reproduces the OPEN-focused
continuation from the frozen e2 adapter. This archive does **not** claim
bitwise end-to-end retraining of e2 from the bare base model: the exact
historical e2 CTGM corpus snapshot was not retained after later CTGM artifact
revisions. The shipped, hashed e2 adapter is therefore the authoritative e2
starting state. Final model inference, both verifier stages, all scoring, and
all SFT-control training inputs are retained.

## 6. Reproduce the ordinary-SFT comparison

The package also ships the completed same-base, same-split SFT controls:

| Model | Overall exact | OPEN exact | Overall semantic | OPEN semantic |
|---|---:|---:|---:|---:|
| Conventional SFT, one epoch | 53.66% | 17.88% | 66.74% | 48.60% |
| Compute-matched SFT | 70.07% | 46.93% | 78.05% | 65.36% |
| Method, model only | 73.84% | 58.10% | 79.60% | 70.39% |

Rerun all frozen SFT scoring and paired tests without a GPU:

```bash
bash controls/sft/run_cached_sft_reproduction.sh
```

Regenerate the predictions from the shipped adapters, or retrain all controls:

```bash
bash controls/sft/run_sft_inference.sh \
  /path/to/Lingshu-7B /path/to/VQA-RAD-dir cuda:0
bash controls/sft/run_sft_training.sh \
  /path/to/Lingshu-7B /path/to/VQA-RAD-dir 0
```

See `controls/sft/README.md`. The full recipe significantly improves OPEN
exact over the compute-matched SFT control, but the semantic difference is
smaller and not significant at alpha 0.05 in the paired test; the package does
not claim that CTGM alone causes the full gap.

## Method provenance and limitations

- Selector v3 was fitted only on TRAIN features and grouped by image in
  five-fold calibration. It was frozen before test application.
- Retrieval v7 chooses its threshold by TRAIN leave-one-question-out precision
  and does not use test answers for selection.
- The verifier uses same-image TRAIN QA retrieval. VQA-RAD's supplied split
  contains questions sharing source images across train/test; report this
  transductive property explicitly.
- The final verifier changed 15 responses relative to openfocusv2 baseline:
  11 exact fixes, 2 exact breaks, and 2 wrong-to-wrong changes (net +9).
- Strictly reproducible scope: frozen-adapter inference, verifier application,
  exact/semantic scoring, OPEN continuation, and ordinary-SFT controls. Bare
  base-to-e2 retraining is outside the guaranteed scope described above.
- This is a research benchmark package, not clinical decision software.

See `UPSTREAM_ASSETS.md` before redistribution.
