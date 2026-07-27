# Ordinary-SFT controls

These controls test the DiMed/openfocusv2 model against ordinary SFT with the
same Lingshu-7B revision and frozen classic VQA-RAD 3,064/451 split.

## Frozen results

| Model | Overall exact | OPEN exact | CLOSED exact | Overall semantic | OPEN semantic |
|---|---:|---:|---:|---:|---:|
| Conventional SFT, 1 epoch | 53.66% | 17.88% | 77.21% | 66.74% | 48.60% |
| Compute-matched SFT | 70.07% | 46.93% | 85.29% | 78.05% | 65.36% |
| Method, model only | 73.84% | 58.10% | 84.19% | 79.60% | 70.39% |
| Method + verifier | 75.83% | 61.45% | 85.29% | 80.93% | 73.74% |

The compute-matched control uses the same 1,182-step LR 2e-4 stage and
250-step LR 2e-6 continuation schedule as the method. It contains only original
QA rows: no pattern rewrites, CTGM rows, invariant pairs, semantic negatives,
retrieval, or verifier.

The conventional one-epoch control is descriptive because it uses less
compute. Exact-match superiority of the method is stronger than semantic
superiority; see `sft_gap_analysis.json` for paired tests and limitations.

## Reproduction routes

From the package root:

```bash
# CPU-only: rescore frozen predictions and rerun the paired analysis
bash controls/sft/run_cached_sft_reproduction.sh

# GPU: regenerate predictions from the three shipped SFT adapters
bash controls/sft/run_sft_inference.sh \
  /path/to/Lingshu-7B /path/to/VQA-RAD-dir cuda:0

# GPU: retrain all SFT controls from the pinned base and split
bash controls/sft/run_sft_training.sh \
  /path/to/Lingshu-7B /path/to/VQA-RAD-dir 0
```

Full retraining takes roughly seven hours on one NVIDIA A40. CUDA kernels are
not guaranteed bitwise deterministic across different hardware; frozen adapter
inference and CPU cached scoring are the strict result-reproduction paths.

`results/` contains raw predictions, exact metrics, semantic audits, and
semantic metrics. `provenance/` contains trainer states. `logs/` contains the
original training and inference logs. Only the small LoRA adapters are bundled;
the licensed base model and VQA-RAD images must be obtained separately and are
validated by the package's upstream SHA-256 checks.
