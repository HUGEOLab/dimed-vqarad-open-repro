# Source provenance

- Original method packaging date: 2026-07-19 UTC.
- SFT-control integration and clean-package audit: 2026-07-24 UTC.
- Source repository commit: `5bd0924d1332eae96a7bdcf02fc81480245db27c`.
- The source worktree contained unrelated uncommitted changes. Therefore the
  archive's `MANIFEST.sha256`, rather than the repository commit alone, is the
  authoritative identity of the delivered implementation.
- All result-producing scripts included here passed Python compilation, shell
  syntax validation, the cached 451-row response-equivalence test, SFT metric
  recomputation, and paired-analysis reproduction before packaging.
- The frozen e2 adapter is retained and hashed. The exact historical e2 CTGM
  corpus snapshot is not retained, so the guaranteed training scope begins at
  the e2 adapter for the method and at the base model for the SFT controls.
