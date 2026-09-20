# Independent validation for complete-word generation

Author: **Sonja Sahebzad**

This study changes candidate generation while retaining the operational local256 weights, context and canonical whole-word scorer. It does not train a new model. The two declared alternatives are `beam16` and `beam64`, with at most six proposal tokens and 128 terminal proposals. Beam search determines which words are proposed; canonical reranking scores determine which proposed words reach the top three.

## Data and single primary comparison

Reuse the existing 600 development cases explicitly. Select one expanded configuration by integer top-three successes, then integer top-one successes, then independently measured latency on the fixed 60-case sample, then the fixed order `beam16`, `beam64`. Never select by new final outcomes or whichever token subgroup looks most favorable.

The new final set has 900 unique normalized lines, 300 from each source. It must exclude the earlier v3 final 3,000 lines and the neural, adaptation and learning-curve final sets of 900 each, a total of 5,700 earlier final lines. All known development lines and exact normalized training matches must also be absent. `tests/audit_word_generation_split.R` reconstructs earlier samples, verifies reserved hashes/source labels against the existing official test partition, and checks separation without opening the new final case file or model outcomes. The existing global normalized-text deduplication occurs before partitioning. Exact checks do not remove near duplicates, related documents, or unknown pretrained-model overlap.

Compare the development-selected expanded configuration with the fixed baseline on the same ordered final cases. The only confirmatory endpoint is exact normalized-word top-three accuracy. Use a paired source-stratified line bootstrap with 10,000 replicates and seed 20261590. Promote only when the paired 95% interval has a strictly positive lower bound and the two-sided exact McNemar p-value is below 0.05. A failure to pass is not evidence of equivalence. Single-token and multi-token groups, source groups, coverage and error transitions are descriptive.

## Candidate and score invariants

Every expanded list contains the complete original candidate list. Original scores are stored once and reused unchanged. Added-only score lists must not overwrite baseline words. Supplying options must not inject them into the free list, change original scores, or change generation. Supplied options are scored independently using the same model in both arms, so their accuracy is necessarily identical. It is a fixed-model diagnostic, not evidence that candidate generation improves multiple-choice performance.

For an unchanged canonical scorer and tie rule, adding candidates cannot raise the rank of a previously covered target. A new top-three success must therefore concern a target absent from the baseline list. Added competitors can also displace originally correct predictions. The audit reports gains, losses, both-correct, both-wrong, coverage gains and any violation of that mechanism. It validates canonical target-token counts as positive integers; their meaning must remain tokenization of the observed normalized word using the frozen tokenizer, not the length of a beam proposal path.

Scores for the same word obtained from independently padded FP16 calculations can differ slightly. The pre-development neutral diagnostic justified a separate absolute log-score tolerance of 0.02, while the FP32 formula check requires at most 0.0001. Shared original scores and independence from options remain exact requirements. A log-score difference of 0.02 corresponds to about a 2.02% likelihood ratio difference, not a 2.02 percentage-point accuracy allowance. The initial stricter check and full diagnostic must remain in the record. Three neutral contexts do not establish an error bound or unchanged rankings for all corpus cases. The fixed neutral checks retain strict rejection at those limits. Independent corpus batches instead produce descriptive diagnostics: the comparison count, maximum discrepancy, affected case and word identifiers, and flags outside the unchanged neutral reference are retained. Such flags are not an unregistered corpus-wide rejection gate and do not alter scores, model selection or promotion. Preserve and investigate unusual discrepancies before final interpretation; never silently widen the neutral reference after performance results are seen.

The audit also counts cases with a third-to-fourth free-word log-score margin at most 0.04, twice the numerical bound, and identifies gains or losses at such a boundary. This is a descriptive diagnostic only. It does not change candidate scores, model selection, significance or the promotion decision.

## CPU audit interfaces

`python/word_generation_audit.py` does not load weights, run a GPU, or automatically discover final files. It reads explicitly supplied artifacts containing canonical score lists and recomputes results, rather than trusting stored percentages. The frozen evaluator labels splits `development` and `final`; this study does not use the archived experiments' `test` label.

- `derive_artifacts(raw, candidates=("beam16", "beam64"), ...)` validates the raw canonical scores, reconstructs free rankings and supplies aligned metric artifacts.
- `select_generation(raw, milliseconds)` accepts development data only and requires latency for both declared configurations.
- `primary_comparison(raw, selected_candidate, ...)` accepts a `split="final"` artifact containing only the frozen selected arm and returns the paired primary result plus descriptive groups.
- `audit_reserved_hashes(...)` checks hash manifests for all previous final sets, development exclusions and normalized training separation.

All functions accept explicit case identities and, where relevant, the declared numerical tolerance. Default bootstrap seed is 20261590. The `score_tolerance` argument retains the unchanged 0.02 neutral reference for descriptive corpus flags and close-boundary counts; it does not impose a new corpus-wide rejection threshold. Each derived artifact and the selection/primary summaries expose `numerical_diagnostics` separately from `hard_invariants_passed`. Tests cover score tampering, canonical ranking, option separation, missing previous holdouts, duplicates, exact selection ties, paired counts and small-boundary diagnostics. No archived learning-curve code is edited.

```powershell
& '.venv-neural/Scripts/python.exe' -m unittest discover -s tests -p 'test_word_generation_audit.py'
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' tests/audit_word_generation_split.R `
  'C:/path/to/Data Science Capstone' 'C:/path/to/word_generation_partition_audit.json'
```

Preserve protocol, model/data/code fingerprints and the frozen selection with the score artifacts. A later round of candidate or parameter tuning requires another untouched final holdout.