# Independent validation review

Author: Sonja Sahebzad

Date: 2026-09-20

The proposed learning-curve study can support a measured improvement claim if the final comparison remains reserved until all choices are fixed. It cannot establish an expectation of 85% or 99.8% free next-word accuracy.

## Findings from the existing implementation

The original partition contains 4,016,699 globally unique normalized lines. Deduplication precedes the train/validation/test split, so identical normalized lines cannot cross those partitions. The new audit reproduces the original v3 test sampling exactly and reconstructs all four previous neural/adaptation exports from their recorded seeds and unchanged source ordering. The 7,500 prior evaluation cases are distinct. All 48,000 local adaptation lines belong to the training partition. See `models/learning_curve_prior_split_audit.json`.

All four previous development artifacts pass an independent recomputation of top-1, top-3, truncated reciprocal rank, synthetic-choice accuracy and shortlist recall. For example, the local adapter's development top-3 is 37.0% and coverage is 77.8%. These are development findings, not new final-test results. The previous 40.2% and 36.2% final scores used different cases and cannot measure a change in model quality directly.

The scoring interface keeps free candidate membership separate from supplied choices. It ranks canonical whole-word token sequences with a following boundary term. This is a finite candidate approximation, not exhaustive vocabulary search. A score is not a calibrated probability of being correct. Numbers, capitalization and most punctuation are removed; internal apostrophes remain. The measured task is therefore the next normalized English word.

## Frozen design and selection

Use the unchanged 600 adaptation development cases openly for model selection. Train three trajectories with seeds 20261410, 20261411 and 20261412, retaining checkpoints at 256, 512 and 1,024 steps. Use one fixed 1,024-step cosine schedule for every trajectory. The 256-step checkpoint is an early point on that schedule; it is not a replication of the old separately completed 256-step schedule. Keep the old operational local256 adapter intact as the benchmark.

Evaluate both 64-token and 256-token shortlist settings at each checkpoint and seed. For efficiency, generate the ordered 256-token list once, take its first 64 entries for the nested smaller list, add the identical independent R candidates to each, and score their union once. Neither the correct answer nor the supplied options may change either free membership list. Freeze score and word tie-breaking. Check direct versus shared scoring on neutral examples before final evaluation, including top-3 and choice invariance. Independent `topk(64)` calls need not resolve tied logits identically to a prefix of `topk(256)`.

Measure each width's latency separately on the first 20 development cases from each source, selected before outcomes. The first 60 rows of the CSV are source ordered and would not represent all three sources. Shared-run timing must not be attributed to each individual width. State whether timing includes the R shortlist, model loading and warm-up; also record device, batch size and fixed case identities.

Select checkpoint and shortlist by mean development top-3 across all three seeds, then mean top-1, then mean latency. Resolve complete ties by the fixed checkpoint and shortlist order. Choose the representative seed closest to the winning group's mean development top-3, then closest to its mean top-1, then fixed seed order. Save the selection before accessing final outcomes. Never choose the seed with the best final score.

## One confirmatory deployment comparison

The primary endpoint is exact normalized-word top-3 accuracy of that development-selected representative compared with the untouched operational local256 model on the same new 900 final cases, balanced at 300 per source. Exact ordered `(source, line_hash)` alignment is required before pairing.

Resample unique lines within each source, preserving paired outcomes and equal source weights, for 10,000 bootstrap replicates. Promote only when the paired 95% percentile interval has a strictly positive lower bound and the two-sided exact McNemar test has p < 0.05. The extra exact test is a conservative gate on the same primary contrast, not a second opportunity to declare success. Paired resampling uses matching observations together; the exact McNemar calculation uses a binomial distribution for discordant pairs. See [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) and [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

Evaluate the other two seeds at the selected setting as descriptive replication. For a mean-of-seeds interval, average the three correctness indicators for each line first, then bootstrap lines. Do not turn 900 lines and three models into 2,700 independent observations. This interval is conditional on those three trained models, does not estimate variation across future seeds, and is not an ensemble score. Other metrics and source comparisons are descriptive; do not promote on whichever secondary endpoint looks favorable.

## Limits that must remain visible

- Reusing development data permits optimization and may overfit that set. The untouched final comparison is essential.
- Exact normalized-line checks do not remove near duplicates, related documents, shared authors or unknown overlap with public pretrained weights. Line-level uncertainty can be optimistic when documents are correlated.
- The equally weighted three-source estimand describes this balanced benchmark. It does not estimate the natural traffic mix of an app.
- Synthetic choices use the target plus three frequency-matched distractors. Their accuracy is a separate diagnostic, not a Coursera quiz score or free-text accuracy.
- An unsuccessful promotion test means this experiment did not establish improvement. It does not prove equivalence or that further training can never help.
- More training changes both the number of updates and token exposure. This study estimates their combined effect under a fixed schedule; it does not isolate a single cause.

## Reproducible controls

`python/learning_curve_audit.py` is CPU-only and never discovers or opens new final data automatically. It validates hash manifests, exact alignment, per-case ranks and coverage, choice independence, stored metric calculations, complete candidate grids, deterministic selection and paired inference. The full-artifact selection API verifies metrics from details. The optional flat-summary API requires the caller to perform those artifact checks first.

Run `python -m unittest discover -s tests -p test_learning_curve_audit.py` for 17 targeted tests. `tests/audit_split_reconstruction.R` independently reconstructs the prior sampling and checks the sanitized reserved hash manifest against prior evaluation and adaptation-training hashes. It does not read the new held-out prefixes, targets or prediction outcomes. Hash checks do not by themselves prove what a model trained on, so preserve training code, data, adapter and evaluation hashes together.