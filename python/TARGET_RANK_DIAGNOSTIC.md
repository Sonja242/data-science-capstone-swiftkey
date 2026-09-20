# Target-aware development diagnosis

Author: Sonja Sahebzad

This is an exploratory diagnosis on the existing 600-case development set. It
deliberately supplies missing observed words. The resulting counts are not
production accuracy, a new final test, or a deployable answer-aware predictor.

## Result

The existing candidate pool covers 467 of 600 targets. Of these, 222 already rank
among the first three and 245 rank lower. Adding the 133 missing targets with the
unchanged FP16 score rule yields no further top-three successes. The closest
missing target ranks eighth, 0.543 log-score point below the third other word.

The prior complete-word pool has 220 existing top-three successes, 252 covered
but lower-ranked targets and 128 missing targets. Target insertion adds no
top-three success in that pool either.

All 16 cases within the registered 0.10 log-score margin were checked in FP32;
there were no hypothetical missing-word recoveries requiring additional checks.
No missing target was close enough to enter that subset. The existing pool has
one top-three exit and one entry; the complete-word pool has one entry. These
changes concern words already available. All checked target/third-other scores
agree with full-sequence FP32 calculations within the registered 0.0001 limit
(maximum difference 0.0000315).

The conclusion prioritizes the model and score/ranking criterion for further
development. It does not establish that search improvements can never help or
that a particular future accuracy level is achievable. Production is unchanged.

## Reproduction

Use the existing Capstone RStudio project, official corpus preparation, locally
pinned model and adapter, and Python environment from the earlier studies.

For saved evidence, source `27_target_rank_diagnostic.R`. This runs both evidence
checks and knits `28_target_rank_diagnostic.Rmd`, with results printed in the R
Console. Knitting the Rmd alone reconstructs the report from saved artifacts.

For a fresh reproduction in a separate preserved project copy:

```powershell
& '.venv-neural/Scripts/python.exe' -m unittest discover -s tests -p test_target_rank_diagnostic.py -v
& '.venv-neural/Scripts/python.exe' python/target_rank_diagnostic.py --run
& '.venv-neural/Scripts/python.exe' python/target_rank_diagnostic.py --verify
& '.venv-neural/Scripts/python.exe' python/audit_target_rank.py
```

The GPU script refuses to overwrite a registered run. Keep the original
registration and evidence rather than silently replacing it. Source, input,
model and artifact fingerprints identify this diagnosis.

## Method details

Old candidate scores are reused exactly. Only an absent target is scored as a
singleton using the archived `AdaptedRanker` and the same merged FP16 weights.
Rank ties use lexical order. FP32 checks are selected when either pool is within
0.10 log-score point of its third highest other word or a missing target would
reach the first three. The entire selected candidate union is rescored, with
identical merged weights promoted to FP32, TF32 disabled and batch size four.
Target and cutoff words also receive an uncached full-sequence formula check.

Only development outcomes and model metadata are read. Synthetic answer options
are not used. No fitting, shortlist regeneration, tuning or production-default
change occurs. Saved FP16 and FP32 outcomes remain separate. Unchecked cases are
not represented as FP32-verified, and no complete FP32 performance estimate is
constructed from the selectively checked subset.

`models/target_rank_cases.csv` provides the compact rank table.
`models/target_rank_fp32.json` preserves every rescored candidate for the selected
examples. The original development candidate evidence supplies the unchanged
competitors for `models/target_rank_fp16.json`. The independent audit recomputes
ranks, counts, subset selection and score comparisons without fitting a model.

## References

- [Official Coursera corpus](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
- [PyTorch 2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html).
- [Earlier complete-word experiment](WORD_GENERATION.md).
