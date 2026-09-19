# Learning curves, ranking audit and reserved evaluation

Author: **Sonja Sahebzad**

This study compares three training durations and two candidate-list widths across
three random seeds. It uses the existing official-corpus training pool, the same
600 development cases and a newly reserved 900-case final sample. It tests an
improvement hypothesis, not a promised accuracy. Synthetic four-choice results
are reported separately from free-word predictions.

## Read or knit the result

Open `23_learning_curve_report.Rmd` in **Data Science Capstone.Rproj**. Click
**Knit** to rebuild its HTML from the committed result tables; model training and
GPU inference are not repeated by knitting. The source, local HTML and project
file remain together. The public report is `docs/learning-curves.html`.

```r
rmarkdown::render("23_learning_curve_report.Rmd")
```

The report needs R Markdown, jsonlite, data.table, ggplot2, scales and plotly,
plus Pandoc supplied by RStudio. Its `sessionInfo()` records the actual R setup.

## Provenance and boundaries

- `models/learning_curve_protocol.json` records the hypotheses, data hashes,
  seeds, grid, common learning-rate schedule and promotion rule before training.
- `22_prepare_learning_curve.R` reserves 300 new test lines per source. It
  reconstructs and excludes every earlier evaluation sample, checks exact
  normalized-line separation and precomputes the unchanged R candidate lists.
- `models/learning_curve_reserved_case_hashes.csv` shares identifiers without
  republishing the held-out text. The independent split audit checks these
  identifiers without opening final prefixes or outcomes.
- `models/learning_curve_implementation.json` freezes evaluator code, scorer,
  base weights, vocabulary, R model and old operational adapter after neutral
  GPU scoring checks, before development scoring.
- `models/learning_curve_selection.json` freezes the development-selected
  checkpoint, width and representative seed, with hashes of all 18 artifacts,
  before final inference starts.
- Per-case result files retain ranks, coverage and source/hash identifiers so
  aggregate metrics and paired comparisons can be independently recomputed.

The ranking, training and validation agents worked in separate branches. Only
the coordinator ran GPU jobs, sequentially. See the [ranking audit](../research/ranking-audit.md),
[training specification](LEARNING_CURVE_TRAINING.md) and
[independent validation review](LEARNING_CURVE_VALIDATION.md).

## Dependencies and reproduction

The existing local setup uses Windows, Python 3.13, the packages in
`requirements-adaptation-lock.txt`, the pinned Qwen3-1.7B-Base revision and an
8 GiB NVIDIA GPU. Start with [neural setup](README.md) and
[the previous adaptation experiment](ADAPTATION.md). The expanded R model,
original corpus partitions, unchanged adaptation development CSV, local token
cache and old operational adapter are prerequisites. Raw data and model weights
are deliberately outside Git.

All commands below run from the project directory. Preserve existing experiment
directories and result artifacts before starting a separate full reproduction.
The trainer refuses overwrites, including partial runs; the evaluator only
reuses complete artifacts whose identities, hashes and metrics verify.

```powershell
# CPU controls can be rerun without a GPU or changing model outputs.
& '.venv-neural/Scripts/python.exe' -m unittest discover -s tests -p 'test_learning_curve_*.py'
& '.venv-neural/Scripts/python.exe' -m unittest discover -s tests -p 'test_ranking_audit.py'

# Prepare the reserved split before any model selection or final evaluation.
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' 22_prepare_learning_curve.R
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' tests/audit_split_reconstruction.R . models/learning_curve_prior_split_audit.json

# Run one GPU job at a time. Microbatch eight was fixed by a training-only pilot.
foreach ($seed in @(20261410,20261411,20261412)) {
  & '.venv-neural/Scripts/python.exe' python/learning_curve_training.py --seed $seed --microbatch 8
  if ($LASTEXITCODE -ne 0) { throw 'Training failed. Preserve and inspect the run.' }
}

# Freeze the evaluator only after its neutral equivalence checks succeed.
& '.venv-neural/Scripts/python.exe' python/learning_curve_experiment.py --check-scoring
if ($LASTEXITCODE -ne 0) { throw 'Scoring check failed. Do not evaluate.' }

# All 18 configurations use the same 600 development examples.
foreach ($seed in @(20261410,20261411,20261412)) {
  foreach ($step in @(256,512,1024)) {
    & '.venv-neural/Scripts/python.exe' python/learning_curve_experiment.py --evaluate-development --seed $seed --steps $step
    if ($LASTEXITCODE -ne 0) { throw 'Development check failed. Do not select.' }
  }
}

# Freeze selection, evaluate the old model and three selected checkpoints,
# then calculate the prespecified paired comparison and default model.
& '.venv-neural/Scripts/python.exe' python/learning_curve_experiment.py --finish
if ($LASTEXITCODE -ne 0) { throw 'Final comparison failed. Preserve the evidence.' }
```

These commands require inputs matching the published protocol's hashes. A fresh
reproduction on another GPU or software platform may not be bitwise identical,
even with the same seeds. If inputs or implementation differ, keep the published
study intact and record a separate protocol before evaluation. Do not rewrite
hashes to make incompatible results appear to belong to this study. A new round
of development after seeing the final results also needs a new reserved test.

## Candidate scoring and runtime

The 64 and 256 cutoffs refer to eligible neural **token IDs**, before lowercase
aliases are combined into distinct words. Each list receives the same up to 20
R candidates. Token ties resolve by ascending token ID; word-score ties resolve
alphabetically. The smaller neural list is a prefix of the larger one.

Whole-word scores are computed once for the larger free list during development.
Only missing supplied choices are scored afterward, so adding options cannot
alter free-word batches or predictions. Neutral checks compare shared and direct
scoring, followed by identical-output checks on all 60 fixed timing cases at
each checkpoint. Numerical differences that change these checked predictions
stop the experiment before selection.

Development latency uses separate full calls for each width on the first 20
cases per source, alternating width order. Final latency uses all 900 cases.
Both exclude model loading, warm-up and generation of the precomputed R lists;
both include free and synthetic-choice scoring. These are inference timings,
not end-to-end Shiny response times.

## Selection and interpretation

Select the stage/width group by mean development top-three accuracy, then
top-one accuracy, then independent latency. Use integer success totals for
accuracy ties. Select the representative seed nearest its group mean, not the
highest-scoring seed on final data.

One confirmatory contrast compares that representative with the untouched old
operational model on exactly aligned final cases. A paired source-stratified
bootstrap uses 10,000 replicates with seed 20261490. Promotion requires both a
positive 95% interval lower bound and two-sided exact McNemar p below 0.05.
Failure to pass does not prove equivalence. The mean of three seeds is a
descriptive repeated-training result, not an ensemble or a confidence interval
over future training seeds.

Exact-line deduplication does not eliminate related documents, near duplicates
or unknown base-model pretraining overlap. Equal weighting of the three sources
describes this benchmark, not a guaranteed user population. Multiple plausible
next words can exist, while accuracy counts only the recorded continuation.

## Use the documented default in RStudio

Open `18_try_neural_predictor.R` and click **Source**. Type only the English
sentence fragment at the first Console prompt. Press Enter, then optionally
type comma-separated answer options at the second prompt. The script prints
three free suggestions and, separately, the ranking of supplied options.

The saved promotion decision determines the default. Explicit older-model
arguments remain available for reproducible comparisons. The lightweight R
model remains in `13_try_predictor.R`. Suggestions are rankings, not calibrated
percentages of correctness.
