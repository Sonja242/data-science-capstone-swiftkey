# Ranking rules and first-suggestion calibration

Author: Sonja Sahebzad

This controlled study keeps the operational model weights, 128-token context
and target-blind candidate generation fixed. It compares seven predefined
score rules. No learned reranker or new base model is trained.

## Data and sequence

- Reuse 600 development examples only for selecting a ranking rule.
- Reserve 900 previously unused validation lines for confidence fitting.
- Reserve 900 previously unused test lines for final accuracy and calibration.
- Exclude 9,300 prior evaluation lines, official training overlap and external
  adaptation text overlap. Exact-line checks do not exclude near duplicates or
  unknown overlap with the pretrained model's original data.

The candidate pool never receives the observed word or multiple-choice options.
Every archived development candidate and original score is checked for exact
agreement. The earlier target-aware diagnosis is not an input to prediction.

## Fixed rules

Let L be canonical token-sequence log likelihood, B the following word-boundary
log mass, n the canonical model-token count, and c the number of characters.

| Rule | Score |
|---|---|
| Original | L+B, preserving the archived operation order |
| No boundary | L |
| Half boundary | L+0.5B |
| Token mean | L/n+B |
| Token mean without boundary | L/n |
| Character adjustment | L/sqrt(c)+B |
| Two tokenizations | Log-sum-exp of complete-word scores on two distinct token paths |

The extra path tokenizes a space separately from the word. Its decoded surface
must exactly equal the canonical surface. An identical path is counted once.
This checks segmentation sensitivity without replacing the model's tokenizer;
it is not exhaustive marginalization over all possible paths. The combined
boundary/token-mean rule is the prespecified interaction check.

Selection maximizes development top-three count, then first-suggestion count,
then a fixed simplicity order favoring the original rule. No variants are added
after inspecting outcomes. The original rule wins this development comparison.

## Confidence and abstention

The event is whether the **first suggestion** equals the observed word. Top-three
accuracy is measured separately. A fixed logistic map takes the logit of the
shortlist's relative top-score share as its input. It minimizes summed binary
log loss plus 0.5 times the squared slope. The intercept is unpenalized. This map
is fitted only on the separate calibration examples, including misses outside
the candidate list; it cannot change the ranking.

The final assessment includes accuracy over all cases, Brier score, log loss,
ten equal-width reliability bins, and answer coverage versus selective accuracy
at fixed thresholds 0, 0.50, 0.70, 0.85 and 0.95. Wilson intervals describe
uncertainty in the observed accuracy among answered cases. An empty group has
no accuracy estimate. A high point estimate on a small subset is not a reliable
95%-confidence guarantee.

If the original rule wins development selection, the final test has no ranking
challenger. An internal zero-difference consistency calculation must not be
interpreted as statistical evidence of improvement. It instead evaluates the
retained predictor and its independently fitted confidence map.

## Reproduce

Use the existing local environment and pinned model/adapter from the preceding
studies. Keep previous registrations and artifacts. Fresh execution refuses to
overwrite evidence.

```powershell
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' 29_prepare_ranking_study.R
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' tests/audit_ranking_partitions.R
& '.venv-neural/Scripts/python.exe' -m unittest discover -s tests -p test_ranking_rules.py -v
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --freeze
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --evaluate development
& '.venv-neural/Scripts/python.exe' python/audit_ranking_study.py
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --select
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --evaluate calibration
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --fit-calibration
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' tests/audit_ranking_calibration.R
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --evaluate final
& '.venv-neural/Scripts/python.exe' python/ranking_study.py --summarize
& '.venv-neural/Scripts/python.exe' python/audit_ranking_study.py --final
```

After completion, source `31_verify_ranking_study.R` in RStudio. It verifies
saved evidence and knits `30_ranking_and_calibration.Rmd` with printed results
and expandable code. It does not restart GPU evaluation. `18_try_neural_predictor.R`
remains the standard predictor while no challenger passes the promotion rule.

## Numerical and implementation notes

Neutral canonical component scores agree exactly with the archived scorer.
Alternate-tokenization cached/full-sequence FP16 checks precede development
evaluation; the registered absolute tolerance is 0.06 log-score point. This is a
formula check, not an accuracy tolerance. Floating-point batching can still
affect small score differences.

An independent development audit initially omitted the boundary term in its
character-adjustment formula. That audit defect was corrected and covered by a
regression test before selection or calibration. The frozen predictor, rules
and saved model outcomes were not changed.

Standalone final timing reruns the first 20 cases from each source after loading,
with GPU synchronization and exact prediction checks. It excludes the R shortlist,
model loading and answer-option scoring. Earlier studies timed different work.

All score arrays, candidate words, observed targets, fitted coefficients, case
identifiers and summary formulas are retained. Full input prefixes and model
weights are not republished. Runtime and software versions follow the pinned
environment in the existing project.

Before final testing, an independent R `optim` BFGS fit from zero reproduced the
Python Newton calibration coefficients within 3.4e-9. Its saved audit binds the
calibration and fitted-coefficient hashes and confirms that no final outcomes
were read. The separate final audit reconstructs rankings, counts, reliability
bins, proper scores and Wilson intervals from saved case-level evidence.

## References

- [Official corpus](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [PyTorch 2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html).
- [Target-aware diagnosis](TARGET_RANK_DIAGNOSTIC.md).
