# Complete-word proposal experiment

Author: **Sonja Sahebzad**

This controlled study changes candidate generation while preserving the existing
operational `local256` model, adapter, context limit and complete-word scoring.
There is no additional training. The question is whether proposing more complete
words improves exact next-word prediction enough to justify their additional cost.

## What changes

The baseline retrieves the top 256 eligible token IDs and adds up to 20 local
n-gram words. Token aliases and lowercasing mean that 256 token IDs are not
necessarily 256 distinct words. The new generator retains every baseline word
and its exact numerical score, then proposes words from a trie built exclusively
from the existing training vocabulary. A trie is a tree of valid token prefixes;
only a terminal vocabulary word can become a new candidate.

The registered alternatives use beam widths 16 and 64, a six-token depth limit
and at most 128 completed proposals. The added trie contains multiple-token words;
single-token entries are not added through this route. Terminal words can remain
eligible for expansion when longer valid words share their prefix. No target,
answer option or validation word is supplied to the proposal generator.

Beam paths accumulate log probabilities normalized over the full model vocabulary,
not probabilities renormalized among a node's children. Completed proposals are
first capped by canonical token-path likelihood. New words are then scored using
the unchanged canonical likelihood plus the existing word-boundary probability
mass. This delayed boundary term is an approximation: a word pruned before final
scoring might have ranked higher afterward. Neither wider beams nor higher
coverage guarantee a better final top three.

The fixed training lexicon contains 503,523 word types. Of these, 476,776 have
canonical encodings of two to six tokens; 22,055 are single-token words, and
4,692 exceed the new depth limit. The resulting trie has 655,490 nodes, including
60,188 terminal nodes that also have children. The excluded 0.93% is a fraction
of vocabulary types, not the fraction of observed future words. Existing baseline
words remain available even if they exceed the new proposal-depth limit.

## Frozen comparison

`models/word_generation_protocol.json` records the design. All model/input files,
code and protocol are fingerprinted in `models/word_generation_implementation.json`.
The original scorer and earlier experiment sources remain unchanged.

The same 600 development cases are reused. Selection between beam 16 and beam 64
uses integer top-three successes, then top-one successes, then measured mean
latency, then the smaller beam. Only the chosen alternative is compared with the
unchanged baseline on a newly reserved set of 900 cases: 300 each from blogs,
news and Twitter. Previous evaluation cases and exact normalized training-line
overlaps are excluded. Repeated reuse can adapt decisions to the development set;
the new final set protects this specific final comparison, not future iterations.

The primary outcome is paired exact top-three accuracy. Promotion requires a
positive lower bound of the source-stratified paired bootstrap 95% interval and
an exact two-sided McNemar p-value below 0.05. The bootstrap uses 10,000 replicates
and seed 20261590. Top-one accuracy, coverage, gains/losses, gains from newly covered
words, token-length/source groups and runtime are descriptive secondary measures.
There is no post-hoc latency rule that can replace this registered promotion rule.

Answer options are a separate scoring task executed after free prediction.
Because weights and the option-scoring calculation are fixed, the two arms have
the same synthetic four-choice outcomes. This diagnostic cannot demonstrate a
benefit from the new free-text generator, and synthetic options are not quiz scores.

## Numerical and independent checks

The independent CPU review exercises trie termination, continuing terminal paths,
depth limits, deterministic ties, cumulative scoring, baseline preservation and
option isolation. Its 12 semantic tests pass. CPU tests cannot validate CUDA caches
or numerical precision; see `research/word-generation-review.md` for scope.

An initial neutral cached/full-sequence check failed the absolute log-score
tolerance of 0.001. The expanded neutral FP16 check found a maximum difference
of 0.015710830688476562. Repeating the calculation with the same merged weights
converted to FP32 reduced the maximum difference to 0.00001239776611328125,
supporting the scoring formula while identifying finite-precision sensitivity.
All neutral tested rankings agreed. The initial failure is retained in
`models/word_generation_neutral_precision_diagnostic.json`.

Before development outcomes were inspected, an FP16 absolute log-score tolerance
of 0.02 was registered for this numerical check. This is not a predictive accuracy
tolerance or permission to alter old scores. Every old free candidate and score
is preserved exactly in the candidate union, and supplying options leaves free
predictions unchanged. The limited neutral checks cannot prove equivalence for
all conceivable inputs.

## Reproduction

Use the pinned local environment from `requirements-adaptation-lock.txt`, the
official corpus and existing data/model preparation. Run from the project root.
These commands describe a fresh reproduction: scripts refuse to overwrite their
existing registrations, reserved holdout or completed outcome artifacts. Preserve
the complete previous experiment before intentionally creating a fresh run.

```r
source("24_prepare_word_generation.R")
```

```powershell
# Run the independent partition reconstruction, then one GPU process at a time.
& 'C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe' tests/audit_word_generation_split.R (Get-Location).Path models/word_generation_partition_audit.json
& '.venv-neural/Scripts/python.exe' python/word_generation_precision_check.py
& '.venv-neural/Scripts/python.exe' python/word_generation_experiment.py --freeze
& '.venv-neural/Scripts/python.exe' python/word_generation_experiment.py --evaluate development
& '.venv-neural/Scripts/python.exe' python/word_generation_experiment.py --select
& '.venv-neural/Scripts/python.exe' python/verify_word_generation.py --development
& '.venv-neural/Scripts/python.exe' python/analyze_word_generation_development.py
& '.venv-neural/Scripts/python.exe' python/diagnose_word_generation_batches.py
& '.venv-neural/Scripts/python.exe' python/word_generation_experiment.py --evaluate final
& '.venv-neural/Scripts/python.exe' python/summarize_word_generation.py
& '.venv-neural/Scripts/python.exe' python/verify_word_generation.py --final
```

`--freeze` validates neutral behavior and saves code/model/data fingerprints.
The final evaluator requires an existing selection with matching development and
implementation hashes. Do not rerun selection after viewing final outcomes.

Joint predictions share work only for efficiency. Standalone runtime is measured
separately, with CUDA synchronization and alternating condition order. It covers
the first 20 development cases per source, then every final case. Timed outputs
must equal the corresponding saved predictions. Measurements exclude model/trie
loading and R shortlist construction, so they are not cold-start app latency.

Case-level JSON retains source, line hash, observed target, canonical target-token
count, baseline/new candidate scores, rankings and options, without full prefixes.
These records permit independent recomputation of ranks, paired successes,
coverage and synthetic-choice outcomes. Aggregate tables and
`25_word_generation_report.Rmd` provide the readable report. Raw corpus files and
model weights remain excluded from Git. The normalized lexicon, bounded search,
canonical tokenization and unknown pretraining overlap remain limitations.

## Recorded outcome and RStudio entry points

The fresh final comparison adds 12 covered targets and two top-three successes,
with no top-three losses. Accuracy is 355/900 versus 357/900; the paired 95%
interval is 0.00 to +0.56 percentage points and exact McNemar p is 0.50.
The existing default remains in place. Runtime rises from 326 to 780 ms,
including independent synthetic-choice scoring after model loading.

The development audit retained 371 cross-batch differences above the neutral
0.02 log-score reference. Rechecking its largest case with identical merged
weights in FP32 reduced the difference to 0.00000954. The final audit retains
one separately batched comparison above the reference. These numerical
observations did not change selection, scoring or the promotion rule.

```r
# Rebuild the report from the saved, audited results.
rmarkdown::render("25_word_generation_report.Rmd")
# In the RStudio Console, compare suggestions for a new English phrase.
source("26_try_complete_words.R")
```

## References

- [Official Coursera SwiftKey corpus](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base), revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- [Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [Independent CPU review](../research/word-generation-review.md).
