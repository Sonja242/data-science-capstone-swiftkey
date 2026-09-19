# Controlled adaptation experiment

Author: **Sonja Sahebzad**

This extends the local neural experiment without altering its archived results.
The question is whether expanding candidate coverage, adapting to the official
corpus, or adding English task conversations improves exact next-word prediction.
The requested 85% and 99.8% targets are evaluated, not assumed.

## Data and attribution

Use the official Coursera corpus and existing v3 partitions. The additional
source is **Taskmaster-1**, by Bill Byrne and colleagues at Google LLC (2019),
licensed **CC BY 4.0**. Only official training conversation IDs from the self-dialog
subset are used. Normalization, deduplication and filtering modify the data.

- Paper: https://research.google/pubs/taskmaster-1-toward-a-realistic-and-diverse-dialog-dataset/
- Documentation: https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/README.md
- License: https://creativecommons.org/licenses/by/4.0/

Taskmaster is task-oriented conversation, not a representative sample of tweets.
The experiment tests this possible mismatch. Complete normalized matches to any
official local holdout line are excluded; near duplicates may remain. Unknown
overlap with Qwen's external pretraining also remains a limitation.

## Prepare and train

Use the existing Windows Python 3.13 CUDA environment from `python/README.md`.
The exact expanded environment is recorded in `requirements-adaptation-lock.txt`.
Download the pinned source and run data preparation from the project directory:

```powershell
& '.venv-neural/Scripts/python.exe' -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r python/requirements-adaptation-lock.txt
& '.venv-neural/Scripts/python.exe' python/download_taskmaster.py
```

```r
source("20_prepare_adaptation.R")
```

The R preparation exports 48,000 official training lines, eligible Taskmaster
training utterances, 600 development cases and 900 new final cases. Prior v3 and
neural evaluation lines are excluded from the new development and test sets.
Raw text, token caches and model weights remain in ignored local directories.

```powershell
& '.venv-neural/Scripts/python.exe' python/train_adaptation.py --recipe local
& '.venv-neural/Scripts/python.exe' python/train_adaptation.py --recipe augmented
```

Both recipes use the same frozen Qwen3-1.7B-Base revision, 256 AdamW steps,
effective batch 8, 128-token blocks, LoRA rank 8, alpha 16, dropout 0.05,
query/value projections, learning rate peaking at 0.0001, 16 warmup steps,
cosine decay and gradient clipping at 1. Only adapter parameters receive gradients.
Training uses bfloat16 on the GPU and gradient checkpointing. Inference uses the
same float16 base as the earlier experiment, with the learned adapter merged.

Each recipe consumes 262,144 input tokens and 260,096 supervised positions.
Documents are shuffled with fixed seeds and joined using end-of-text tokens,
then packed into blocks. Attention can cross those separators. The augmented
recipe replaces two of the eight blocks at every update with Taskmaster blocks;
the other six local blocks match the local-only recipe. This is a matched-compute
comparison of data mixtures, not an exhaustive use of the full training pools.

Training refuses to overwrite an existing adapter directory. Archive old adapters,
training manifests and evaluation outputs before intentionally starting a new run.

## Fixed comparison

`models/adaptation_protocol.json` records the design before final evaluation.
The four candidates are `base64`, `base256`, `local256`, and `augmented256`.
The number is the neural shortlist width; all also receive up to 20 local n-gram
candidates. The wider candidates use an inference batch size of 64; the existing
baseline retains its batch size of 32. Candidate scoring and the 128-token context
limit are otherwise unchanged. The supplied choices cannot alter the free list. The wider candidates use vectorized whole-word scoring, checked against the original implementation before the final test; the mathematical scoring rule is unchanged.

```powershell
& '.venv-neural/Scripts/python.exe' python/adaptation_experiment.py --diagnose
foreach ($candidate in @('base64','base256','local256','augmented256')) {
  & '.venv-neural/Scripts/python.exe' python/adaptation_experiment.py --evaluate $candidate --split development
}
& '.venv-neural/Scripts/python.exe' python/adaptation_experiment.py --compare
```

Selection uses development top-three accuracy, then synthetic four-choice accuracy,
then latency. All four final-test contrasts are predeclared; they are not a new
hyperparameter search. The paired selected-versus-baseline test determines promotion:
retain the previous default unless the 95% lower bound for the top-three gain is
above zero. Secondary source-stratified paired intervals are exploratory, without
multiplicity adjustment. Exact-word outcomes, rather than semantic judgments, define
success. The constant-word reference is chosen from official training frequencies.

Hashes verify the inference code, adapters, data and row identities. Identical
completed evaluations are reused. Existing saved artifacts should be archived
before a full fresh reproduction, especially after a change in platform or line
endings. The error analysis uses previous development data, not the new final test.
Run `tests/test_adaptation_artifacts.py` for cross-artifact leakage and hash checks;
`tests/test_adaptation_scoring.py` checks numerical equivalence on the GPU.

## Reports and interactive use

Knit `21_data_adaptation_report.Rmd` to view the tables, interactive chart,
discussion, conclusions and references. It reads saved results, so knitting does
not repeat training. Use `18_try_neural_predictor.R` in RStudio for the current
default predictor. The Python interface also allows explicit experimental use:

```powershell
& '.venv-neural/Scripts/python.exe' python/adaptation_experiment.py --candidate augmented256 --request request.json --output result.json
```

The request JSON contains a `phrase` string and optional `ngram_candidates` and
`choices` arrays. The R interface supplies n-gram candidates automatically and normalizes input using the same lowercase word tokenizer as training and evaluation.
Predictions are ranked suggestions, not calibrated percentages of correctness.
