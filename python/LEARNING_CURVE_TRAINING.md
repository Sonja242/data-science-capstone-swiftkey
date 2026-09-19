# Local corpus learning curves

Author: **Sonja Sahebzad**

This experiment asks whether more training on the existing official-text pool
improves exact next-word prediction. It changes training duration and random seed,
not the source data. It does not assume that an accuracy target will be reached.

## Fixed trajectories

Three seeds, `20261410`, `20261411` and `20261412`, each train one trajectory with
checkpoints at 256, 512 and 1024 optimizer updates. Each seed controls adapter
initialization, dropout, and a permutation of the same 11,282 cached local blocks.
The first 8192 blocks are used without repetition. Checkpoints are nested stages
of one run, not independent models trained with separate schedules.

The original 48,000-line pool contains 24,000 Twitter, 12,000 blog and 12,000 news
lines from the training partition. The original normalized-text cache is retained,
including its fixed document shuffle, EOS separators and 128-token packing.
Attention can cross EOS separators, and the final incomplete block was discarded.
No external conversations are used. This trainer does not read final-test cases.
It only hashes the fixed 600-case development CSV for an audit fingerprint;
evaluation is performed separately after training.

Each trajectory uses the frozen Qwen3-1.7B-Base revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1` and 1,605,632 trainable LoRA parameters:
rank 8, alpha 16, dropout 0.05 and query/value projection targets. Optimization
uses AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01, norm clipping at 1,
effective batch 8, BF16, SDPA and gradient checkpointing without reentrant mode.
Only adapter parameters receive gradients. Microbatch 2, 4 or 8 may be chosen
using training-only memory/timing pilots, then must stay fixed across official runs.
Larger microbatches change dropout draws and floating-point summation, so they
should not be mixed in a seed comparison.

The learning rate rises linearly to 0.0001 at update 16 and follows a common
cosine decay to zero at update 1024. The 256-update checkpoint is consequently
different from the earlier experiment's complete 256-update cosine schedule.
Compare stages within these new trajectories. An earlier adapter can remain a
separate baseline, but it is not an identical 256-update control.

| Checkpoint | Distinct blocks consumed | Input tokens | Supervised next-token positions |
| --- | ---: | ---: | ---: |
| 256 | 2048 | 262144 | 260096 |
| 512 | 4096 | 524288 | 520192 |
| 1024 | 8192 | 1048576 | 1040384 |

The first token of every block is context only, accounting for the difference
between input tokens and supervised positions. Training loss includes EOS tokens
and is not an exact-word accuracy measure.

## Reproduction and safety

Use the existing Python environment recorded in `requirements-adaptation-lock.txt`.
From PowerShell, with the live project selected as the working directory:

```powershell
& '.venv-neural/Scripts/python.exe' python/learning_curve_training.py --check-only --microbatch 8
& '.venv-neural/Scripts/python.exe' tests/test_learning_curve_training.py
# Optional training-only timing/memory pilot, saved separately:
& '.venv-neural/Scripts/python.exe' python/learning_curve_training.py --steps 8 --microbatch 2 --run-label mb2
# Run official seeds serially with the recorded microbatch eight:
foreach ($seed in @(20261410,20261411,20261412)) {
  & '.venv-neural/Scripts/python.exe' python/learning_curve_training.py --seed $seed --microbatch 8
  if ($LASTEXITCODE -ne 0) { throw 'Stop the queue and inspect the failed run.' }
}
```

Use `--project-root 'C:/path/to/Data Science Capstone'` when the script is invoked
from a separate worktree. The preflight hashes source text, token cache, cache
metadata, development CSV, base configuration/tokenizer and base weights. It
checks the immutable revision, expected block dimensions and token vocabulary.
The fixed code, complete block schedule and consumed prefix have separate hashes.
Hashes identify the local downloaded weights; their revision attribution relies
on the existing pinned downloader's manifest.

Official adapters are saved below
`models/neural/learning_curve_seed_<seed>/step_<step>`. The per-checkpoint metadata
records all hyperparameters, software/hardware details, actual tokens consumed,
loss history, timing excluding checkpoint I/O, peak allocated/reserved GPU memory,
and adapter/configuration hashes. The complete schedule and initial protocol
remain in the trajectory directory. A public summary is written to
`models/learning_curve_training_seed_<seed>.json` only after completion.
Raw text, caches and adapter weights remain excluded from Git.

The trainer refuses existing run directories or public summaries. It never
automatically resumes or overwrites a partial run. A failure record preserves
the error and any completed checkpoints. Inspect and explicitly archive the
entire failed run before a fresh attempt. Pilot labels isolate short checks
from official trajectories; pilots always retain the common 1024-update horizon.

Seeded operations and deterministic settings improve reproducibility. SDPA/CUDA
kernels and hardware/software versions may still prevent bitwise reproduction;
the manifest records that limitation. Run GPU jobs serially to avoid resource
contention and misleading latency comparisons.

## Evaluation responsibilities

Use the same fixed development cases and definitions for all three seeds and
three stages. Report exact top-1 and top-3 accuracy, candidate coverage, separately
labelled synthetic four-choice accuracy, and latency. Show variation across seeds;
do not pick the best seed after looking at final-test results. Candidate-list
width is a separate development comparison. Freeze the selection rule before
opening a newly reserved final test. The trainer has no test-set selection logic.

## References

- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base).
- [Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [PEFT LoRA configuration documentation](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora).
- [Official Coursera SwiftKey corpus](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
