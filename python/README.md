# Local neural comparison

Author: **Sonja Sahebzad**

This is an optional GPU experiment alongside the corpus-trained R model. It uses
externally pretrained Apache-2.0 Qwen3 Base models. External pretraining overlap
with the public SwiftKey corpus is unknown. The reported four-choice accuracy
uses synthetic distractors and is not a Coursera score.

## Environment

The recorded run uses Windows, Python 3.13, PyTorch 2.8.0 with CUDA 12.8,
Transformers 4.56.2 and an NVIDIA RTX 2000 Ada laptop GPU with 8 GB VRAM.
Allow approximately 12 GB of disk space for the environment and model files.
The R training pipeline additionally requires substantial RAM and disk space.

Run these PowerShell commands from the project directory. An existing working
environment can be reused; the models and data are not committed to Git.

```powershell
py -3.13 -m venv .venv-neural
& '.venv-neural/Scripts/python.exe' -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& '.venv-neural/Scripts/python.exe' -m pip install -r python/requirements-neural-lock.txt
& '.venv-neural/Scripts/python.exe' python/download_pinned_models.py --size 0.6B
& '.venv-neural/Scripts/python.exe' python/download_pinned_models.py --size 1.7B
```

`download_pinned_models.py` fixes both repository revisions, so later repository
changes do not change the models in a reproduction.

## Reproduce the comparison

The committed result files allow the report to be knitted without retraining.
For a complete fresh rerun, first archive the existing `models/neural_*.json` and
`models/neural_comparison_*.csv` files in a dated backup folder. Keep the downloaded
`models/neural/` directory and the local R model objects. This avoids mixing saved
results with regenerated files whose byte hashes may differ by platform or line endings.

1. Rebuild the local n-gram pipeline with `15_train_expanded_predictor.R`.
2. Run `source("17_prepare_neural_comparison.R")` in RStudio to prepare the same
   development/test rows and synthetic choices. The fixed seeds exclude prior
   evaluation rows and retain separate local training partitions.
3. Run the two development evaluations sequentially, then freeze the selection
   and evaluate its reserved test set:

```powershell
& '.venv-neural/Scripts/python.exe' python/neural_predictor.py --size 0.6B --evaluate development
& '.venv-neural/Scripts/python.exe' python/neural_predictor.py --size 1.7B --evaluate development
& '.venv-neural/Scripts/python.exe' python/run_neural_comparison.py
```

4. Knit `19_neural_predictor_evaluation.Rmd`. It reads the saved aggregate results,
   so inspecting or knitting the report does not repeat model inference.

Selection uses development top-three accuracy, then synthetic four-choice
accuracy, then latency. The final script verifies code/data hashes and row
alignment and rejects a changed selection when a selection file already exists.
An unchanged completed test artifact is reused rather than silently rerun.
Small numerical variation is possible across GPU hardware or software versions.

## Try a sentence in RStudio

Open `18_try_neural_predictor.R` and click **Source**. At the Console prompt, type
only the English phrase. At the next prompt, optionally enter four words separated
by commas. The first run loads the model and can take several seconds.

The Console prints three free-text suggestions, followed by a separate ranking
of supplied options. Model log scores are not percentages of answer correctness.
The existing `13_try_predictor.R` remains available for the lightweight R-only model.

## Scoring details

The neural model proposes up to 64 single-token word candidates that occur in the
local training vocabulary. Up to 20 R-model candidates add multi-token words.
Candidates are lowercased English words with optional internal apostrophes.
Each complete candidate is scored by its canonical token sequence plus the
probability that the following token begins at a word boundary. Candidates are
scored in batches using a cached prefix; a full-sequence calculation independently
checks the cache path before every evaluation.

Only the last 128 model tokens of context are used. The shortlist is finite and
does not exhaust all possible words. Its measured target recall is an upper bound
on its top-three accuracy. Supplied choices cannot be added to the free-text
shortlist. No hidden continuation from the original corpus is consulted at inference.

Evaluation files preserve one random word position per sampled line, equal
source weighting and paired outcomes. The report includes Wilson intervals,
a paired source-stratified bootstrap, source-specific accuracy, runtime and
limitations. It does not compare token perplexity with word-level n-gram perplexity.
