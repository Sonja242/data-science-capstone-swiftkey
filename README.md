# Data Science Capstone: SwiftKey Corpus Analysis

**Author:** Sonja Sahebzad  
**Tools:** R, R Markdown, data.table, stringi, ggplot2, Python and PyTorch

This repository contains a reproducible workflow for the Johns Hopkins University Data Science Capstone. It explores the official English SwiftKey corpus and compares experimental next-word predictors on separate training, validation and test data. Predictions are fallible; this project is not an answer key.

## View the finished reports

Open the published [Data Science Capstone report website](https://sonja242.github.io/data-science-capstone-swiftkey/). The site renders the knitted HTML reports as complete web pages. The `.Rmd` files remain available as reproducible source code.

## What the project demonstrates

- inspecting and cleaning the official English corpus;
- processing large text files with memory-conscious methods;
- reproducible sampling, deduplication and train-validation-test separation;
- building frequency tables up to five words and comparing training sizes;
- combining long and short contexts in one normalized probability distribution;
- measuring free-text accuracy and synthetic four-choice accuracy separately;
- measuring mean reciprocal rank, perplexity, response time and model size;
- saving a selected model for an interactive Shiny prototype.

## Main reports

| File | Purpose |
|---|---|
| `01_getting_started.Rmd` | Initial corpus inspection and project setup |
| `03_tokenization_cleaning_case_v4.Rmd` | Tokenization and cleaning case study |
| `04_professional_corpus_pipeline.Rmd` | Reusable, memory-conscious corpus pipeline |
| `10_quiz1_reproducible_verification.Rmd` | Reproducible verification of introductory corpus operations |
| `11_milestone_report.Rmd` | Exploratory analysis and prediction product plan |
| `12_predictive_model_evaluation.Rmd` | Archived v2 evaluation and limitations |
| `13_try_predictor.R` | Interactive console interface for testing any short English phrase |
| `14_rebuild_prediction_v2.R` | Rebuild and compare predictors, then evaluate the selected model on reserved test data |
| `15_train_expanded_predictor.R` | Train six expanded candidates and evaluate on fresh test lines |
| `16_expanded_predictor_evaluation.Rmd` | Corpus-trained model with separate free-text and four-choice results |
| `17_prepare_neural_comparison.R` | Prepare fresh development and test cases without changing local training |
| `18_try_neural_predictor.R` | Run the optional local GPU model from the RStudio Console |
| `19_neural_predictor_evaluation.Rmd` | Initial neural comparison, uncertainty intervals and the 85% target assessment |
| `20_prepare_adaptation.R` | Prepare matched training pools and fresh holdout examples |
| `21_data_adaptation_report.Rmd` | Error analysis, two LoRA training recipes and a controlled extra-data comparison |

The knitted `.html` files preserve the corresponding code and printed results.

## Reproduce the analysis

1. Download the official [Coursera SwiftKey dataset](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
2. Extract it so the English files are available under `data/final/en_US/`.
3. Open `Data Science Capstone.Rproj` in RStudio.
4. Run `source("15_train_expanded_predictor.R")` to train and compare the expanded candidates. It rebuilds the v2 baseline if required. Then knit `16_expanded_predictor_evaluation.Rmd`. The full build requires substantial time, RAM and disk space; its caches are reused.
5. Run `source("13_try_predictor.R")` in the RStudio Console to try the selected predictor.

The corpus, ZIP archive and generated `.rds` model objects are intentionally excluded from Git. The model objects can be rebuilt from the official data and the committed source code.

## Reusable R code

The current predictor is in `R/predictive_model_v3.R`. Results are in `models/expanded_validation_v3.csv`, `models/expanded_test_v3.csv` and `models/expanded_by_source_v3.csv`. Four-choice scores use synthetic frequency-matched alternatives; they are not actual quiz scores. The earlier v2 source and reports remain available. `R/predictive_model.R` and `models/candidate_evaluation.csv` describe the superseded v1 approach; its scores must not be compared directly with the new test scores because the data and evaluation design changed. The optional raw-corpus lookup from v1 is no longer used by the interactive script.

Run `source("tests/test_predictive_model_v3.R")` from the project to check probability normalization, ranking, rare-context smoothing, punctuation and unseen-word handling. Model estimates are not calibrated confidence that an answer is correct. Optional user-supplied choices are ranked separately from unrestricted predictions.

## Notes

Results depend on the official Coursera corpus version and the documented random seeds. The selected model is a transparent benchmark for further improvement, profanity filtering and integration into a mobile-friendly Shiny application.

## Optional neural experiment

The [neural model report](https://sonja242.github.io/data-science-capstone-swiftkey/neural-model.html) compares two pretrained Qwen3 Base models with the local n-gram model. It reports free-text and synthetic four-choice scores separately, with evidence for whether the requested 85% threshold is met. External pretraining overlap with the public corpus is unknown. See [the Python setup and reproduction instructions](python/README.md) for pinned model revisions, exact packages, GPU requirements and evaluation commands.

In RStudio, open `18_try_neural_predictor.R` and click Source to enter a phrase and optional comma-separated answer choices. The lightweight R interface remains in `13_try_predictor.R`. Neither interface promises that its first suggestion is correct.

## Does extra training data help?

The [data-adaptation report](https://sonja242.github.io/data-science-capstone-swiftkey/data-adaptation.html) compares candidate-list expansion, adaptation to official training texts, and a training mixture with Google Taskmaster-1 conversations (CC BY 4.0, with attribution). Four candidates are compared using new development and test lines, with selection made before final testing. The report distinguishes measured gains from inconclusive results and retains separate free-text and synthetic four-choice scores. See [the full reproduction guide](python/ADAPTATION.md).

On 900 fresh test cases, local adaptation raises free-text top-three accuracy from **34.1% to 36.2%** (paired 95% interval for the gain: **0.7 to 3.6 percentage points**). Synthetic four-choice accuracy is **85.7%**. The extra dialogue mixture does not establish a further benefit, and the 85% free-text target remains unmet.

The existing `18_try_neural_predictor.R` remains the entry point. It uses the candidate admitted by the documented final-test promotion rule, with an explicit option to try experimental candidates. Earlier evaluation reports and source files remain available.
