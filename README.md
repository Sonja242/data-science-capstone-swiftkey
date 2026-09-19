# Data Science Capstone: SwiftKey Corpus Analysis

**Author:** Sonja Sahebzad  
**Tools:** R, R Markdown, data.table, stringi and ggplot2

This repository contains a reproducible workflow for the Johns Hopkins University Data Science Capstone. It explores the official English SwiftKey corpus and compares experimental next-word predictors on separate training, validation and test data. Predictions are fallible; this project is not an answer key.

## View the finished reports

Open the published [Data Science Capstone report website](https://sonja242.github.io/data-science-capstone-swiftkey/). The site renders the knitted HTML reports as complete web pages. The `.Rmd` files remain available as reproducible source code.

## What the project demonstrates

- inspecting and cleaning the official English corpus;
- processing large text files with memory-conscious methods;
- reproducible sampling, deduplication and train-validation-test separation;
- building unigram, bigram, trigram and four-gram frequency tables;
- combining long and short contexts in one normalized probability distribution;
- comparing pruning thresholds with top-1, top-2 and top-3 accuracy;
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
| `12_predictive_model_evaluation.Rmd` | Corrected evaluation, interpretation, limitations and references |
| `13_try_predictor.R` | Interactive console interface for testing any short English phrase |
| `14_rebuild_prediction_v2.R` | Rebuild and compare predictors, then evaluate the selected model on reserved test data |

The knitted `.html` files preserve the corresponding code and printed results.

## Reproduce the analysis

1. Download the official [Coursera SwiftKey dataset](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip).
2. Extract it so the English files are available under `data/final/en_US/`.
3. Open `Data Science Capstone.Rproj` in RStudio.
4. Run `source("14_rebuild_prediction_v2.R")` to rebuild and compare models, then knit `12_predictive_model_evaluation.Rmd` to view the report. Knitting can run the build if no v2 results exist.
5. Run `source("13_try_predictor.R")` in the RStudio Console to try the selected predictor.

The corpus, ZIP archive and generated `.rds` model objects are intentionally excluded from Git. The model objects can be rebuilt from the official data and the committed source code.

## Reusable R code

The current predictor is in `R/predictive_model_v2.R`. Results are in `models/prediction_validation_v2.csv`, `models/prediction_test_v2.csv` and `models/prediction_test_by_source_v2.csv`. `R/predictive_model.R` and `models/candidate_evaluation.csv` describe the superseded v1 approach; its scores must not be compared directly with the new test scores because the data and evaluation design changed. The optional raw-corpus lookup from v1 is no longer used by the interactive script.

Run `source("tests/test_predictive_model_v2.R")` from the project to check probability normalization, ranking, rare-context smoothing, punctuation and unseen-word handling. Model estimates are not calibrated confidence that an answer is correct. Optional user-supplied choices are ranked separately from unrestricted predictions.

## Notes

Results depend on the official Coursera corpus version and the documented random seeds. The first model is a transparent benchmark for further improvement, profanity filtering and integration into a mobile-friendly Shiny application.
