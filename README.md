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
| `22_prepare_learning_curve.R` | Reserve a new final sample with no exact-line overlap against training or earlier evaluation cases |
| `23_learning_curve_report.Rmd` | Three-seed learning curves, independent ranking audit and one prespecified final comparison |
| `24_prepare_word_generation.R` | Reserve and audit a new 900-case final test |
| `25_word_generation_report.Rmd` | Controlled complete-word generation, paired outcomes and independent verification |
| `26_try_complete_words.R` | Compare original and experimental complete-word suggestions in RStudio |
| `27_target_rank_diagnostic.R` | Verify the development-only known-target diagnosis and knit its report |
| `28_target_rank_diagnostic.Rmd` | Target-aware ranks, FP32 sensitivity, interpretation and reproducibility |
| `29_prepare_ranking_study.R` | Reserve separate confidence-calibration and final-test lines |
| `30_ranking_and_calibration.Rmd` | Fixed ranking rules, first-suggestion confidence and accuracy versus coverage |
| `31_verify_ranking_study.R` | Independently verify saved outcomes, print results and knit the report |

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


## Longer training and independent audits

The [learning-curve report](https://sonja242.github.io/data-science-capstone-swiftkey/learning-curves.html) compares 256, 512 and 1,024 updates, two candidate-list widths and three random seeds. It reuses a fixed development set for selection and reserves a fresh 900-case test. The ranking audit distinguishes missing candidates from words ranked below the first three, including the limitation on words represented by several model tokens.

See [the complete reproduction guide](python/LEARNING_CURVE.md), [ranking findings](research/ranking-audit.md), [training specification](python/LEARNING_CURVE_TRAINING.md) and [independent validation review](python/LEARNING_CURVE_VALIDATION.md). Selection, case identifiers, adapter hashes and paired test outcomes remain auditable. The default RStudio entry point stays `18_try_neural_predictor.R`; `23_learning_curve_report.Rmd` knits from saved tables without retraining.

On the same new 900-case test, the development-selected representative and the prior operational model both achieved **329/900 = 36.6% top-three accuracy**. The paired difference was **0.0 percentage points**, with 95% interval **-0.8 to +0.8** and exact McNemar **p = 1.00**. The new variant used about **38% less inference time** (176 versus 283 ms), but did not pass the prespecified accuracy-improvement rule. The prior operational model therefore remains the default. An explicitly labeled experimental option is available with `candidate="learning_curve"`.

The selected representative scored **87.3% on synthetic four-choice cases**. This is not a quiz score, and the 85% free-text target remains unmet. The learning curves show no top-three gain from increasing this fixed-pool training schedule from 256 to 1,024 updates. Generating complete candidate words made of several model tokens is the next targeted research question.

## Complete words across multiple model tokens

The [complete-word report](https://sonja242.github.io/data-science-capstone-swiftkey/word-generation.html) tests bounded token-path search while preserving model weights, context and every original candidate score. Development selects between two new search widths. The separately reserved 900-case final test increases coverage from **81.0% to 82.3%**, but exact top-three accuracy only changes from **355/900 (39.4%) to 357/900 (39.7%)**: two gains, zero losses.

The paired difference is **+0.22 percentage points**, with a source-stratified 95% bootstrap interval of **0.00 to +0.56** and exact McNemar **p = 0.50**. Mean measured runtime increases from **326 to 780 ms**. The accuracy promotion rule fails, so the existing default remains unchanged. Both arms score **89.6% on synthetic four-choice cases**, a separate task that is not a quiz score or free-word accuracy.

The [reproduction guide](python/WORD_GENERATION.md), [compact final predictions](models/word_generation_final_cases.csv), [complete scores](models/word_generation_final.json) and [independent audit](models/word_generation_independent_audit.json) make the conclusion inspectable. Open `25_word_generation_report.Rmd` to knit the saved results, or source `26_try_complete_words.R` to try the research comparison. The standard interface remains `18_try_neural_predictor.R`. The 85% free-text target is not reached.

## Target-aware diagnosis: retrieval or ranking?

The [new diagnostic report](https://sonja242.github.io/data-science-capstone-swiftkey/target-rank-diagnostic.html) deliberately supplies the known target on the existing 600 development examples. It is not a production-performance estimate or new final test. In the existing pool, 222 targets already rank in the first three, 245 are available but rank lower, and 133 are missing. Adding all missing targets yields **zero additional top-three successes** under the current score rule. The complete-word pool also gains none.

FP32 checks cover all 16 cases within the predeclared boundary rule. The existing pool has one top-three exit and one entry; these concern already available words. No missing target was near enough to meet that check rule. The model and ranking criterion are therefore the next research priority for this setting, while the production default remains unchanged.

See the [reproduction guide](python/TARGET_RANK_DIAGNOSTIC.md), [case-level rank table](models/target_rank_cases.csv), [FP32 candidate scores](models/target_rank_fp32.json) and [independent audit](models/target_rank_independent_audit.json). The RStudio report is `28_target_rank_diagnostic.Rmd`; source `27_target_rank_diagnostic.R` to verify the saved evidence and rebuild its HTML.


## Fixed ranking rules and first-suggestion confidence

The [ranking and calibration report](https://sonja242.github.io/data-science-capstone-swiftkey/ranking-calibration.html) compares seven predefined scoring rules while keeping model weights and candidates fixed. None improves the registered development top-three criterion, so the original ranking is retained. A separate set of 900 examples fits confidence for **first-suggestion correctness**, with top-three performance measured separately.

On **900 new final-test examples**, the retained model achieves **23.8% first-suggestion accuracy** and **37.1% top-three accuracy**. Calibration changes the test Brier score from **0.1498 to 0.1460**, without changing predictions. At an estimated-confidence threshold of 95%, only **4/900** cases are answered; their measured first-suggestion accuracy is 100.0% (4/4). The report supplies Wilson intervals and lower-threshold results, so subset accuracy is never presented as overall accuracy or a service guarantee.

The [reproduction guide](python/RANKING_STUDY.md), [case-level predictions](models/ranking_study_final_cases.csv), [fixed protocol](models/ranking_study_protocol.json) and [independent final audit](models/ranking_study_independent_audit.json) document selection before calibration and final testing. An independent R optimizer also verifies the calibration fit. Open `30_ranking_and_calibration.Rmd`, or source `31_verify_ranking_study.R` in RStudio to verify and knit saved results. Confidence remains a research assessment; the operational interface `18_try_neural_predictor.R` is unchanged.
