# Sonja PhraseFlow 1.3: experiment record

Author: Sonja Sahebzad. Location: Utrecht, the Netherlands. Sonja Projects. 26 September 2026.

This extension adds Dutch, French, Korean, Mandarin Chinese, Portuguese, Italian, Hindi and Spanish to the existing English, German, Finnish and Russian preview. The English production model is unchanged. New models are experimental. No larger neural training, production deployment or course submission was performed.

## Saved evidence

`protocol.json` fixes model settings, sampling and evaluation units. `source_manifest.json` records official download URLs, source dates and SHA-256 hashes. `metrics.json` and the eight `*_case_metrics.csv` files contain the evaluation summaries and anonymized per-example outcomes. Each language has 600 held-out sentences; candidates are generated from prefixes only. Private prefixes, targets and source text remain under the project `data/languages_20260926_v13` directory. The reserved English 900-case file was not opened.

`checks.json` records the successful Shiny tests and exact comparison of words and scores on all 600 English development cases. `release_check.json` distinguishes automated checks from visual browser review. `release_sha256.json` identifies all release files. It should be regenerated after deliberate artifact changes.

## Reproducing safely

Run commands from the Data Science Capstone project root, with R 4.6.1 and the packages recorded in `session-info.txt`. Preserve the existing completed experiment. Use a separate project copy for retraining; never remove completed metrics to bypass the overwrite guard in place.

The one-time build order was `prepare_extension.py`, `extend_tokenizer.py`, `train_extension.R`, `update_app.py`, `fix_language_reset.py` and `final_ui_copy.py`. The first script creates the v1.2 backup and freezes eight official exports; it intentionally refuses existing output directories. To replicate the original training data in a clean project copy, reuse the saved sampled CSV files and verify the original export hashes instead of assuming weekly exports remain identical. `extend_tokenizer.py` also records the final Chinese prefix-only segmentation protocol. The scripts use paths relative to the project root and are one-time migration scripts, not an idempotent installer.

The current working implementation is also retained directly in `app.R`, `predictor.R`, `input.js` and `styles.css`. The core training function is in the project `R/predictive_model_v3.R`; compaction and interval functions come from `final_project/research/multilingual-20260926/build_multilingual.R`, as explicitly loaded in `train_extension.R`.

To repeat functional checks without retraining, run `Rscript --vanilla final_project/research/languages-20260926-v13/test_extension.R`. This checks all languages, Unicode handling, composition pause, cleared state on a language change, the cache bound and unchanged English development predictions. It reads development data only. Browser checks separately exercised Dutch, Hindi and Chinese, including appending Chinese suggestions without spaces. Native IME end-to-end interaction and native-speaker validation remain future work.

## Figures, guide and native presentation

`build_figures.R` creates scientific figures from stored aggregate metrics; `figure_values.csv` records plotted values. Run `rmarkdown::render('final_project/phraseflow_multilingual/Multilingual_Guide.Rmd')` and copy the knitted HTML to the app's `www` directory. The HTML is self-contained. Keep the source beside the output.

`PhraseFlow_Pitch.Rpres` is the five-slide RStudio Presenter source. In RStudio, open it with `.rs.showPresentation()` and use More > Save As Web Page. The verified output is `PhraseFlow_Pitch.html`. Its three images are embedded. The similarly named `PhraseFlow_Pitch-rpubs.html` is an older local export, not the 1.3 deck.

The app's `Sonja PhraseFlow.Rproj` opens the project. Run `app.R` for the local twelve-language preview. Serve the deck and sibling guide together for local link testing. Before publishing a standalone deck to RPubs, replace relative guide and data-credit links with verified public URLs. The original English PDF remains a historical 1.1 artifact; the current twelve-language guide is the knitted HTML.

## Interpretation

First-suggestion and top-three accuracy are exact matches to recorded next words. Wilson 95% intervals describe sampling uncertainty, not calibrated confidence in a suggestion. Tatoeba differs from course news, blogs and Twitter; different language units and training sizes prevent a language ranking. Exact duplicates were removed, but similar templates may remain. Korean and Hindi have smaller training samples. Performance on representative everyday writing and review by native speakers are not yet established.
