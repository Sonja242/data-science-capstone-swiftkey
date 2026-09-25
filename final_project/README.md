# Sonja Next Word

Author: **Sonja Sahebzad**. Sonja Projects.

A CPU Shiny product for English next-word completion. It accepts a phrase, returns one primary word and two alternatives, and lets the user append a suggestion. The app documents its own use and measured limitations.

- [Live Shiny app](https://sonjasahebzad.shinyapps.io/sonja-next-word/)
- [Five-slide pitch on RPubs](https://rpubs.com/Sonja_Janssen/sonja-next-word); `Next_Word_Pitch.Rpres` is the RStudio Presenter source.
- `Final_Product_Report.Rmd` and knitted HTML: concise methods, results, discussion, conclusion and references.
- `results/`: frozen selection, metrics, case hashes, test details and verification.

## Run the product in RStudio

Open `Sonja Next Word.Rproj`. Use R 4.5 or later. Install missing dependencies once:

```r
install.packages(c("shiny", "data.table", "stringi", "jsonlite"))
shiny::runApp("app")
```

The model is included. No GPU, raw corpus, Python runtime, API key or retraining is required. Model loading is once per R process. The compressed model is about 19.2 MiB; the R model object is about 172.6 MiB before indexing and 215.5 MiB prepared for faster inference. Allow additional memory for R, dependencies and concurrent sessions.

## User documentation

Type up to 500 characters of English text. With automatic updates enabled, suggestions refresh after a space or punctuation and a pause of at least 150 ms. A partial word does not trigger a next-word prediction. Without a trailing space, select **Predict next word** or press Ctrl + Enter / Command + Enter. Select a suggestion to append it with a space and continue automatically. Clear resets the text. Example buttons start an automatic prediction. Disable automatic updates to use the button only.

The context label counts actual input words, excluding start-of-text markers. The total input word count is displayed separately. Only the last four normalized words supply context. Unseen contexts fall back to shorter patterns or general frequencies. This product does not correct spelling or reason over a whole sentence. A small explicit profanity blocklist is not a comprehensive safety filter. Input is processed on the hosting server; application code does not persist phrases or call external prediction services.

## Reproduce or audit

The shipped model checksum is in `results/selection.json`. `scripts/build_and_evaluate.R` records its protocol before model comparison, selects among three compact variants on 600 development cases, and evaluates the frozen choice on 900 unused test cases. Raw evaluation phrases remain in the ignored parent `data/final_product` folder; public case metrics contain only source labels, hashes and outcomes.

Full rebuilding requires the original official corpus and the earlier project outputs `models/selected_predictor_v3.rds`, `models/expanded_partitions_v3.rds`, development cases and previous-test reservation manifests. Generate these with the existing numbered project pipelines. Run the final build in a fresh copy of this folder without prior generated results, preserving completed experiments. The script refuses to replace a finished result. Exact metrics depend on the recorded source artifacts and package versions, not only the code.

`scripts/verify_app.R` checks dense/sparse equivalence and server behavior, then records a separate timing pass using a high-resolution clock. It requires the local reserved-case CSV and updates timing fields only. The initial Windows process timer was too coarse for individual predictions; its measurements remain distinguishable from the refined final latency report. Timing varies by hardware and server load.

Knit `Final_Product_Report.Rmd` with RStudio's Knit button. Open `Next_Word_Pitch.Rpres` and select Preview to use RStudio Presenter. Export as a standalone webpage for RPubs. Both publication artifacts must correspond to the evaluated app/model checksum.

### Continuous-writing verification

`scripts/verify_continuous_writing.R` compares predictions and scores with the preserved v1 predictor on the 600 development and 900 existing test examples. All 1,500 outputs are identical. These are regression checks, not a new independent accuracy test. It also checks automatic refresh, partial words, rapid edits, punctuation, clearing a pending timer, manual mode and input limits. The UI guards against selecting suggestions from older input. See `results/continuous-writing.json` for the recorded version and checksums. The model, normalization and ranking are unchanged; informal corpus spellings can still be suggested.

### Speed and quality follow-up

Automatic waiting was reduced from 400 to 150 ms without changing predictions. Three one-setting compression changes were compared on the same 600 development cases. The best challenger produced 166 correct top-three results versus 164 for the current model and missed the predeclared minimum gain of six cases. Production is retained; 900 newly reserved cases remain unscored. The [follow-up report](https://sonja242.github.io/data-science-capstone-swiftkey/speed-quality-check.html), `research/speed-quality-20260925/compare_quality.R`, protocol and case-level metrics document the comparison. These development percentages are not replacement test accuracy.

### Exact indexed inference

A sorted numeric lookup reduced median local computation from 1.92 to 0.58 ms (69.9%) in five randomized interleaved rounds on 600 development phrases. The words, scores and metadata were identical on 5,510 checked phrases; 30 cases also matched dense scoring. Original independent accuracy remains 17.0% top-one and 28.6% top-three. The 150 ms typing delay is separate. Runtime timing and prepared memory are recorded in `app/runtime-metrics.json`; the original accuracy file is unchanged. The [study report](https://sonja242.github.io/data-science-capstone-swiftkey/context-speed-check.html) and `research/context-speed-20260925` contain the preregistered protocol, reproducible scripts, a rejected repetition-cache comparison and results. No quality challenger qualified, so the newly reserved 900-case test remains unscored.

### Dictionary trial

The [dictionary trial](https://sonja242.github.io/data-science-capstone-swiftkey/dictionary-check.html) tested American and British English Hunspell dictionaries as a ranking aid. All three penalty strengths retained 164/600 top-three successes. Of 436 misses, 424 target words were already in the model vocabulary and 12 were absent. Dictionary recognition is not next-word accuracy. The dictionaries were not added to the deployed app and add no runtime dependency. The protocol, exact dictionary checksums, error diagnosis, source R Markdown and results are in `research/dictionary-check-20260925`; the new quality test remains unused.

## Deploy

With an authorized rsconnect account, deploy exactly these files:

```r
rsconnect::deployApp(
  appDir = "app",
  appFiles = c("app.R", "predictor.R", "model.rds", "metrics.json", "runtime-metrics.json", "www/styles.css", "www/input.js"),
  appName = "sonja-next-word",
  account = "sonjasahebzad", server = "shinyapps.io"
)
```

Keep account tokens and `rsconnect/` records private. Never upload the raw corpus, private assessment files or full quiz reports. A public visitor needs no account to use the app or deck.

## Evidence and scope

Final first-choice accuracy is 153/900 (17.0%); top-three accuracy is 257/900 (28.6%). Approximate Wilson 95% intervals are 14.7% to 19.6% and 25.7% to 31.6%, respectively. All cases return suggestions, which is availability rather than correctness. Scores are not calibrated confidence. Equal source weighting, short context and possible near-duplicate text limit generalization. Prior assisted quiz scores and GPU model results are not the accuracy of this app.

Data: [official SwiftKey corpus](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip). Method background: [Chen and Goodman (1996)](https://aclanthology.org/P96-1041/). Framework: [Posit Shiny](https://shiny.posit.co/).
