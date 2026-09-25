# Sonja Next Word

Author: **Sonja Sahebzad**. Sonja Projects.

A CPU Shiny product for English next-word completion. It accepts a phrase, returns one primary word and two alternatives, and lets the user append a suggestion. The app documents its own use and measured limitations.

- [Live Shiny app](https://sonjasahebzad.shinyapps.io/sonja-next-word/)
- [Five-slide pitch on RPubs](https://rpubs.com/Sonja_Janssen/sonja-next-word); `Next_Word_Pitch.Rpres` is the RStudio Presenter source.
- `Final_Product_Report.Rmd` and knitted HTML: concise methods, results, discussion, conclusion and references.
- `results/`: frozen selection, metrics, case hashes, test details and verification.

## Run the product in RStudio

Open `Sonja Next Word.Rproj`. Install missing dependencies once:

```r
install.packages(c("shiny", "data.table", "stringi", "jsonlite"))
shiny::runApp("app")
```

The model is included. No GPU, raw corpus, Python runtime, API key or retraining is required. Model loading is once per R process. The compressed model is about 19.2 MiB; the R model object is about 172.6 MiB. Allow additional memory for R, dependencies and concurrent sessions.

## User documentation

Type up to 500 characters of English text, finishing the last word. Select **Predict next word** or press Ctrl + Enter / Command + Enter. Select the primary word or one of the two alternatives to append it, then predict again. Clear resets the text. Example buttons supply starting phrases.

Only the last four normalized words supply context. Unseen contexts fall back to shorter patterns or general frequencies. This product does not correct spelling or reason over a whole sentence. A small explicit profanity blocklist is not a comprehensive safety filter. Input is processed on the hosting server; application code does not persist phrases or call external prediction services.

## Reproduce or audit

The shipped model checksum is in `results/selection.json`. `scripts/build_and_evaluate.R` records its protocol before model comparison, selects among three compact variants on 600 development cases, and evaluates the frozen choice on 900 unused test cases. Raw evaluation phrases remain in the ignored parent `data/final_product` folder; public case metrics contain only source labels, hashes and outcomes.

Full rebuilding requires the original official corpus and the earlier project outputs `models/selected_predictor_v3.rds`, `models/expanded_partitions_v3.rds`, development cases and previous-test reservation manifests. Generate these with the existing numbered project pipelines. Run the final build in a fresh copy of this folder without prior generated results, preserving completed experiments. The script refuses to replace a finished result. Exact metrics depend on the recorded source artifacts and package versions, not only the code.

`scripts/verify_app.R` checks dense/sparse equivalence and server behavior, then records a separate timing pass using a high-resolution clock. It requires the local reserved-case CSV and updates timing fields only. The initial Windows process timer was too coarse for individual predictions; its measurements remain distinguishable from the refined final latency report. Timing varies by hardware and server load.

Knit `Final_Product_Report.Rmd` with RStudio's Knit button. Open `Next_Word_Pitch.Rpres` and select Preview to use RStudio Presenter. Export as a standalone webpage for RPubs. Both publication artifacts must correspond to the evaluated app/model checksum.

## Deploy

With an authorized rsconnect account, deploy exactly these files:

```r
rsconnect::deployApp(
  appDir = "app",
  appFiles = c("app.R", "predictor.R", "model.rds", "metrics.json", "www/styles.css"),
  appName = "sonja-next-word",
  account = "sonjasahebzad", server = "shinyapps.io"
)
```

Keep account tokens and `rsconnect/` records private. Never upload the raw corpus, private assessment files or full quiz reports. A public visitor needs no account to use the app or deck.

## Evidence and scope

Final first-choice accuracy is 153/900 (17.0%); top-three accuracy is 257/900 (28.6%). Approximate Wilson 95% intervals are 14.7% to 19.6% and 25.7% to 31.6%, respectively. All cases return suggestions, which is availability rather than correctness. Scores are not calibrated confidence. Equal source weighting, short context and possible near-duplicate text limit generalization. Prior assisted quiz scores and GPU model results are not the accuracy of this app.

Data: [official SwiftKey corpus](https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip). Method background: [Chen and Goodman (1996)](https://aclanthology.org/P96-1041/). Framework: [Posit Shiny](https://shiny.posit.co/).
