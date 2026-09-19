# Interactive next-word predictor
# Author: Sonja Sahebzad

source(file.path("R", "predictive_model.R"))

model_path <- file.path("models", "selected_backoff_model.rds")
if (!file.exists(model_path)) {
  stop(
    "Model not found. Knit 12_predictive_model_evaluation.Rmd first.",
    call. = FALSE
  )
}

model <- readRDS(model_path)
phrase <- readline("Type an English phrase and press Enter: ")

if (!nzchar(trimws(phrase))) {
  stop("Please enter at least one word.", call. = FALSE)
}

result <- predict_next(model, phrase, top_n = 3L)
cat("\nPredicted next words:\n")
cat(format_predictions(result), sep = "\n")
cat("\n")
