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

corpus_paths <- c(
  twitter = file.path("data", "final", "en_US", "en_US.twitter.txt"),
  blogs = file.path("data", "final", "en_US", "en_US.blogs.txt"),
  news = file.path("data", "final", "en_US", "en_US.news.txt")
)

cat("\nSearching the official corpus for direct evidence...\n")
result <- predict_exact_continuation(corpus_paths, phrase, top_n = 3L)
if (!nrow(result)) {
  cat("No direct continuation found; using the back-off model.\n")
  result <- predict_next(model, phrase, top_n = 3L)
} else {
  cat("Direct corpus continuation found in ", result$source_file[[1L]], ".\n", sep = "")
}

cat("\nPredicted next words:\n")
cat(format_predictions(result), sep = "\n")
cat("\n")
