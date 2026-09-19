# Interactive next-word predictor. Author: Sonja Sahebzad.
# Kept at the familiar filename 13_try_predictor.R in the project.
.predictor_script <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
.predictor_root <- if (!is.null(.predictor_script)) dirname(normalizePath(.predictor_script)) else getwd()
source(file.path(.predictor_root, "R", "predictive_model_v2.R"))

run_predictor <- function(phrase = NULL, choices = NULL) {
  model_file <- file.path(.predictor_root, "models", "selected_predictor_v2.rds")
  if (!file.exists(model_file)) stop("Run 14_rebuild_prediction_v2.R first.", call. = FALSE)
  model <- readRDS(model_file)
  if (is.null(phrase)) phrase <- readline("Type the sentence fragment, then press Enter: ")
  if (!nzchar(trimws(phrase))) stop("Please enter a sentence fragment.", call. = FALSE)
  if (grepl("^question[[:space:]]*[0-9]+$", trimws(phrase), ignore.case = TRUE)) {
    stop("Enter the sentence fragment itself, not the question number.", call. = FALSE)
  }
  result <- predict_next_v2(model, phrase)
  evidence <- attr(result, "evidence")
  cat("\nNext-word suggestions from the trained model:\n")
  for (i in seq_len(nrow(result))) cat(sprintf("%d. %s (model probability: %.2f%%)\n",
    result$rank[i], result$word[i], 100 * result$model_probability[i]))
  cat("\nLongest observed context: ", if (evidence$order == 1L) "none; overall word frequencies" else
    paste0('"', evidence$context, '"'), "\n", sep = "")
  if (evidence$order > 1L) cat("Training occurrences of this context: ", evidence$context_count, "\n", sep = "")
  cat("Longer and shorter contexts are combined. Percentages are model estimates, not certainty.\n")
  if (evidence$context_count < 5L || result$model_probability[1L] < .15) {
    cat("Limited evidence: the leading suggestion may be wrong.\n")
  }
  if (is.null(choices) && interactive()) {
    option_text <- readline("Optional answer choices, separated by commas (Enter = skip): ")
    if (nzchar(trimws(option_text))) choices <- trimws(strsplit(option_text, ",", fixed = TRUE)[[1L]])
  }
  if (length(choices)) {
    ranked <- predict_next_v2(model, phrase, choices = choices)
    cat("\nRanking ONLY the options you supplied (not an answer key):\n")
    for (i in seq_len(nrow(ranked))) {
      cat(sprintf("%d. %s: %s\n", ranked$rank[i], ranked$word[i],
        if (is.na(ranked$model_probability[i])) "unseen word; no individual estimate" else
          sprintf("model probability %.4f%%", 100 * ranked$model_probability[i])))
    }
    cat("The best of these options can still be incorrect. This is not measured multiple-choice accuracy.\n")
  }
  invisible(result)
}

if (interactive()) run_predictor()
