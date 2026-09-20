# Compare the unchanged predictor with complete-word proposals.
# Author: Sonja Sahebzad. Open this file in the Capstone RStudio project.
.complete_source <- tryCatch(sys.frame(1)$ofile, error=function(e) NULL)
.complete_root <- if (!is.null(.complete_source)) dirname(normalizePath(.complete_source)) else getwd()
.complete_previous <- getwd()
setwd(.complete_root)
tryCatch(source("R/predictive_model_v3.R"), finally=setwd(.complete_previous))

try_complete_words <- function(phrase=NULL, choices=character()) {
  if (is.null(phrase)) phrase <- readline("English sentence fragment: ")
  normalized <- normalize_prediction_text(trimws(phrase))
  if (!nzchar(normalized)) stop("Enter an English sentence fragment.", call.=FALSE)
  selection_file <- file.path(.complete_root, "models", "word_generation_selection.json")
  summary_file <- file.path(.complete_root, "models", "word_generation_summary.json")
  if (!file.exists(selection_file) || !file.exists(summary_file)) {
    stop("The controlled experiment is not yet complete.", call.=FALSE)
  }
  selection <- jsonlite::read_json(selection_file)
  summary <- jsonlite::read_json(summary_file)
  model <- readRDS(file.path(.complete_root, "models", "selected_predictor_v3.rds"))
  candidates <- predict_next_v3(model, normalized, top_n=20L)$word
  request <- tempfile(fileext=".json")
  output <- tempfile(fileext=".json")
  on.exit(unlink(c(request, output)), add=TRUE)
  jsonlite::write_json(list(phrase=normalized, choices=as.list(choices),
                           ngram_candidates=as.list(candidates)), request, auto_unbox=TRUE)
  python <- file.path(.complete_root, ".venv-neural", "Scripts", "python.exe")
  script <- file.path(.complete_root, "python", "word_generation_experiment.py")
  cat("Loading the local model and word dictionary. Loading takes a moment.\n")
  status <- system2(python, c(shQuote(script), "--request", shQuote(request), "--output", shQuote(output)))
  if (status != 0 || !file.exists(output)) stop("Prediction did not finish; see the message above.", call.=FALSE)
  result <- jsonlite::read_json(output, simplifyVector=TRUE)
  expanded <- result$candidates[[selection$candidate]]
  cat("\nOriginal predictor:\n", paste(seq_along(result$baseline$words), result$baseline$words, collapse="\n"), "\n")
  cat("\nComplete-word search (", selection$candidate, "):\n", sep="")
  cat(paste(seq_along(expanded$words), expanded$words, collapse="\n"), "\n")
  cat("Added candidates:", if (length(expanded$scores)) nrow(expanded$scores) else 0L, "\n")
  if (!isTRUE(summary$promote)) cat("Research option: the final comparison did not establish an accuracy improvement.\n")
  if (length(result$options$words)) {
    cat("\nSeparate ranking of supplied choices:\n")
    cat(paste(seq_along(result$options$words), result$options$words, collapse="\n"), "\n")
  }
  cat("Ranked suggestions are not guaranteed answers.\n")
  invisible(result)
}

if (interactive()) try_complete_words()

