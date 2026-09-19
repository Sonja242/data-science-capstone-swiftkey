# Interactive next-word prediction. Author: Sonja Sahebzad.
.predictor_script <- tryCatch(sys.frame(1)$ofile, error=function(e) NULL)
.predictor_root <- if (!is.null(.predictor_script)) dirname(normalizePath(.predictor_script)) else getwd()
.predictor_previous_wd <- getwd()
setwd(.predictor_root)
tryCatch(source("R/predictive_model_v3.R"), finally=setwd(.predictor_previous_wd))
.predictor_loaded <- NULL

run_predictor <- function(phrase=NULL, choices=NULL) {
  if (is.null(.predictor_loaded)) {
    cat("Loading the expanded model...\n")
    .predictor_loaded <<- readRDS(file.path(.predictor_root,"models","selected_predictor_v3.rds"))
  }
  model <- .predictor_loaded
  if (is.null(phrase)) phrase <- readline("Type the sentence fragment, then press Enter: ")
  if (!nzchar(trimws(phrase))) stop("Please enter a sentence fragment.",call.=FALSE)
  if (grepl("^question[[:space:]]*[0-9]+$", trimws(phrase), ignore.case=TRUE))
    stop("Enter the sentence fragment itself, not the question number.",call.=FALSE)
  result <- predict_next_v3(model, phrase)
  evidence <- attr(result,"evidence")
  cat("\nNext-word suggestions:\n")
  for (i in seq_len(nrow(result))) cat(sprintf("%d. %s (model probability %.2f%%)\n",
    result$rank[i],result$word[i],100*result$model_probability[i]))
  cat("\nLongest supported context: ",if(evidence$order>1L) evidence$context else "overall word frequencies","\n",sep="")
  cat("These are model estimates; other continuations can also be correct.\n")
  if (result$model_probability[1L]<.15) cat("No single suggestion has strong model support.\n")
  if (is.null(choices) && interactive()) {
    option_text <- readline("Optional choices, separated by commas (Enter = skip): ")
    if (nzchar(trimws(option_text))) choices <- trimws(strsplit(option_text,",",fixed=TRUE)[[1L]])
  }
  if (length(choices)) {
    ranked <- predict_next_v3(model,phrase,choices=choices)
    cat("\nSeparate ranking of your supplied choices:\n")
    for (i in seq_len(nrow(ranked))) cat(sprintf("%d. %s: %s\n",ranked$rank[i],ranked$word[i],
      if(is.na(ranked$model_probability[i])) "unseen word; no individual estimate" else
        sprintf("model probability %.4f%%",100*ranked$model_probability[i])))
    cat("The ranking does not guarantee the correct choice.\n")
    attr(result,"choices") <- ranked
  }
  invisible(result)
}

if (interactive()) run_predictor()
