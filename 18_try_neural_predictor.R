# Local neural predictor, author: Sonja Sahebzad.
# The default follows the documented independent evaluation.
.neural_source <- tryCatch(sys.frame(1)$ofile,error=function(e) NULL)
.neural_root <- if(!is.null(.neural_source)) dirname(normalizePath(.neural_source)) else getwd()
.neural_wd <- getwd()
setwd(.neural_root)
tryCatch(source("R/predictive_model_v3.R"),finally=setwd(.neural_wd))

try_neural_predictor <- function(phrase=NULL,choices=NULL,size=NULL,candidate=NULL) {
  adaptation <- file.path(.neural_root,"models","adaptation_summary.json")
  use_adaptation <- !is.null(candidate) || (is.null(size) && file.exists(adaptation))
  if(use_adaptation) {
    if(is.null(candidate)) candidate <- jsonlite::read_json(adaptation)$default_candidate
    if(!candidate %in% c("base64","base256","local256","augmented256")) stop("Unknown model candidate.")
  } else if(is.null(size)) {
    selected <- file.path(.neural_root,"models","neural_selection.json")
    size <- if(file.exists(selected)) jsonlite::read_json(selected)$size else "1.7B"
  }
  if(is.null(phrase)) phrase <- readline("English sentence fragment: ")
  phrase <- trimws(phrase)
  if(!nzchar(phrase)) stop("Please enter the sentence fragment.",call.=FALSE)
  if(is.null(choices) && interactive()) {
    entry <- readline("Optional answer choices separated by commas (Enter = skip): ")
    choices <- if(nzchar(trimws(entry))) trimws(strsplit(entry,",",fixed=TRUE)[[1L]]) else character()
  }
  model_phrase <- normalize_prediction_text(phrase)
  if(!nzchar(model_phrase)) stop("Please enter a phrase containing English words.",call.=FALSE)
  ngram <- readRDS(file.path(.neural_root,"models","selected_predictor_v3.rds"))
  candidates <- predict_next_v3(ngram,phrase,top_n=20L)$word
  request <- tempfile(fileext=".json"); output <- tempfile(fileext=".json")
  on.exit(unlink(c(request,output)),add=TRUE)
  jsonlite::write_json(list(phrase=model_phrase,choices=as.list(choices),ngram_candidates=as.list(candidates)),
                      request,auto_unbox=TRUE)
  python <- file.path(.neural_root,".venv-neural","Scripts","python.exe")
  if(use_adaptation) {
    script <- file.path(.neural_root,"python","adaptation_experiment.py")
    arguments <- c(shQuote(script),"--candidate",shQuote(candidate))
  } else {
    script <- file.path(.neural_root,"python","neural_predictor.py")
    arguments <- c(shQuote(script),"--size",shQuote(size))
  }
  arguments <- c(arguments,"--request",shQuote(request),"--output",shQuote(output))
  cat("Loading and running the local neural model. This can take a moment.\n")
  status <- system2(python,arguments)
  if(status!=0 || !file.exists(output)) stop("The predictor did not complete; see the error above.",call.=FALSE)
  result <- jsonlite::read_json(output,simplifyVector=TRUE)
  if(use_adaptation) {
    labels <- c(base64="Existing neural model",base256="Wider candidate list",
                local256="Adapted to official texts",augmented256="Adapted with extra dialogue")
    cat("\nModel:",labels[[candidate]],"\n")
  }
  cat("\nNext-word suggestions:\n")
  cat(paste0(seq_along(result$words),". ",result$words,collapse="\n"),"\n")
  if(length(result$choices)) {
    cat("\nSeparate ranking of your supplied choices:\n")
    cat(paste0(seq_along(result$choices),". ",result$choices,collapse="\n"),"\n")
  }
  cat("These are ranked suggestions, not a guarantee of correctness.\n")
  invisible(result)
}
if(interactive()) try_neural_predictor()
