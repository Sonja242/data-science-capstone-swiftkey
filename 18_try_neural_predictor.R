# Experimental neural alternative, author: Sonja Sahebzad.
# Uses an externally pretrained Qwen model with an R-generated shortlist.
.neural_source <- tryCatch(sys.frame(1)$ofile, error=function(e) NULL)
.neural_root <- if(!is.null(.neural_source)) dirname(normalizePath(.neural_source)) else getwd()
.neural_wd <- getwd()
setwd(.neural_root)
tryCatch(source("R/predictive_model_v3.R"),finally=setwd(.neural_wd))

try_neural_predictor <- function(phrase=NULL,choices=NULL,size=NULL) {
  if(is.null(size)) {
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
  ngram <- readRDS(file.path(.neural_root,"models","selected_predictor_v3.rds"))
  candidates <- predict_next_v3(ngram,phrase,top_n=20L)$word
  request <- tempfile(fileext=".json")
  output <- tempfile(fileext=".json")
  on.exit(unlink(c(request,output)),add=TRUE)
  jsonlite::write_json(list(phrase=phrase,choices=as.list(choices),ngram_candidates=as.list(candidates)),request,auto_unbox=TRUE)
  python <- file.path(.neural_root,".venv-neural","Scripts","python.exe")
  script <- file.path(.neural_root,"python","neural_predictor.py")
  cat("Loading and running the local neural model. This can take a moment.\n")
  status <- system2(python,c(shQuote(script),"--size",shQuote(size),"--request",shQuote(request),"--output",shQuote(output)))
  if(status!=0 || !file.exists(output)) stop("The neural predictor did not complete; see the error above.",call.=FALSE)
  result <- jsonlite::read_json(output,simplifyVector=TRUE)
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
