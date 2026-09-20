# Recompute the report from saved, independently checked outcomes.
# Author: Sonja Sahebzad. No model training or final-test rerun is started.
.ranking_source <- tryCatch(sys.frame(1)$ofile,error=function(e) NULL)
.ranking_root <- if (!is.null(.ranking_source)) dirname(normalizePath(.ranking_source)) else getwd()
verify_ranking_report <- function() {
  previous <- getwd();on.exit(setwd(previous),add=TRUE);setwd(.ranking_root)
  if (!file.exists("models/ranking_study_summary.json")) stop("The study is not complete yet. Preserve the running experiment.",call.=FALSE)
  python <- file.path(.ranking_root,".venv-neural","Scripts","python.exe")
  script <- file.path(.ranking_root,"python","audit_ranking_study.py")
  status <- system2(python,c(shQuote(script),"--final"))
  if(status!=0L) stop("Independent verification failed; see the Console output.",call.=FALSE)
  report <- rmarkdown::render("30_ranking_and_calibration.Rmd",quiet=TRUE)
  result <- jsonlite::fromJSON("models/ranking_study_summary.json")
  print(result$final,row.names=FALSE)
  print(result$selective[result$selective$variant==result$selected,],row.names=FALSE)
  cat("\nReport:",normalizePath(report,winslash="/"),"\n")
  cat("Confidence refers to the first suggestion. Top-three accuracy is separate.\n")
  invisible(report)
}
verify_ranking_report()
