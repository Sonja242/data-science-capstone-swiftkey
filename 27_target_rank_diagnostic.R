# Development-only diagnosis with known targets. Author: Sonja Sahebzad.
# No production settings are changed. Existing evidence is verified and reused.
.target_source <- tryCatch(sys.frame(1)$ofile,error=function(e) NULL)
.target_root <- if (!is.null(.target_source)) dirname(normalizePath(.target_source)) else getwd()

run_target_diagnosis <- function() {
  previous <- getwd()
  on.exit(setwd(previous),add=TRUE)
  setwd(.target_root)
  python <- file.path(.target_root,".venv-neural","Scripts","python.exe")
  script <- file.path(.target_root,"python","target_rank_diagnostic.py")
  if (!file.exists("models/target_rank_summary.json")) {
    if (file.exists("models/target_rank_protocol.json")) {
      stop("An incomplete registered run exists. Preserve its files and inspect it before restarting.",call.=FALSE)
    }
    cat("Running a development-only diagnosis. Known targets are used; these are not performance scores.\n")
    status <- system2(python,c(shQuote(script),"--run"))
    if (status!=0L) stop("Diagnosis did not finish; inspect the Console output.",call.=FALSE)
  }
  status <- system2(python,c(shQuote(script),"--verify"))
  if (status!=0L) stop("Saved diagnostic evidence failed verification.",call.=FALSE)
  audit_script <- file.path(.target_root,"python","audit_target_rank.py")
  status <- system2(python,shQuote(audit_script))
  if (status!=0L) stop("Independent evidence audit failed.",call.=FALSE)
  html <- rmarkdown::render("28_target_rank_diagnostic.Rmd",quiet=TRUE)
  summary <- jsonlite::fromJSON("models/target_rank_summary.json")
  print(summary$groups[summary$groups$group=="All",],row.names=FALSE)
  cat("\nDiagnostic report:",normalizePath(html,winslash="/"),"\n")
  cat("These target-aware counts are not production prediction accuracy.\n")
  invisible(html)
}

run_target_diagnosis()
