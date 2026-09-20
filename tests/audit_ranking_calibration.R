# Independent R optimizer check of the frozen confidence map.
# Author: Sonja Sahebzad. Reads calibration data only, not final outcomes.
data <- jsonlite::fromJSON("models/ranking_study_calibration.json",simplifyVector=FALSE)
fit <- jsonlite::fromJSON("models/ranking_study_calibrators.json",simplifyVector=FALSE)
stopifnot(fit$calibration_sha256==digest::digest(file="models/ranking_study_calibration.json",algo="sha256"))
checks <- lapply(names(fit$models),function(variant) {
  raw <- vapply(data$details,function(d) d$predictions[[variant]]$raw_shortlist_share,numeric(1))
  y <- vapply(data$details,function(d) as.numeric(d$predictions[[variant]]$words[[1]]==d$actual),numeric(1))
  x <- qlogis(pmin(1-1e-6,pmax(1e-6,raw)))
  objective <- function(beta) {
    z <- beta[1]+beta[2]*x
    sum(pmax(z,0)+log1p(exp(-abs(z)))-y*z)+.5*beta[2]^2
  }
  gradient <- function(beta) {
    error <- plogis(beta[1]+beta[2]*x)-y
    c(sum(error),sum(error*x)+beta[2])
  }
  reference <- optim(c(0,0),objective,gradient,method="BFGS",control=list(reltol=1e-13,maxit=1000))
  stored <- c(fit$models[[variant]]$intercept,fit$models[[variant]]$slope)
  stopifnot(reference$convergence==0L,max(abs(reference$par-stored))<1e-5,
            max(abs(gradient(stored)))<1e-6)
  list(variant=variant,cases=length(y),positives=sum(y),stored_coefficients=stored,
       independent_R_coefficients=reference$par,
       maximum_coefficient_difference=max(abs(reference$par-stored)),
       stored_gradient_maximum=max(abs(gradient(stored))),
       R_optimizer="optim BFGS, start(0,0); independent of Python Newton implementation")
})
result <- list(author="Sonja Sahebzad",passed=TRUE,checks=checks,
  calibration_artifact_sha256=digest::digest(file="models/ranking_study_calibration.json",algo="sha256"),
  calibrators_sha256=digest::digest(file="models/ranking_study_calibrators.json",algo="sha256"),
  final_outcomes_read=FALSE)
destination <- "models/ranking_study_calibration_fit_audit.json"
if (!file.exists(destination)) jsonlite::write_json(result,destination,auto_unbox=TRUE,pretty=TRUE)
print(result)
