# A fresh final holdout for the complete-word generation experiment.
# Author: Sonja Sahebzad.
library(data.table)
source("R/predictive_model_v3.R")
setDTthreads(4L)
destination <- "data/word_generation/final_test.csv"
if (file.exists(destination)) stop("Preserve the reserved final holdout; it already exists.")
dir.create(dirname(destination), recursive=TRUE, showWarnings=FALSE)
rows <- readRDS("models/expanded_partitions_v3.rds")
rows[, row_id := .I]
eligible <- rows[stringi::stri_count_fixed(text," ") >= 2L]
pick <- function(x,n,seed) {
  set.seed(seed)
  x[, .SD[sample.int(.N,min(.N,n))], by=source]
}
old_test <- pick(eligible[split=="test"],1000L,20261103L)$row_id
neural_test <- pick(eligible[split=="test" & !row_id %in% old_test],300L,20261202L)$row_id
adaptation_test <- pick(eligible[split=="test" & !row_id %in% c(old_test,neural_test)],300L,20261302L)$row_id
curve_test <- pick(eligible[split=="test" & !row_id %in% c(old_test,neural_test,adaptation_test)], 300L,20261402L)$row_id
chosen <- pick(eligible[split=="test" & !row_id %in% c(old_test,neural_test,adaptation_test,curve_test)],
               300L,20261502L)
stopifnot(nrow(chosen)==900L, all(chosen$split=="test"),
          !any(chosen$row_id %in% c(old_test,neural_test,adaptation_test,curve_test)))
case_hashes <- vapply(chosen$text,digest::digest,character(1),algo="sha256",serialize=FALSE)
stopifnot(!anyDuplicated(case_hashes))
prior_files <- c("data/neural_evaluation/development.csv","data/neural_evaluation/test.csv",
                 "data/adaptation/development.csv","data/adaptation/test.csv", "data/learning_curve/final_test.csv")
prior_hashes <- unique(unlist(lapply(prior_files,function(f) fread(f,select="line_hash")$line_hash)))
stopifnot(!any(case_hashes %chin% prior_hashes))
# Complete normalized lines were deduplicated before the existing corpus split.
stopifnot(!any(chosen$text %chin% rows[split=="train",text]))
if (file.exists("data/adaptation/local_training.txt")) {
  training <- readLines("data/adaptation/local_training.txt",encoding="UTF-8",warn=FALSE)
  stopifnot(!any(chosen$text %chin% training))
  rm(training)
}
jsonlite::write_json(list(
  recorded_utc=format(Sys.time(),tz="UTC",format="%Y-%m-%dT%H:%M:%SZ"),
  cases=900L,cases_per_source=300L,selection_seed=20261502L,
  position_seed=20261503L,choice_seed=20261504L,
  excluded_prior_v3_test_lines=length(old_test),
  excluded_prior_neural_test_lines=length(neural_test),
  excluded_prior_adaptation_test_lines=length(adaptation_test), excluded_prior_learning_curve_test_lines=length(curve_test),
  normalized_training_line_overlap=0L,previous_case_overlap=0L,
  development="Unchanged data/adaptation/development.csv; reused for model selection",
  sources=as.data.frame(chosen[,.N,by=source])),
  "models/word_generation_holdout_manifest.json",auto_unbox=TRUE,pretty=TRUE)
fwrite(data.table(source=chosen$source,line_hash=case_hashes),
       "models/word_generation_reserved_case_hashes.csv")
rm(rows,eligible,old_test,neural_test,adaptation_test); gc(FALSE)
model <- readRDS("models/selected_predictor_v3.rds")
counts <- readRDS("models/expanded_counts_small_v3.rds")
frequency <- data.table(word=counts$vocabulary[counts$tables[[1L]]$w1],
                        count=counts$tables[[1L]]$count)
rm(counts); gc(FALSE)
set.seed(20261503L)
words <- strsplit(chosen$text," ",fixed=TRUE)
positions <- vapply(words,function(w) sample(2:length(w),1L),integer(1))
cases <- data.table(source=chosen$source,line_hash=case_hashes,
  prefix=mapply(function(w,k) paste(head(w,k-1L),collapse=" "),words,positions),
  actual=mapply(function(w,k) w[k],words,positions))
options <- make_choice_cases(cases,frequency,20261504L)
cases[,options_json:=vapply(options,jsonlite::toJSON,character(1),auto_unbox=FALSE)]
predictions <- lapply(seq_len(nrow(cases)),function(i) {
  distribution <- expanded_distribution(model,cases$prefix[i])
  list(words=rank_expanded(model,distribution,top_n=20L)$word,
       choice=rank_expanded(model,distribution,choices=options[[i]])$word[1L])
})
cases[,ngram_candidates_json:=vapply(predictions,function(z) jsonlite::toJSON(z$words),character(1))]
cases[,ngram_choice:=vapply(predictions,function(z) z$choice,character(1))]
cases[,ngram_rank:=mapply(function(z,a) match(a,head(z$words,3L),nomatch=0L),predictions,actual)]
fwrite(cases,destination)
manifest <- jsonlite::read_json("models/word_generation_holdout_manifest.json",simplifyVector=TRUE)
manifest$final_csv_sha256 <- digest::digest(file=destination,algo="sha256")
manifest$development_csv_sha256 <- digest::digest(file="data/adaptation/development.csv",algo="sha256")
jsonlite::write_json(manifest,"models/word_generation_holdout_manifest.json",auto_unbox=TRUE,pretty=TRUE)
cat("Reserved 900 fresh final cases; no neural final-test outcome has been evaluated; R candidates and synthetic options are prepared.\n")

