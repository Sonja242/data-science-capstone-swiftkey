# Hash-only validation for the new word-generation holdout. Sonja Sahebzad.
# Reads existing partition text only to hash it. Never reads new final phrases,
# prediction targets, or model outcomes from data/word_generation/final_test.csv.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
project <- normalizePath(args[1L],winslash="/",mustWork=TRUE)
output <- args[2L]
setwd(project)
library(data.table)
setDTthreads(4L)
rows <- readRDS("models/expanded_partitions_v3.rds")
rows[, row_id:=.I]
stopifnot(uniqueN(rows$text)==nrow(rows))
eligible <- rows[stringi::stri_count_fixed(text," ")>=2L]
pick <- function(x,n,seed) {
  set.seed(seed)
  x[, .SD[sample.int(.N,min(.N,n))],by=source]
}
hashes <- function(text) vapply(text,digest::digest,character(1),algo="sha256",serialize=FALSE,USE.NAMES=FALSE)
old_val <- pick(eligible[split=="validation"],600L,20261102L)
old_test <- pick(eligible[split=="test"],1000L,20261103L)
neural_val <- pick(eligible[split=="validation" & !row_id %in% old_val$row_id],100L,20261201L)
neural_test <- pick(eligible[split=="test" & !row_id %in% old_test$row_id],300L,20261202L)
adapt_val <- pick(eligible[split=="validation" & !row_id %in% c(old_val$row_id,neural_val$row_id)],200L,20261301L)
adapt_test <- pick(eligible[split=="test" & !row_id %in% c(old_test$row_id,neural_test$row_id)],300L,20261302L)
curve_test <- pick(eligible[split=="test" & !row_id %in% c(old_test$row_id,neural_test$row_id,adapt_test$row_id)],300L,20261402L)
prior <- list(v3=old_test,neural=neural_test,adaptation=adapt_test,learning_curve=curve_test)
expected <- c(v3=3000L,neural=900L,adaptation=900L,learning_curve=900L)
stopifnot(identical(vapply(prior,nrow,integer(1)),expected))
old_curve <- fread("models/learning_curve_reserved_case_hashes.csv")
stopifnot(identical(old_curve$source,curve_test$source),identical(old_curve$line_hash,hashes(curve_test$text)))
all_prior <- rbindlist(c(prior,list(old_val,neural_val,adapt_val)))
stopifnot(uniqueN(all_prior$row_id)==nrow(all_prior))
reserved <- fread("models/word_generation_reserved_case_hashes.csv")
stopifnot(nrow(reserved)==900L,uniqueN(reserved$line_hash)==900L,
          setequal(reserved$source,c("blogs","news","twitter")),
          all(reserved[,.N,by=source]$N==300L),
          !any(reserved$line_hash %chin% hashes(all_prior$text)))
# Hash existing test-partition candidates, retaining only source/split metadata
# for matching reserved hashes. No new target position or outcome is inspected.
test_rows <- rows[split=="test"]
matches <- list()
for(start in seq.int(1L,nrow(test_rows),10000L)) {
  end <- min(start+9999L,nrow(test_rows))
  h <- hashes(test_rows$text[start:end])
  at <- which(h %chin% reserved$line_hash)
  if(length(at)) matches[[length(matches)+1L]] <- data.table(
    line_hash=h[at],source=test_rows$source[start+at-1L])
}
matched <- rbindlist(matches)
stopifnot(nrow(matched)==900L,uniqueN(matched$line_hash)==900L,
          all(reserved$source==matched$source[match(reserved$line_hash,matched$line_hash)]))
# Global normalized-line uniqueness and membership in the test partition imply
# no complete-line overlap with the entire official training partition.
local <- readLines("data/adaptation/local_training.txt",encoding="UTF-8",warn=FALSE)
external <- readLines("data/adaptation/taskmaster_training.txt",encoding="UTF-8",warn=FALSE)
stopifnot(all(local %chin% rows[split=="train",text]),
          !any(reserved$line_hash %chin% hashes(c(local,external))))
result <- list(author="Sonja Sahebzad",passed=TRUE,
  partition_normalized_lines=nrow(rows),global_normalized_line_uniqueness=TRUE,
  prior_final_counts=as.list(expected),prior_final_unique_lines=sum(expected),
  all_prior_evaluation_unique_lines=nrow(all_prior),reserved_cases=nrow(reserved),
  reserved_per_source=300L,reserved_hashes_confirmed_in_official_test_partition=900L,
  reserved_source_labels_verified=TRUE,prior_evaluation_overlap=0L,
  official_training_partition_overlap=0L,local_and_external_adaptation_training_overlap=0L,
  reserved_hash_manifest_sha256=digest::digest(file="models/word_generation_reserved_case_hashes.csv",algo="sha256"),
  partition_rds_sha256=digest::digest(file="models/expanded_partitions_v3.rds",algo="sha256"),
  new_final_case_file_or_model_outcomes_read=FALSE,
  method="Reconstruct prior sampled rows; compare reserved hashes; hash old test-partition pool to verify membership/source. Global normalized-text uniqueness establishes official training separation.",
  limitations="Exact normalized-line checks do not exclude near duplicates, related documents, or unknown pretrained-model overlap.")
jsonlite::write_json(result,output,auto_unbox=TRUE,pretty=TRUE)
print(result)