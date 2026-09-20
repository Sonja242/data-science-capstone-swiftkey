# Reserve independent calibration and final examples before ranking comparison.
# Author: Sonja Sahebzad.
library(data.table)
source("R/predictive_model_v3.R")
setDTthreads(4L)
dest <- "data/ranking_study"
if (dir.exists(dest) || file.exists("models/ranking_study_data_manifest.json"))
  stop("Preserve the existing study preparation.")
rows <- readRDS("models/expanded_partitions_v3.rds")
stopifnot(uniqueN(rows$text)==nrow(rows))
rows[,row_id:=.I]
eligible <- rows[stringi::stri_count_fixed(text," ")>=2L]
pick <- function(x,n,seed) {
  set.seed(seed)
  x[,.SD[sample.int(.N,min(.N,n))],by=source]
}
v3_val <- pick(eligible[split=="validation"],600L,20261102L)
neural_val <- pick(eligible[split=="validation" & !row_id %in% v3_val$row_id],100L,20261201L)
adapt_val <- pick(eligible[split=="validation" & !row_id %in% c(v3_val$row_id,neural_val$row_id)],200L,20261301L)
v3_test <- pick(eligible[split=="test"],1000L,20261103L)
neural_test <- pick(eligible[split=="test" & !row_id %in% v3_test$row_id],300L,20261202L)
adapt_test <- pick(eligible[split=="test" & !row_id %in% c(v3_test$row_id,neural_test$row_id)],300L,20261302L)
curve_test <- pick(eligible[split=="test" & !row_id %in% c(v3_test$row_id,neural_test$row_id,adapt_test$row_id)],300L,20261402L)
word_test <- pick(eligible[split=="test" & !row_id %in% c(v3_test$row_id,neural_test$row_id,adapt_test$row_id,curve_test$row_id)],300L,20261502L)
prior <- rbindlist(list(v3_val,neural_val,adapt_val,v3_test,neural_test,adapt_test,curve_test,word_test))
hashes <- function(x) vapply(x,digest::digest,character(1),algo="sha256",serialize=FALSE,USE.NAMES=FALSE)
stopifnot(identical(hashes(adapt_val$text),fread("data/adaptation/development.csv")$line_hash),
          identical(hashes(word_test$text),fread("models/word_generation_reserved_case_hashes.csv")$line_hash))
external <- readLines("data/adaptation/taskmaster_training.txt",encoding="UTF-8",warn=FALSE)
calibration <- pick(eligible[split=="validation" & !row_id %in% prior$row_id & !text %chin% external],300L,20261601L)
final <- pick(eligible[split=="test" & !row_id %in% prior$row_id & !text %chin% external],300L,20261602L)
stopifnot(nrow(calibration)==900L,nrow(final)==900L,
          !any(calibration$row_id %in% final$row_id),
          !any(c(calibration$text,final$text) %chin% rows[split=="train",text]))
dir.create(dest,recursive=TRUE)
reserved <- rbindlist(list(
  data.table(role="calibration",source=calibration$source,line_hash=hashes(calibration$text)),
  data.table(role="final",source=final$source,line_hash=hashes(final$text))))
stopifnot(uniqueN(reserved$line_hash)==1800L,!any(reserved$line_hash %chin% hashes(prior$text)))
fwrite(reserved,"models/ranking_study_reserved_hashes.csv")
manifest <- list(author="Sonja Sahebzad",recorded_utc=format(Sys.time(),tz="UTC",format="%Y-%m-%dT%H:%M:%SZ"),
  calibration_cases=900L,final_cases=900L,cases_per_source=300L,
  selection_seeds=c(calibration=20261601L,final=20261602L),
  position_seeds=c(calibration=20261603L,final=20261604L),
  original_partition_lines=nrow(rows),global_normalized_line_uniqueness=TRUE,
  prior_evaluation_cases=nrow(prior),prior_evaluation_overlap=0L,calibration_final_overlap=0L,
  official_training_overlap=0L,external_training_overlap=0L,
  calibration_partition="Previously unused validation lines",
  final_partition="Previously unused test lines",
  limitation="Exact normalized-line exclusion; near duplicates and unknown pretrained-model overlap may remain.",
  partition_rds_sha256=digest::digest(file="models/expanded_partitions_v3.rds",algo="sha256"),
  source_code_sha256=digest::digest(file="29_prepare_ranking_study.R",algo="sha256"))
rm(rows,eligible,prior,external,v3_val,neural_val,adapt_val,v3_test,neural_test,adapt_test,curve_test,word_test);gc(FALSE)
model <- readRDS("models/selected_predictor_v3.rds")
make_cases <- function(chosen,seed,path) {
  set.seed(seed)
  words <- strsplit(chosen$text," ",fixed=TRUE)
  positions <- vapply(words,function(w) sample(2:length(w),1L),integer(1))
  cases <- data.table(source=chosen$source,line_hash=hashes(chosen$text),
    prefix=mapply(function(w,k) paste(head(w,k-1L),collapse=" "),words,positions),
    actual=mapply(function(w,k) w[k],words,positions))
  proposals <- lapply(cases$prefix,function(p) predict_next_v3(model,p,top_n=20L)$word)
  cases[,ngram_candidates_json:=vapply(proposals,jsonlite::toJSON,character(1))]
  fwrite(cases,path)
  digest::digest(file=path,algo="sha256")
}
manifest$calibration_csv_sha256 <- make_cases(calibration,20261603L,file.path(dest,"calibration.csv"))
manifest$final_csv_sha256 <- make_cases(final,20261604L,file.path(dest,"final.csv"))
manifest$development_csv_sha256 <- digest::digest(file="data/adaptation/development.csv",algo="sha256")
manifest$reserved_hashes_sha256 <- digest::digest(file="models/ranking_study_reserved_hashes.csv",algo="sha256")
jsonlite::write_json(manifest,"models/ranking_study_data_manifest.json",auto_unbox=TRUE,pretty=TRUE)
cat("Reserved 900 calibration and 900 final cases. No neural outcomes inspected.\n")
