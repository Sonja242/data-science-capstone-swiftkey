# Controlled data-adaptation experiment. Author: Sonja Sahebzad.
library(data.table)
source("R/predictive_model_v3.R")
setDTthreads(4L)
dir.create("data/adaptation",recursive=TRUE,showWarnings=FALSE)
rows <- readRDS("models/expanded_partitions_v3.rds")
rows[, row_id := .I]
pick <- function(x,n,seed) {
  set.seed(seed)
  x[, .SD[sample.int(.N,min(.N,n))],by=source]
}
eligible <- rows[stringi::stri_count_fixed(text," ")>=2L]
old_val <- pick(eligible[split=="validation"],600L,20261102L)$row_id
old_test <- pick(eligible[split=="test"],1000L,20261103L)$row_id
neural_val <- pick(eligible[split=="validation" & !row_id %in% old_val],100L,20261201L)$row_id
neural_test <- pick(eligible[split=="test" & !row_id %in% old_test],300L,20261202L)$row_id
development <- pick(eligible[split=="validation" & !row_id %in% c(old_val,neural_val)],200L,20261301L)
final_test <- pick(eligible[split=="test" & !row_id %in% c(old_test,neural_test)],300L,20261302L)
stopifnot(!any(development$row_id %in% c(old_val,neural_val)),
          !any(final_test$row_id %in% c(old_test,neural_test)),
          !any(final_test$row_id %in% development$row_id))
local <- rbindlist(list(
  pick(eligible[split=="train" & source=="twitter"],24000L,20261303L),
  pick(eligible[split=="train" & source!="twitter"],12000L,20261304L)))
stopifnot(all(local$split=="train"),!any(local$row_id %in% c(development$row_id,final_test$row_id)))
writeLines(local$text,"data/adaptation/local_training.txt",useBytes=TRUE)

dialogs <- jsonlite::fromJSON("data/external/taskmaster/self-dialogs.json",simplifyVector=FALSE)
train_ids <- fread("data/external/taskmaster/train.csv",header=FALSE,select=1L)$V1
dialogs <- Filter(function(x) isTRUE(x$conversation_id %in% train_ids),dialogs)
stopifnot(length(dialogs)>0L)
raw <- unlist(lapply(dialogs,function(d) vapply(d$utterances,`[[`,character(1),"text")),use.names=FALSE)
external <- unique(normalize_prediction_text(raw))
external <- external[stringi::stri_count_fixed(external," ")>=2L]
before <- length(external)
# Remove complete normalized utterances that match any official local holdout.
external <- external[!external %chin% rows[split!="train",text]]
writeLines(external,"data/adaptation/taskmaster_training.txt",useBytes=TRUE)
metadata <- list(local_lines=nrow(local),local_sources=as.data.frame(local[,.N,by=source]),
  external_dialogs=length(dialogs),external_raw_utterances=length(raw),
  external_normalized_unique_lines=length(external),excluded_external_holdout_matches=before-length(external),
  development_cases=nrow(development),test_cases=nrow(final_test),seeds=20261301:20261308,
  limitations="Complete-line duplicates excluded; near duplicates and pretrained-model overlap can remain.")
jsonlite::write_json(metadata,"models/adaptation_data_manifest.json",pretty=TRUE,auto_unbox=TRUE)
rm(rows,eligible,local,dialogs,raw,external); gc(FALSE)

model <- readRDS("models/selected_predictor_v3.rds")
counts <- readRDS("models/expanded_counts_small_v3.rds")
frequency <- data.table(word=counts$vocabulary[counts$tables[[1]]$w1],count=counts$tables[[1]]$count)
jsonlite::write_json(list(word=frequency$word[which.max(frequency$count)],
  rule="Most frequent word in the smaller official training partition"),
  "models/adaptation_constant_baseline.json",auto_unbox=TRUE,pretty=TRUE)
file.copy("data/external/taskmaster/manifest.json","models/adaptation_external_source.json",overwrite=TRUE)
rm(counts); gc(FALSE)
for(split_name in c("development","test")) {
  x <- if(split_name=="development") development else final_test
  seed <- if(split_name=="development") 20261305L else 20261306L
  set.seed(seed)
  words <- strsplit(x$text," ",fixed=TRUE)
  positions <- vapply(words,function(w) sample(2:length(w),1L),integer(1))
  cases <- data.table(source=x$source,
    line_hash=vapply(x$text,digest::digest,character(1),algo="sha256",serialize=FALSE),
    prefix=mapply(function(w,k) paste(head(w,k-1L),collapse=" "),words,positions),
    actual=mapply(function(w,k) w[k],words,positions))
  options <- make_choice_cases(cases,frequency,seed+2L)
  cases[,options_json:=vapply(options,jsonlite::toJSON,character(1),auto_unbox=FALSE)]
  predictions <- lapply(seq_len(nrow(cases)),function(i) {
    d <- expanded_distribution(model,cases$prefix[i])
    list(words=rank_expanded(model,d,top_n=20L)$word,
         choice=rank_expanded(model,d,choices=options[[i]])$word[1L])
  })
  cases[,ngram_candidates_json:=vapply(predictions,function(z) jsonlite::toJSON(z$words),character(1))]
  cases[,ngram_choice:=vapply(predictions,function(z) z$choice,character(1))]
  cases[,ngram_rank:=mapply(function(z,a) match(a,head(z$words,3L),nomatch=0L),predictions,actual)]
  fwrite(cases,file.path("data/adaptation",paste0(split_name,".csv")))
  message(split_name,": ",nrow(cases)," fresh examples exported.")
}
cat("Preparation complete. The final test remains unused for model selection.\n")
