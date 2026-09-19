# Fresh local comparison data. Author: Sonja Sahebzad.
library(data.table)
source("R/predictive_model_v3.R")
setDTthreads(4L)
rows <- readRDS("models/expanded_partitions_v3.rds")
rows[, row_id := .I]
rows <- rows[stringi::stri_count_fixed(text," ") >= 2L]
pick <- function(x,n,seed) {
  set.seed(seed)
  x[, .SD[sample.int(.N,min(.N,n))], by=source]
}
# Reconstruct the lines used in v3, then exclude them from the new comparison.
old_validation <- pick(rows[split=="validation"],600L,20261102L)$row_id
old_test <- pick(rows[split=="test"],1000L,20261103L)$row_id
dev_rows <- pick(rows[split=="validation" & !row_id %in% old_validation],100L,20261201L)
test_rows <- pick(rows[split=="test" & !row_id %in% old_test],300L,20261202L)
stopifnot(!any(dev_rows$row_id %in% test_rows$row_id),
          !any(test_rows$row_id %in% old_test))
model <- readRDS("models/selected_predictor_v3.rds")
# The smaller training vocabulary supplies fixed synthetic distractors, as in v3.
raw_counts <- readRDS("models/expanded_counts_small_v3.rds")
frequency <- data.table(word=raw_counts$vocabulary[raw_counts$tables[[1]]$w1],
                        count=raw_counts$tables[[1]]$count)
rm(raw_counts, rows)
gc(FALSE)
dir.create("data/neural_evaluation",recursive=TRUE,showWarnings=FALSE)
for (split_name in c("development","test")) {
  chosen <- if(split_name=="development") dev_rows else test_rows
  seed <- if(split_name=="development") 20261203L else 20261204L
  set.seed(seed)
  words <- strsplit(chosen$text," ",fixed=TRUE)
  positions <- vapply(words,function(w) sample(2:length(w),1L),integer(1))
  cases <- data.table(source=chosen$source,
    line_hash=vapply(chosen$text,digest::digest,character(1),algo="sha256",serialize=FALSE),
    prefix=mapply(function(w,k) paste(head(w,k-1L),collapse=" "),words,positions),
    actual=mapply(function(w,k) w[k],words,positions))
  options <- make_choice_cases(cases,frequency,seed+10L)
  cases[, options_json := vapply(options,jsonlite::toJSON,character(1),auto_unbox=FALSE)]
  started <- proc.time()[[3L]]
  predictions <- lapply(seq_len(nrow(cases)),function(i) {
    dist <- expanded_distribution(model,cases$prefix[i])
    ranked <- rank_expanded(model,dist,top_n=20L)
    opts <- rank_expanded(model,dist,choices=options[[i]])
    list(words=ranked$word,choice=opts$word[1L])
  })
  elapsed <- proc.time()[[3L]]-started
  cases[, ngram_candidates_json := vapply(predictions,function(x) jsonlite::toJSON(x$words),character(1))]
  cases[, ngram_choice := vapply(predictions,function(x) x$choice,character(1))]
  cases[, ngram_rank := mapply(function(x,y) match(y,head(x$words,3L),nomatch=0L),predictions,actual)]
  fwrite(cases,file.path("data/neural_evaluation",paste0(split_name,".csv")))
  message(split_name,": ",nrow(cases)," cases; ngram top-3 ",round(mean(cases$ngram_rank>0)*100,1),"%")
}
fwrite(data.table(word=model$vocabulary[model$vocabulary!="<unk>"]),"data/neural_evaluation/vocabulary.csv")
writeLines(c("No new local training uses these cases.",
  "Rows previously evaluated in v3 are excluded.",
  "Unknown overlap with a downloaded model's pretraining cannot be ruled out.",
  "Four-choice options are synthetic frequency-matched distractors, not actual Coursera questions."),
  "data/neural_evaluation/README.txt")
cat("Fresh development and test cases prepared.\n")
