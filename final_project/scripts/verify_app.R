# Run from final_project. Does not tune the frozen model or rerun selection.
library(shiny)
library(data.table)
source("app/predictor.R")
cases <- fread("../data/final_product/final_test.csv")
start <- as.numeric(Sys.time())
model <- readRDS("app/model.rds")
load_seconds <- as.numeric(Sys.time())-start
# proc.time() was too coarse on Windows for single predictions. Use wall time
# with microsecond resolution; preserve the original measurements privately.
metadata <- jsonlite::fromJSON("results/metrics.json",simplifyVector=FALSE)
old_path <- "../data/final_product/initial_metrics_coarse_timer.json"
if (!file.exists(old_path)) file.copy("results/metrics.json",old_path)
invisible(predict_word(model,"a simple example"))
timing <- vapply(cases$prefix,function(phrase){
  a <- as.numeric(Sys.time());invisible(predict_word(model,phrase));
  (as.numeric(Sys.time())-a)*1000
},numeric(1))
fwrite(data.table(source=cases$source,line_hash=cases$line_hash,elapsed_ms=timing),"results/timing.csv")
metadata$overall$median_ms <- median(timing)
metadata$overall$p95_ms <- unname(quantile(timing,.95))
metadata$overall$mean_ms <- mean(timing)
metadata$load_seconds <- load_seconds
metadata$timing_method <- "Separate frozen-model timing pass using Sys.time wall clock; proc.time initial pass was too coarse on Windows. No accuracy values or model choices changed. Excludes model load, UI rendering and network."
jsonlite::write_json(metadata,"results/metrics.json",pretty=TRUE,auto_unbox=TRUE,digits=10)
file.copy("results/metrics.json","app/metrics.json",overwrite=TRUE)

edge <- c("", "12345!!!", "https://example.com", "xqzzzz", "HELLO WORLD", "I'm looking forward to", "<script>alert('x')</script>")
for(phrase in edge){p<-predict_word(model,phrase);stopifnot(length(p$words)==3L,all(grepl("^[a-z]+('[a-z]+)*$",p$words)),!any(p$words %in% model$blocked_terms))}
for(phrase in head(cases$prefix,50L)){
  a<-predict_word(model,phrase); b<-predict_word(model,phrase,dense=TRUE)
  stopifnot(identical(a$words,b$words),max(abs(a$scores-b$scores))<1e-12,abs(sum(b$distribution)-1)<1e-10)
}
rm(model);gc(FALSE)
setwd("app")
source("app.R",local=TRUE)
shiny::testServer(server,{
  session$setInputs(phrase="The weather today is");session$setInputs(predict=1)
  stopifnot(length(result()$words)==3L, !is.null(output$prediction))
  session$setInputs(phrase="A new phrase");stopifnot(is.null(result()))
  session$setInputs(phrase="");session$setInputs(predict=2)
  stopifnot(is.null(result()),!is.null(error()))
  session$setInputs(phrase=paste(rep("a",501L),collapse=""));session$setInputs(predict=3)
  stopifnot(is.null(result()),!is.null(error()))
  session$setInputs(phrase="zxqvv");session$setInputs(predict=4)
  stopifnot(length(result()$words)==3L)
  session$setInputs(clear=1);stopifnot(is.null(result()))
})
setwd("..")
writeLines(c("PASS: sparse and dense rankings agree on 50 fresh cases.",
 "PASS: 7 tokenizer/fallback edge cases return three safe-format words.",
 "PASS: Shiny server prediction, edit invalidation, empty input, input limit, fallback and clear.",
 sprintf("Timing on 900 cases: median %.3f ms, p95 %.3f ms; load %.3f s.",median(timing),quantile(timing,.95),load_seconds)),"results/verification.txt")
cat(readLines("results/verification.txt"),sep="\n")
