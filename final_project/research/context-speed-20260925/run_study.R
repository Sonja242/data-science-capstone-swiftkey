# Sonja Projects | Author: Sonja Sahebzad
# One controlled study. Existing results are never overwritten.
library(data.table)
library(jsonlite)
args <- commandArgs(trailingOnly=TRUE)
project <- normalizePath(args[1],winslash="/")
out <- normalizePath(args[2],winslash="/")
if(file.exists(file.path(out,"protocol.json"))) stop("This experiment already exists. Preserve its results.")
protocol <- list(author="Sonja Sahebzad", date=as.character(Sys.Date()),
  question="Can exact indexed inference reduce CPU time, and does a small content-word cache improve next-word ranking?",
  speed="Precompute sorted numeric context keys. Verify identical words and scores on all 1500 existing development/test cases, edge cases, and 4000 sampled trained contexts. Compare five randomized interleaved timing rounds on 600 development cases. Promote speed alone only with exact equality, at least 30% lower median time, no higher p95, and a prepared model below 256 MiB.",
  quality="Same 600 development cases; unchanged trained model. Compare cache weights 0.05, 0.10 and 0.20. Cache the last 100 normalized input words, exclude unknown words and the 100 highest base-probability words, and normalize remaining occurrence counts. Mix cache and full n-gram scores. Only the supplied prefix is used; no target or continuation is used in inference.",
  interactions="Interpolation allows earlier repeated content words to interact with local n-gram context. No fitted regression interaction terms are needed. Cache weight is the only varied quality setting.",
  selection="Require at least six more top-3 successes than current on development, no lower top-1, median below 20 ms and model below 256 MiB. Select highest top-3, then top-1, then smallest cache weight. Otherwise keep original scoring.",
  final="Only a qualifying frozen quality challenger may use the 900 cases reserved by speed-quality-20260925, still unscored. Require a positive lower bound of the paired source-stratified bootstrap 95% top-3 difference and no lower top-1. Do not inspect these cases if no challenger qualifies. Speed equivalence uses existing test cases for regression only.",
  timing="Warm local computation includes normalization and lookup, excludes loading, typing delay, networking and browser display. No artificial delay reduction is included in the computation speed claim.",
  seeds=list(contexts=20260930L,benchmark=20261001L,bootstrap=20261002L))
write_json(protocol,file.path(out,"protocol.json"),pretty=TRUE,auto_unbox=TRUE)
old <- new.env(); sys.source(file.path(out,"predictor-before.R"),envir=old)
source(file.path(out,"predictor.R"))
source(file.path(out,"cache_candidate.R"))
model <- readRDS(file.path(project,"final_project/app/model.rds"))
t0<-as.numeric(Sys.time()); fast <- prepare_predictor(model)
prepare_ms <- (as.numeric(Sys.time())-t0)*1000
dev <- fread(file.path(project,"data/adaptation/development.csv"))[,.(source,line_hash,prefix,actual)]
regression <- rbindlist(list(dev, fread(file.path(project,"data/final_product/final_test.csv"))[,.(source,line_hash,prefix,actual)]))
stopifnot(nrow(dev)==600L,nrow(regression)==1500L,
  !any(regression$line_hash %in% fread(file.path(project,"final_project/research/speed-quality-20260925/reserved_hashes.csv"))$line_hash))
edge <- c("", "123 456", "xqzz qzzx", "top of the", "the", "the the the", "Hello\nworld!",
          "I\u2019m looking forward to", "https://example.com @someone", paste(rep("word",100),collapse=" "))
set.seed(20260930L)
trained <- unlist(lapply(2:5,function(n) {
  tab<-model$tables[[n]]; idx<-sample.int(nrow(tab),1000L)
  vapply(idx,function(i) {
    ids<-as.integer(unlist(tab[i,paste0("w",seq_len(n-1L)),with=FALSE]))
    paste(model$vocabulary[ids[ids>0L]],collapse=" ")
  },character(1))
}))
phrases <- c(regression$prefix,edge,trained)
message("Check exact equivalence on ",length(phrases)," phrases")
for(phrase in phrases) {
  a<-old$predict_word(model,phrase); b<-predict_word(fast,phrase)
  stopifnot(identical(a,b))
}
for(phrase in head(dev$prefix,30L)) {
  a<-old$predict_word(model,phrase,dense=TRUE); b<-predict_word(fast,phrase,dense=TRUE)
  stopifnot(identical(a,b),abs(sum(b$distribution)-1)<1e-10)
  for(lambda in c(.05,.10,.20)) {
    c<-predict_cached(fast,phrase,lambda=lambda)
    d<-predict_cached(fast,phrase,lambda=lambda,dense=TRUE)
    stopifnot(identical(c$words,d$words),identical(c$scores,d$scores),abs(sum(d$distribution)-1)<1e-10)
  }
}
message("Exact equivalence passed. Compare cache settings on development only.")
config<-data.table(name=c("Current","Cache_05","Cache_10","Cache_20"),lambda=c(0,.05,.10,.20))
results<-lapply(seq_len(nrow(config)),function(j) {
  cfg<-config[j]
  detail<-rbindlist(lapply(seq_len(nrow(dev)),function(i) {
    start<-as.numeric(Sys.time())
    p<-if(cfg$lambda==0) predict_word(fast,dev$prefix[i]) else predict_cached(fast,dev$prefix[i],lambda=cfg$lambda)
    dt<-(as.numeric(Sys.time())-start)*1000
    data.table(source=dev$source[i],line_hash=dev$line_hash[i],rank=match(dev$actual[i],p$words,nomatch=0L),milliseconds=dt)
  }))
  fwrite(detail,file.path(out,paste0(cfg$name,"_development.csv")))
  data.table(name=cfg$name,lambda=cfg$lambda,cases=nrow(detail),top1=sum(detail$rank==1L),top3=sum(detail$rank>0L),median_ms=median(detail$milliseconds))
})
comparison<-rbindlist(results); print(comparison)
fwrite(comparison,file.path(out,"development_comparison.csv"))
message("Five randomized paired benchmark rounds")
set.seed(20261001L)
schedule<-CJ(round=1:5,case=seq_len(nrow(dev)))
schedule<-schedule[sample.int(.N)]
schedule[,fast_first:=sample(c(TRUE,FALSE),.N,replace=TRUE)]
timing<-rbindlist(lapply(seq_len(nrow(schedule)),function(k) {
  row<-schedule[k];phrase<-dev$prefix[row$case]
  order<-if(row$fast_first)c("Indexed","Original")else c("Original","Indexed")
  rbindlist(lapply(order,function(method) {
    t0<-as.numeric(Sys.time())
    p<-if(method=="Indexed")predict_word(fast,phrase)else old$predict_word(model,phrase)
    data.table(round=row$round,case=row$case,method=method,milliseconds=(as.numeric(Sys.time())-t0)*1000)
  }))
}))
fwrite(timing,file.path(out,"timing.csv"))
timing_summary<-timing[,.(calls=.N,median_ms=median(milliseconds),p95_ms=unname(quantile(milliseconds,.95))),by=method]
fwrite(timing_summary,file.path(out,"timing_summary.csv"));print(timing_summary)
size<-as.numeric(object.size(fast))/1024^2
base<-comparison[name=="Current"]
eligible<-comparison[name!="Current" & top3>=base$top3+6L & top1>=base$top1 & median_ms<20 & size<=256]
if(nrow(eligible)) {setorder(eligible,-top3,-top1,lambda);choice<-eligible$name[1L]}else choice<-"Current"
speed_ok<-timing_summary[method=="Indexed",median_ms] <= .7*timing_summary[method=="Original",median_ms] &&
  timing_summary[method=="Indexed",p95_ms] <= timing_summary[method=="Original",p95_ms] && size<=256
decision<-list(author="Sonja Sahebzad",selected_quality=choice,quality_final_test_used=FALSE,
  speed_qualified=speed_ok,regression_phrases=length(phrases),dense_checks=30L,
  model_md5=unname(tools::md5sum(file.path(project,"final_project/app/model.rds"))),
  original_memory_mib=as.numeric(object.size(model))/1024^2,prepared_memory_mib=size,prepare_ms=prepare_ms,
  median_reduction=1-timing_summary[method=="Indexed",median_ms]/timing_summary[method=="Original",median_ms],
  quality_reason=if(choice=="Current")"No cache variant met the predeclared development threshold. Keep original scoring."else"Candidate frozen before one reserved final comparison; not yet eligible for production.")
write_json(decision,file.path(out,"decision.json"),pretty=TRUE,auto_unbox=TRUE,digits=10)
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
print(decision)
