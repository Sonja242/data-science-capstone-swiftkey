# Sonja Projects | Author: Sonja Sahebzad
# A bounded development comparison. Never overwrites production or prior results.
library(data.table)
library(jsonlite)
setDTthreads(2L)
project <- normalizePath(commandArgs(trailingOnly=TRUE)[1],winslash="/")
out <- file.path(project,"final_project/research/speed-quality-20260925")
private <- file.path(project,"data/final_product_quality")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
dir.create(private,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(out,"protocol.json"))) stop("Experiment already started. Preserve it.")
source(file.path(project,"final_project/app/predictor.R"))
protocol <- list(author="Sonja Sahebzad",date=as.character(Sys.Date()),
  question="Does relaxing one compression setting improve the existing CPU product?",
  development="The existing 600 cases, 200 per source. These are reused development data, not independent performance estimates.",
  candidates="Current; support 2 instead of 5; 16 retained followers instead of 8; vocabulary 100000 instead of 50000. All other settings unchanged.",
  resources="At most 256 MiB for the R model object and 50 MiB xz file, leaving headroom within the default 1024 MiB Shiny instance. This study relaxes the previous 200 MiB object budget before comparison.",
  selection="Require at least 6 additional correct top-3 cases (+1 percentage point), no fewer top-1 successes, and the resource limits. Among qualifying candidates maximize top-3, then top-1, then smaller memory. Otherwise retain production.",
  final_test="Reserve 900 fresh cases before comparison. Only if a candidate qualifies, compare frozen winner and production once on these same cases. Never reuse the old 900 cases for selection.",
  promotion="A candidate is eligible for release only if the paired stratified bootstrap 95% interval for the top-3 difference is above zero, top-1 does not decrease, and median inference is below 20 ms. Otherwise retain production. Do not tune further against the final test.",
  ui="Separately test 150 ms instead of 400 ms automatic waiting. This does not alter model outputs.",
  seeds=list(reservation=20260927L,word_position=20260928L,bootstrap=20260929L))
write_json(protocol,file.path(out,"protocol.json"),auto_unbox=TRUE,pretty=TRUE)

message("Reserve unseen final cases")
rows <- readRDS(file.path(project,"models/expanded_partitions_v3.rds")); setDT(rows)
rows[,row_id:=.I]
eligible <- rows[split=="test" & stringi::stri_count_fixed(text," ")>=2L]
set.seed(20261103L)
old <- eligible[,.SD[sample.int(.N,min(.N,1000L))],by=source]
pool <- eligible[!row_id %in% old$row_id]
hash_files <- c("data/neural_evaluation/test.csv","data/adaptation/test.csv",
 "data/learning_curve/final_test.csv","models/learning_curve_reserved_case_hashes.csv",
 "models/word_generation_reserved_case_hashes.csv","models/ranking_study_reserved_hashes.csv",
 "final_project/results/reserved_hashes.csv")
excluded <- unique(unlist(lapply(hash_files,function(f) {
  x<-fread(file.path(project,f)); stopifnot("line_hash" %in% names(x)); x$line_hash
})))
set.seed(20260927L)
reserved <- pool[,.SD[sample.int(.N,min(.N,1500L))],by=source]
reserved[,line_hash:=vapply(text,digest::digest,character(1),algo="sha256",serialize=FALSE)]
reserved <- reserved[!line_hash %in% excluded,head(.SD,300L),by=source]
stopifnot(nrow(reserved)==900L,!anyDuplicated(reserved$text),
  !any(reserved$text %in% rows[split=="train",text]),
  !any(reserved$text %in% rows[split=="validation",text]))
set.seed(20260928L)
tokens<-strsplit(reserved$text," ",fixed=TRUE)
positions<-vapply(tokens,function(w) sample(2:length(w),1L),integer(1))
reserved[,prefix:=mapply(function(w,k) paste(head(w,k-1L),collapse=" "),tokens,positions)]
reserved[,actual:=mapply(function(w,k) w[k],tokens,positions)]
fwrite(reserved[,.(source,line_hash,prefix,actual)],file.path(private,"reserved_test.csv"))
fwrite(reserved[,.(source,line_hash)],file.path(out,"reserved_hashes.csv"))
rm(rows,eligible,old,pool,reserved,tokens);gc(FALSE)

production<-readRDS(file.path(project,"final_project/app/model.rds"))
original<-readRDS(file.path(project,"models/selected_predictor_v3.rds"))
dev<-fread(file.path(project,"data/adaptation/development.csv"))[,.(source,line_hash,prefix,actual)]
stopifnot(nrow(dev)==600L,!any(dev$line_hash %in% fread(file.path(out,"reserved_hashes.csv"))$line_hash))
original_order<-order(-original$base_probability,original$vocabulary)
original_order<-original_order[!original$vocabulary[original_order] %in% c("<unk>",production$blocked_terms)]
compact <- function(vocab_size,support,followers) {
  keep<-head(original_order,vocab_size)
  vocabulary<-original$vocabulary[keep]
  mapping<-rep.int(-1L,length(original$vocabulary)+1L)
  mapping[1L]<-0L; mapping[keep+1L]<-seq_along(keep)
  base<-original$base_probability[keep]; base<-base/sum(base)
  tables<-vector("list",5L)
  for(n in 2:5) {
    ctx<-paste0("w",seq_len(n-1L))
    tab<-copy(original$tables[[n]][total>=support,c(ctx,"word_id","weight"),with=FALSE])
    for(col in c(ctx,"word_id")) set(tab,j=col,value=mapping[tab[[col]]+1L])
    valid<-tab$word_id>0L
    for(col in ctx) valid<-valid & tab[[col]]>=0L
    tab<-tab[valid]
    setorderv(tab,c(ctx,"weight","word_id"),c(rep(1L,length(ctx)),-1L,1L))
    tab<-tab[rowidv(tab,cols=ctx)<=followers]
    tab[,backoff:=pmax(0,1-sum(weight)),by=ctx]
    setkeyv(tab,ctx); tables[[n]]<-tab
  }
  lookup<-data.table(word=vocabulary,id=seq_along(vocabulary)); setkey(lookup,word)
  list(version="shiny-quality-study-1.0",vocabulary=vocabulary,lookup=lookup,
    base_probability=base,base_order=order(-base,vocabulary),tables=tables,
    maximum_order=5L,support=support,followers=followers,
    training_lines=original$training_lines,training_tokens=original$training_tokens,
    method=production$method,blocked_terms=production$blocked_terms)
}
evaluate<-function(model,cases) {
  invisible(predict_word(model,"a warm up phrase"))
  rbindlist(lapply(seq_len(nrow(cases)),function(i) {
    start<-as.numeric(Sys.time()); p<-predict_word(model,cases$prefix[i])
    elapsed<-(as.numeric(Sys.time())-start)*1000
    data.table(source=cases$source[i],line_hash=cases$line_hash[i],
      rank=match(cases$actual[i],p$words,nomatch=0L),milliseconds=elapsed)
  }))
}
configs<-data.table(name=c("Current","More_contexts","More_followers","Larger_vocabulary"),
  vocabulary=c(50000L,50000L,50000L,100000L),support=c(5L,2L,5L,5L),followers=c(8L,8L,16L,8L))
summaries<-list()
for(i in seq_len(nrow(configs))) {
  cfg<-configs[i];message("Evaluate ",cfg$name)
  model<-if(i==1L) production else compact(cfg$vocabulary,cfg$support,cfg$followers)
  detail<-evaluate(model,dev)
  fwrite(detail,file.path(out,paste0(cfg$name,"_development.csv")))
  memory<-as.numeric(object.size(model))/1024^2
  # Save potential contenders only; every construction remains reproducible.
  path<-file.path(private,paste0(cfg$name,".rds"))
  if(i==1L) size<-file.info(file.path(project,"final_project/app/model.rds"))$size/1024^2 else {
    if(memory<=256) {saveRDS(model,path,compress="xz");size<-file.info(path)$size/1024^2} else size<-NA_real_
  }
  summaries[[i]]<-cbind(cfg,data.table(cases=nrow(dev),top1=sum(detail$rank==1L),
    top3=sum(detail$rank>0L),median_ms=median(detail$milliseconds),
    p95_ms=unname(quantile(detail$milliseconds,.95)),object_mib=memory,file_mib=size))
  print(summaries[[i]]); rm(model);gc(FALSE)
}
comparison<-rbindlist(summaries)
fwrite(comparison,file.path(out,"development_comparison.csv"))
base<-comparison[name=="Current"]
qualified<-comparison[name!="Current" & top3>=base$top3+6L & top1>=base$top1 & object_mib<=256 & !is.na(file_mib) & file_mib<=50]
if(nrow(qualified)) {
  setorderv(qualified,c("top3","top1","object_mib"),c(-1L,-1L,1L))
  choice<-qualified$name[1L]
} else choice<-"Current"
write_json(list(selected=choice,selected_before_final_evaluation=TRUE,
  final_test_used=FALSE,reason=if(choice=="Current") "No challenger satisfies the predeclared development improvement and resource criteria." else "Qualifies for one reserved final comparison; production not replaced."),
  file.path(out,"selection.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
message("Finished development comparison. Selected: ",choice)
