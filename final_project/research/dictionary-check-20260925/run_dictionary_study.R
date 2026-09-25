# Sonja Projects | Author: Sonja Sahebzad
# Dictionary recognition is a ranking experiment, not extra context training.
library(data.table)
library(jsonlite)
args <- commandArgs(trailingOnly=TRUE)
project <- normalizePath(args[1],winslash="/")
out <- normalizePath(args[2],winslash="/")
if(file.exists(file.path(out,"protocol.json"))) stop("Preserve the completed or started experiment.")
protocol <- list(author="Sonja Sahebzad",date=as.character(Sys.Date()),
  objective="Test whether English spelling-dictionary recognition improves next-word ranking without a material latency increase.",
  dictionary="hunspell 3.0.6; union of bundled en_US and en_GB. Recognize a lowercase vocabulary word if its lowercase, initial-capital or uppercase form passes either dictionary. Dictionary recognition is computed once, never per prediction. Do not add any development/test answer to the dictionary.",
  candidates="Current penalty 1; unrecognized-word multipliers 0.5, 0.1 and 0. All recognized words retain multiplier 1. Apply multipliers to the complete n-gram probability, normalize, and rank. Recompute the base-word ordering so sparse inference remains exactly equivalent to dense scoring.",
  diagnosis="On the same 600 development cases, distinguish correct top-3, known words ranked too low, and words outside the retained vocabulary. Independently record dictionary recognition of targets and suggestions. Non-recognition is not a verified spelling error.",
  selection="Use only the 600 development cases, 200 per source. Require at least six additional top-3 successes, no fewer top-1 successes, prepared model below 256 MiB, and median time at most 125% of current and below 20 ms. Three randomized interleaved rounds, 1800 timed calls per method. Select highest top-3, then top-1, then least aggressive penalty; otherwise retain production.",
  final="Only a qualifying frozen winner may be tested once on the 900 still-unused cases reserved by speed-quality-20260925. Require paired source-stratified bootstrap 95% top-3 difference lower bound above zero, no lower top-1, and the stated resource limits before deployment. No candidate qualification means no final-test access.",
  interactions="The multiplicative dictionary factor interacts with existing context-dependent scores. It is varied separately; context weights are not refitted. Three strengths assess sensitivity to the dictionary assumption.",
  limitations="Dictionaries test recognized spellings and inflections, not whether a word is the intended next word. Names, new terms, contractions and informal spellings may be missing. This experiment does not retrain the model on dictionary definitions or add vocabulary.",
  seeds=list(timing=20261003L,bootstrap=20261004L))
write_json(protocol,file.path(out,"protocol.json"),auto_unbox=TRUE,pretty=TRUE)
stopifnot(as.character(packageVersion("hunspell"))=="3.0.6")
source(file.path(out,"predictor_snapshot.R"))
model <- prepare_predictor(readRDS(file.path(project,"final_project/app/model.rds")))
dev <- fread(file.path(project,"data/adaptation/development.csv"))[,.(source,line_hash,prefix,actual)]
stopifnot(nrow(dev)==600L,!any(dev$line_hash %in% fread(file.path(project,"final_project/research/speed-quality-20260925/reserved_hashes.csv"))$line_hash))
dicts<-lapply(c("en_US","en_GB"),hunspell::dictionary)
recognized<-function(words) {
  forms<-list(words,stringi::stri_trans_totitle(words),toupper(words))
  Reduce(`|`,lapply(dicts,function(dict) Reduce(`|`,lapply(forms,function(w) hunspell::hunspell_check(w,dict=dict)))))
}
known<-recognized(model$vocabulary)
model$dictionary_recognized<-known
dict_files<-list.files(system.file(package="hunspell"),recursive=TRUE,full.names=TRUE,pattern="en_(US|GB)\\.(dic|aff)$")
stopifnot(length(dict_files)==4L)
fwrite(data.table(file=basename(dict_files),md5=unname(tools::md5sum(dict_files))),file.path(out,"dictionary_checksums.csv"))
fwrite(data.table(word_id=seq_along(known),recognized=known),file.path(out,"dictionary_flags.csv"))
diagnosis<-rbindlist(lapply(seq_len(nrow(dev)),function(i) {
  p<-predict_word(model,dev$prefix[i],dense=TRUE)
  target_id<-match(dev$actual[i],model$vocabulary)
  target_rank<-if(is.na(target_id))NA_integer_ else match(target_id,order(-p$distribution,model$vocabulary))
  data.table(source=dev$source[i],line_hash=dev$line_hash[i],in_vocabulary=!is.na(target_id),
    target_recognized=recognized(dev$actual[i]),rank=target_rank,
    top1=identical(dev$actual[i],p$words[1L]),top3=dev$actual[i] %in% p$words,
    unrecognized_suggestions=sum(!known[match(p$words,model$vocabulary)]))
}))
fwrite(diagnosis,file.path(out,"diagnosis.csv"))
diagnostic_summary<-list(cases=600L,vocabulary=length(known),recognized_vocabulary=sum(known),
  top3=sum(diagnosis$top3),known_outside_top3=sum(diagnosis$in_vocabulary & !diagnosis$top3),
  outside_vocabulary=sum(!diagnosis$in_vocabulary),recognized_targets=sum(diagnosis$target_recognized),
  unrecognized_top3_suggestions=sum(diagnosis$unrecognized_suggestions))
write_json(diagnostic_summary,file.path(out,"diagnostic_summary.json"),pretty=TRUE,auto_unbox=TRUE)
print(diagnostic_summary)

# Exact sparse reweighting. All context-contributing words plus the best
# reweighted base words contain the complete reweighted distribution's top N.
prepare_dictionary_rule<-function(model,penalty) {
  model$dictionary_multiplier<-ifelse(model$dictionary_recognized,1,penalty)
  model$dictionary_base_order<-order(-model$base_probability*model$dictionary_multiplier,model$vocabulary)
  model$dictionary_base_mass<-sum(model$base_probability*model$dictionary_multiplier)
  model
}
predict_dictionary<-function(model,phrase,top_n=3L,dense=FALSE) {
  clean<-normalize_phrase(phrase)
  words<-if(nzchar(clean))strsplit(clean," ",fixed=TRUE)[[1L]]else character()
  ids<-model$lookup[list(tail(words,model$maximum_order-1L)),id];ids[is.na(ids)] <- -1L
  ids<-c(rep.int(0L,model$maximum_order-1L),ids)
  hits<-lapply(2:model$maximum_order,function(n) {
    at<-context_rows(model,n,tail(ids,n-1L));tab<-model$tables[[n]]
    list(word_id=tab$word_id[at],weight=tab$weight[at],backoff=tab$backoff[at])
  })
  candidates<-if(dense)seq_along(model$vocabulary)else unique(c(head(model$dictionary_base_order,top_n),unlist(lapply(hits,`[[`,"word_id"))))
  p<-model$base_probability[candidates]
  mass<-model$dictionary_base_mass
  for(tab in hits) {
    if(!length(tab$word_id))next
    p<-p*tab$backoff[1L]
    at<-match(tab$word_id,candidates);p[at]<-p[at]+tab$weight
    mass<-mass*tab$backoff[1L]+sum(tab$weight*model$dictionary_multiplier[tab$word_id])
  }
  stopifnot(mass>0)
  p<-p*model$dictionary_multiplier[candidates]/mass
  take<-head(order(-p,model$vocabulary[candidates]),top_n)
  list(words=model$vocabulary[candidates[take]],scores=p[take],distribution=if(dense)p else NULL)
}
configs<-data.table(name=c("Current","Gentle","Strong","Strict"),penalty=c(1,.5,.1,0))
variants<-lapply(configs$penalty,function(p)prepare_dictionary_rule(model,p))
message("Verify exact dictionary ranking and compare development accuracy")
for(j in seq_along(variants)) for(phrase in c("","xqzz",head(dev$prefix,50L))) {
  a<-predict_dictionary(variants[[j]],phrase);b<-predict_dictionary(variants[[j]],phrase,dense=TRUE)
  stopifnot(identical(a$words,b$words),isTRUE(all.equal(a$scores,b$scores,tolerance=1e-14)),abs(sum(b$distribution)-1)<1e-10)
  if(j==1L)stopifnot(identical(a$words,predict_word(model,phrase)$words))
}
details<-rbindlist(lapply(seq_len(nrow(configs)),function(j) {
  rbindlist(lapply(seq_len(nrow(dev)),function(i) {
    p<-if(j==1L)predict_word(model,dev$prefix[i])else predict_dictionary(variants[[j]],dev$prefix[i])
    data.table(name=configs$name[j],source=dev$source[i],line_hash=dev$line_hash[i],
      rank=match(dev$actual[i],p$words,nomatch=0L),unrecognized_suggestions=sum(!known[match(p$words,model$vocabulary)]))
  }))
}))
fwrite(details,file.path(out,"development_predictions.csv"))
set.seed(20261003L)
schedule<-CJ(round=1:3,case=1:600)[sample.int(.N)]
timing<-rbindlist(lapply(seq_len(nrow(schedule)),function(k) {
  row<-schedule[k];order<-sample.int(4L)
  rbindlist(lapply(order,function(j) {
    start<-as.numeric(Sys.time())
    p<-if(j==1L)predict_word(model,dev$prefix[row$case])else predict_dictionary(variants[[j]],dev$prefix[row$case])
    data.table(round=row$round,case=row$case,name=configs$name[j],milliseconds=(as.numeric(Sys.time())-start)*1000)
  }))
}))
fwrite(timing,file.path(out,"timing.csv"))
quality<-details[,.(cases=.N,top1=sum(rank==1L),top3=sum(rank>0L),unrecognized_suggestions=sum(unrecognized_suggestions)),by=name]
quality<-merge(configs,quality,by="name",sort=FALSE)
quality<-merge(quality,timing[,.(median_ms=median(milliseconds),p95_ms=unname(quantile(milliseconds,.95))),by=name],by="name",sort=FALSE)
quality[,memory_mib:=vapply(variants,function(m)as.numeric(object.size(m))/1024^2,numeric(1))]
quality[name=="Current",memory_mib:=as.numeric(object.size(model))/1024^2]
baseline<-details[name=="Current"]
quality[,`:=`(gained=0L,lost=0L)]
for(label in configs$name[-1L]) {
  paired<-merge(baseline,details[name==label],by=c("source","line_hash"))
  quality[name==label,`:=`(gained=sum(paired$rank.x==0 & paired$rank.y>0),lost=sum(paired$rank.x>0 & paired$rank.y==0))]
}
fwrite(quality,file.path(out,"development_comparison.csv"));print(quality)
base<-quality[name=="Current"]
eligible<-quality[name!="Current" & top3>=base$top3+6L & top1>=base$top1 & memory_mib<=256 & median_ms<=1.25*base$median_ms & median_ms<20]
if(nrow(eligible)){setorder(eligible,-top3,-top1,-penalty);selected<-eligible$name[1L]}else selected<-"Current"
decision<-list(author="Sonja Sahebzad",selected=selected,final_test_used=FALSE,production_changed=FALSE,
  model_md5=unname(tools::md5sum(file.path(project,"final_project/app/model.rds"))),
  hunspell_version=as.character(packageVersion("hunspell")),dense_equivalence_cases=52L,
  reason=if(selected=="Current")"No dictionary rule meets the predeclared accuracy and resource criteria. Retain deployed scoring."else"Candidate frozen for a reserved final comparison; not yet promoted.")
write_json(decision,file.path(out,"decision.json"),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
print(decision)
