# Sonja Projects | Author: Sonja Sahebzad
# Run from the corpus root. Only existing TRAINING and DEVELOPMENT files are read.
library(data.table);library(jsonlite);setDTthreads(1L)
out <- 'final_project/research/phraseflow-20260926'
private <- 'data/phraseflow_20260926'
stopifnot(file.exists(file.path(out,'protocol.json')))
if(file.exists(file.path(out,'training_manifest.json')))stop('Training preparation already completed.')
dir.create(private,recursive=TRUE,showWarnings=FALSE)
source('final_project/app/predictor.R')
model <- prepare_predictor(readRDS('final_project/app/model.rds'))
stopifnot(unname(tools::md5sum('final_project/app/model.rds'))=='bd58c642df1759c4ce1c39b9ace8927c')
train <- fread('data/student_context/teacher_training.csv')[,.(source,line_hash,prefix,actual)]
dev <- fread('data/student_context/development.csv')[,.(source,line_hash,prefix,actual)]
stopifnot(nrow(train)==3000L,nrow(dev)==600L,uniqueN(train$line_hash)==3000L,
 !any(train$line_hash %in% dev$line_hash),all(train[,.N,by=source]$N==1000L),
 identical(train$prefix,normalize_phrase(train$prefix)))
request_file <- file(file.path(private,'training_requests.jsonl'),'wt',encoding='UTF-8')
results <- vector('list',nrow(train));candidates <- vector('list',nrow(train))
for(i in seq_len(nrow(train))){
 t<-as.numeric(Sys.time());p<-predict_word(model,train$prefix[i],128L);ms<-(as.numeric(Sys.time())-t)*1000
 ids<-match(p$words,model$vocabulary)-1L
 stopifnot(length(ids)==128L,!anyDuplicated(ids),!anyNA(ids))
 writeLines(toJSON(list(case_id=i,source=train$source[i],line_hash=train$line_hash[i],prefix=p$normalized,
 candidate_ids=ids,candidate_words=p$words),auto_unbox=TRUE,digits=16),request_file)
 # Targets are consulted only after all candidates for this prefix exist.
 rank<-match(train$actual[i],p$words,nomatch=0L)
 results[[i]]<-data.table(case_id=i,source=train$source[i],line_hash=train$line_hash[i],cpu_rank=rank,
  in_vocabulary=train$actual[i] %in% model$vocabulary,context_words=p$input_words,cpu_ms=ms)
 candidates[[i]]<-data.table(case_id=i,word_id=ids,cpu_probability=p$scores)
 if(i%%500L==0L)cat('Training prefix candidates:',i,'/3000\n')
}
close(request_file)
fwrite(rbindlist(results),file.path(out,'training_cases.csv'))
fwrite(rbindlist(candidates),file.path(private,'training_candidates.csv'))
write_json(list(cases=3000L,development_cases=600L,training_covered=sum(vapply(results,function(r)r$cpu_rank>0L,logical(1))),
 model_md5=unname(tools::md5sum('final_project/app/model.rds')),
 training_sha256=digest::digest(file='data/student_context/teacher_training.csv',algo='sha256'),
 development_sha256=digest::digest(file='data/student_context/development.csv',algo='sha256'),
 requests_sha256=digest::digest(file=file.path(private,'training_requests.jsonl'),algo='sha256'),
 protocol_sha256=digest::digest(file=file.path(out,'protocol.json'),algo='sha256'),
 generated_from_prefix_only=TRUE,target_inserted=FALSE,reserved_test_opened=FALSE,production_changed=FALSE,
 completed_utc=format(Sys.time(),tz='UTC',usetz=TRUE)),file.path(out,'training_manifest.json'),auto_unbox=TRUE,pretty=TRUE)
