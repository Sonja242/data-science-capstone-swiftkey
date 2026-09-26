# Sonja Projects | Sonja Sahebzad | Fixed settings, independent sentence hold-outs.
library(data.table);library(jsonlite);setDTthreads(2L)
out<-'final_project/research/languages-20260926-v13';private<-'data/languages_20260926_v13'
app<-'final_project/phraseflow_multilingual'
source('R/predictive_model_v3.R');source(file.path(app,'predictor.R'))
cfg<-fromJSON(file.path(out,'protocol.json'),simplifyVector=FALSE)
if(file.exists(file.path(out,'metrics.json')))stop('Preserve completed evaluation')
# Reuse the already tested compaction and interval functions, without running its experiment.
old_source<-readLines('final_project/research/multilingual-20260926/build_multilingual.R',warn=FALSE)
a<-grep('^wilson<-',old_source);b<-grep('^all_metrics<-',old_source)
eval(parse(text=old_source[a:(b-1L)]))
dir.create(file.path(app,'www','attribution'),recursive=TRUE,showWarnings=FALSE)
all_metrics<-list()
for(lang in names(cfg$languages)){
 message('Training ',lang);d<-cfg$languages[[lang]];start<-proc.time()[3]
 x<-fread(file.path(private,paste0(lang,'_sample.csv')),encoding='UTF-8',colClasses='character')
 x[,text:=normalize_phrase(text,d$tokenizer)];x<-unique(x,by='text')
 x[,n_words:=stringi::stri_count_fixed(text,' ')+1L]
 x<-x[nzchar(text)&n_words>=2L&n_words<=cfg$maximum_words_per_line]
 stopifnot(nrow(x)>cfg$heldout_cases+1000)
 set.seed(cfg$seed+match(lang,names(cfg$languages)))
 x<-x[sample.int(.N)];held<-x[seq_len(cfg$heldout_cases)];train<-head(x[-seq_len(cfg$heldout_cases)],cfg$training_cap)
 stopifnot(!any(train$text %in% held$text),!any(train$sentence_id %in% held$sentence_id))
 held[,line_hash:=vapply(text,digest::digest,character(1),algo='sha256',serialize=FALSE)]
 tok<-strsplit(held$text,' ',fixed=TRUE);pos<-vapply(tok,function(w)sample(2:length(w),1L),integer(1))
 sep<-if(d$tokenizer=='chinese')'' else ' '
 held[,prefix:=mapply(function(w,k)paste(head(w,k-1L),collapse=sep),tok,pos)]
 held[,actual:=mapply(function(w,k)w[k],tok,pos)]
 fwrite(train[,.(sentence_id,owner,text)],file.path(private,paste0(lang,'_train.csv')))
 fwrite(held[,.(sentence_id,owner,line_hash,prefix,actual)],file.path(private,paste0(lang,'_heldout.csv')))
 counts<-integer_ngram_counts(train$text,cfg$maximum_order)
 original<-prepare_expanded_model(counts,cfg$maximum_order,'kneser_ney',cfg$minimum_count,.75)
 model<-compact_language(original,lang);model$tokenizer<-d$tokenizer
 model$version<-'phraseflow-language-1.3';model$training_source<-'Tatoeba';model$prediction_unit<-if(lang=='ko_KR')'eojeol' else if(lang=='zh_CN')'ICU segmented word' else 'normalized word'
 rm(counts,original);gc(FALSE)
 path<-file.path(app,'languages',paste0(lang,'.rds'));stopifnot(!file.exists(path));saveRDS(model,path,compress='xz')
 model<-prepare_predictor(model);training_seconds<-proc.time()[3]-start
 details<-rbindlist(lapply(seq_len(nrow(held)),function(i){
  pred<-predict_word(model,held$prefix[i],10L)
  if(i<=30L){dense<-predict_word(model,held$prefix[i],10L,dense=TRUE);stopifnot(identical(pred$words,dense$words),abs(sum(dense$distribution)-1)<1e-10)}
  data.table(source='Tatoeba',line_hash=held$line_hash[i],rank=match(held$actual[i],head(pred$words,3L),nomatch=0L),
   available=length(pred$words)==10L,in_vocabulary=held$actual[i] %in% model$vocabulary,
   unigram_rank=match(held$actual[i],head(model$vocabulary[model$base_order],3L),nomatch=0L))
 }))
 timings<-numeric(1200L);set.seed(cfg$seed)
 for(round in 1:2){ord<-sample.int(600L);for(j in 1:600L){t<-as.numeric(Sys.time());p<-predict_word(model,held$prefix[ord[j]],10L);timings[(round-1)*600L+j]<-1000*(as.numeric(Sys.time())-t)}}
 metric<-list(language=lang,language_name=d$name,overall=as.list(summarize(details)[1]),by_source=details[,summarize(.SD),by=source],
  training_lines=nrow(train),training_tokens=model$training_tokens,vocabulary=length(model$vocabulary),maximum_order=5,
  model_mib=file.info(path)$size/1024^2,memory_mib=as.numeric(object.size(model))/1024^2,
  median_ms=median(timings),p95_ms=unname(quantile(timings,.95)),timing_calls=length(timings),training_seconds=unname(training_seconds),
  model_md5=unname(tools::md5sum(path)),status='Experimental; fixed settings; initial Tatoeba sentence hold-out',
  baseline='Same-language unigram using training data',confidence='Not calibrated',calibration_done=FALSE,
  domain='Tatoeba volunteer example sentences; not a news/chat benchmark',date=as.character(Sys.Date()),
  tokenizer=d$tokenizer,prediction_unit=model$prediction_unit,exact_duplicate_overlap=0)
 write_json(metric,file.path(app,'languages',paste0(lang,'_metrics.json')),auto_unbox=TRUE,pretty=TRUE,digits=10)
 fwrite(details,file.path(out,paste0(lang,'_case_metrics.csv')));fwrite(data.table(milliseconds=timings),file.path(out,paste0(lang,'_timings.csv')))
 # Attribution contains identifiers and credited contributors, never held-out answers.
 credits<-train[,.(sentence_id,owner,url=paste0('https://tatoeba.org/en/sentences/show/',sentence_id))]
 fwrite(credits,file.path(app,'www/attribution',paste0(lang,'_training_credits.tsv.gz')),sep='\t')
 all_metrics[[lang]]<-metric;print(summarize(details));rm(model,train,held,x);gc(FALSE)
}
write_json(all_metrics,file.path(out,'metrics.json'),auto_unbox=TRUE,pretty=TRUE,digits=10)
writeLines(capture.output(sessionInfo()),file.path(out,'session-info.txt'))
write_json(stringi::stri_info(),file.path(out,'unicode-runtime.json'),auto_unbox=TRUE,pretty=TRUE)
