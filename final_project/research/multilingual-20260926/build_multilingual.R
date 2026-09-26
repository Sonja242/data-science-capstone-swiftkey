# Sonja Projects | Author: Sonja Sahebzad
# Fixed non-English prototypes; never opens the English reserved test set.
library(data.table);library(jsonlite);setDTthreads(2L)
out<-'final_project/research/multilingual-20260926';private<-'data/multilingual_20260926'
app<-'final_project/phraseflow_multilingual'
source('R/predictive_model_v3.R')
source(file.path(app,'predictor.R'))
cfg<-fromJSON(file.path(out,'protocol.json'))
if(file.exists(file.path(out,'metrics.json')))stop('Preserve completed measurements')
dir.create(file.path(app,'languages'),showWarnings=FALSE)
wilson<-function(k,n){z<-qnorm(.975);p<-k/n;d<-1+z*z/n;m<-(p+z*z/(2*n))/d;h<-z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d;c(low=m-h,high=m+h)}
compact_language<-function(original,lang){
 keep<-head(order(-original$base_probability,original$vocabulary)[order(-original$base_probability,original$vocabulary)!=length(original$vocabulary)],cfg$vocabulary_cap)
 vocabulary<-original$vocabulary[keep];map<-rep(-1L,length(original$vocabulary)+1L);map[1L]<-0L;map[keep+1L]<-seq_along(keep)
 base<-original$base_probability[keep];base<-base/sum(base);tables<-vector('list',cfg$maximum_order)
 for(n in 2:cfg$maximum_order){
  ctx<-paste0('w',seq_len(n-1L));tab<-copy(original$tables[[n]][total>=cfg$context_support,c(ctx,'word_id','weight'),with=FALSE])
  for(col in c(ctx,'word_id'))set(tab,j=col,value=map[tab[[col]]+1L])
  valid<-tab$word_id>0L;for(col in ctx)valid<-valid & tab[[col]]>=0L
  tab<-tab[valid];setorderv(tab,c(ctx,'weight','word_id'),c(rep(1L,length(ctx)),-1L,1L))
  tab<-tab[,head(.SD,cfg$followers),by=ctx];tab[,backoff:=pmax(0,1-sum(weight)),by=ctx];setkeyv(tab,ctx);tables[[n]]<-tab
 }
 lookup<-data.table(word=vocabulary,id=seq_along(vocabulary));setkey(lookup,word)
 list(version='phraseflow-language-1.0',language=lang,tokenizer='unicode',vocabulary=vocabulary,lookup=lookup,
 base_probability=base,base_order=order(-base,vocabulary),tables=tables,maximum_order=cfg$maximum_order,
 training_lines=original$training_lines,training_tokens=original$training_tokens,method='Pruned interpolated Kneser-Ney')
}
summarize<-function(d){n<-nrow(d);a<-sum(d$rank==1L);b<-sum(d$rank>0L);ci1<-wilson(a,n);ci3<-wilson(b,n)
 data.table(cases=n,top1_count=a,top3_count=b,top1=a/n,top3=b/n,top1_low=ci1[1],top1_high=ci1[2],top3_low=ci3[1],top3_high=ci3[2],
 availability=mean(d$available),vocabulary_coverage=mean(d$in_vocabulary),unigram_top1=mean(d$unigram_rank==1L),unigram_top3=mean(d$unigram_rank>0L))}
all_metrics<-list()
for(lang in cfg$languages){
 message('Language ',lang);began<-proc.time()[3]
 x<-fread(file.path(private,paste0(lang,'_sample.csv')),encoding='UTF-8')
 x[,text:=normalize_phrase(text,'unicode')];x<-unique(x,by='text')
 x[,n_words:=stringi::stri_count_fixed(text,' ')+1L];x<-x[nzchar(text)&n_words>=2L&n_words<=cfg$maximum_words_per_line]
 set.seed(cfg$seed+match(lang,cfg$languages));x[,random_order:=runif(.N)];setorder(x,source,random_order)
 x[,split:=ifelse(seq_len(.N)<=cfg$heldout_cases_per_source,'heldout','train'),by=source]
 train<-x[split=='train',head(.SD,cfg$training_cap_per_source),by=source];held<-x[split=='heldout']
 stopifnot(nrow(held)==600L,!any(train$text %in% held$text))
 train[,line_hash:=vapply(text,digest::digest,character(1),algo='sha256',serialize=FALSE)]
 held[,line_hash:=vapply(text,digest::digest,character(1),algo='sha256',serialize=FALSE)]
 tok<-strsplit(held$text,' ',fixed=TRUE);pos<-vapply(tok,function(w)sample(2:length(w),1L),integer(1))
 held[,prefix:=mapply(function(w,k)paste(head(w,k-1L),collapse=' '),tok,pos)];held[,actual:=mapply(function(w,k)w[k],tok,pos)]
 fwrite(train[,.(source,line_hash,text)],file.path(private,paste0(lang,'_train.csv')))
 fwrite(held[,.(source,line_hash,prefix,actual)],file.path(private,paste0(lang,'_heldout.csv')))
 counts<-integer_ngram_counts(train$text,cfg$maximum_order)
 original<-prepare_expanded_model(counts,cfg$maximum_order,'kneser_ney',cfg$minimum_count,.75)
 model<-compact_language(original,lang);rm(counts,original);gc(FALSE)
 path<-file.path(app,'languages',paste0(lang,'.rds'));saveRDS(model,path,compress='xz')
 model<-prepare_predictor(model);training_seconds<-proc.time()[3]-began
 details<-rbindlist(lapply(seq_len(nrow(held)),function(i){
   pred<-predict_word(model,held$prefix[i],10L)
   if(i<=30L){dense<-predict_word(model,held$prefix[i],10L,dense=TRUE);stopifnot(identical(pred$words,dense$words),abs(sum(dense$distribution)-1)<1e-10)}
   data.table(source=held$source[i],line_hash=held$line_hash[i],rank=match(held$actual[i],head(pred$words,3L),nomatch=0L),
    available=length(pred$words)==10L,in_vocabulary=held$actual[i] %in% model$vocabulary,
    unigram_rank=match(held$actual[i],head(model$vocabulary[model$base_order],3L),nomatch=0L))
 }))
 invisible(predict_word(model,held$prefix[1],10L));timings<-numeric(1200L);set.seed(cfg$seed)
 for(round in 1:2){order<-sample.int(600L);for(j in 1:600L){t<-as.numeric(Sys.time());p<-predict_word(model,held$prefix[order[j]],10L);timings[(round-1)*600L+j]<-1000*(as.numeric(Sys.time())-t)}}
 metric<-list(language=lang,overall=as.list(summarize(details)[1]),by_source=details[,summarize(.SD),by=source],
  training_lines=nrow(train),training_tokens=model$training_tokens,vocabulary=length(model$vocabulary),maximum_order=5,
  model_mib=file.info(path)$size/1024^2,memory_mib=as.numeric(object.size(model))/1024^2,
  median_ms=median(timings),p95_ms=unname(quantile(timings,.95)),timing_calls=length(timings),training_seconds=unname(training_seconds),
  model_md5=unname(tools::md5sum(path)),status='Experimental language prototype; fixed settings; initial held-out evaluation',
  baseline='Same-language unigram using training counts',confidence='Not calibrated',calibration_done=FALSE,
  domain='Official course blogs, news and Twitter; historical corpus',date=as.character(Sys.Date()))
 write_json(metric,file.path(app,'languages',paste0(lang,'_metrics.json')),auto_unbox=TRUE,pretty=TRUE,digits=10)
 fwrite(details,file.path(out,paste0(lang,'case_metrics.csv')))
 fwrite(data.table(milliseconds=timings),file.path(out,paste0(lang,'timings.csv')))
 all_metrics[[lang]]<-metric;print(summarize(details));rm(model,train,held,x);gc(FALSE)
}
write_json(all_metrics,file.path(out,'metrics.json'),auto_unbox=TRUE,pretty=TRUE,digits=10)
writeLines(capture.output(sessionInfo()),file.path(out,'session-info.txt'))
