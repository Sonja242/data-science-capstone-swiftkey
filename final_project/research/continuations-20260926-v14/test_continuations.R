library(data.table);library(jsonlite)
app<-'final_project/phraseflow_multilingual';out<-'final_project/research/continuations-20260926-v14'
source(file.path(app,'predictor.R'));source(file.path(app,'continuations.R'))
toy<-data.table(context=c('we houden van','we houden van','lang en','lang en'),continuation=c('elkaar','ons land','gelukkig leven','prachtig'),count=c(5L,2L,3L,2L));setkey(toy,context)
stopifnot(identical(predict_continuations(toy,'we houden van','unicode')$text,c('elkaar','ons land')),
 identical(predict_continuations(toy,'we leven lang en','unicode')$text,c('gelukkig','prachtig')),
 !length(predict_continuations(toy,'abc xyz onbekend','unicode')$text),
 !length(predict_continuations(toy,'','unicode')$text),
 identical(predict_continuations(toy,'we houden van','unicode',exclude='elkaar')$text,'ons land'))
zh<-data.table(context='\u4eca\u5929 \u5929\u6c14',continuation='\u5f88 \u597d',count=2L);setkey(zh,context)
# No added spaces in Chinese suggestions after normalizing the input's real ICU units.
zprefix<-'\u4eca\u5929\u5929\u6c14';ctx<-normalize_phrase(zprefix,'chinese');zh[,context:=ctx];setkey(zh,context)
stopifnot(length(predict_continuations(zh,zprefix,'chinese')$text)==1L,
 !any(grepl(' ',predict_continuations(zh,zprefix,'chinese')$text,fixed=TRUE)))
en<-prepare_predictor(readRDS(file.path(app,'model.rds')));idx<-readRDS(file.path(app,'continuations/en_US.rds'))
dev<-fread('data/student_context/development.csv');times<-numeric(nrow(dev))
for(i in seq_len(nrow(dev))){
 t<-as.numeric(Sys.time());a<-predict_word(en,dev$prefix[i],20L);c<-predict_continuations(idx,dev$prefix[i],'english');times[i]<-(as.numeric(Sys.time())-t)*1000
 b<-predict_word(en,dev$prefix[i],10L)
 stopifnot(identical(head(a$words,10),b$words),identical(head(a$scores,10),b$scores))
}
fwrite(data.table(milliseconds=times),file.path(out,'cpu_timings.csv'))
write_json(list(cases=length(times),purpose='Warm CPU timing on reused English development prefixes; not a quality evaluation',
 median_ms=median(times),p95_ms=unname(quantile(times,.95)),word_candidates=20,
 includes='Next-word prediction and supplementary ending retrieval',excludes='Loading, 150ms debounce, rendering, network',
 unchanged_top10_all_600=TRUE,toy_context_and_no_match_checks=TRUE),file.path(out,'continuation_checks.json'),auto_unbox=TRUE,pretty=TRUE)
