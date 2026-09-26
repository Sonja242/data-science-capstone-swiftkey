library(data.table);library(jsonlite);library(shiny)
out<-'final_project/research/languages-20260926-v13';app<-'final_project/phraseflow_multilingual'
source(file.path(app,'predictor.R'))
stopifnot(normalize_phrase('e\u0301cole','unicode')==normalize_phrase('\u00e9cole','unicode'))
hindi<-'\u092e\u0941\u091d\u0947 \u0918\u0930 \u091c\u093e\u0928\u093e'
korean<-'\uc800\ub294 \ud55c\uad6d\uc5b4\ub97c'
chinese<-'\u6211\u60f3\u53bb\u5317\u4eac'
stopifnot(normalize_phrase(hindi,'unicode')==hindi,normalize_phrase(korean,'unicode')==korean)
segments<-normalize_phrase(chinese,'chinese')
stopifnot(gsub(' ','',segments,fixed=TRUE)==chinese,length(strsplit(segments,' ',fixed=TRUE)[[1]])>=2L)
baseline<-new.env();sys.source('final_project/app/predictor.R',envir=baseline)
en<-prepare_predictor(readRDS(file.path(app,'model.rds')));old<-baseline$prepare_predictor(readRDS('final_project/app/model.rds'))
dev<-fread('data/student_context/development.csv')
for(phrase in dev$prefix){a<-predict_word(en,phrase,10L);b<-baseline$predict_word(old,phrase,10L);stopifnot(identical(a$words,b$words),identical(a$scores,b$scores))}
rm(en,old);invisible(gc(FALSE))
app_env<-new.env();local({previous<-setwd(app);on.exit(setwd(previous));sys.source('app.R',envir=app_env)})
stopifnot(length(app_env$language_labels)==12L)
testServer(app_env$server,{
 session$setInputs(language='en_US',phrase='',automatic=TRUE,composing=FALSE);session$flushReact()
 for(code in names(app_env$language_labels)){
  message('Switch check: ',code)
  session$setInputs(language=code);session$flushReact();stopifnot(is.null(result()))
  session$setInputs(phrase='');session$elapse(200)
  phrase<-paste0(app_env$language_examples[[code]][1],if(code=='zh_CN')'' else ' ')
  session$setInputs(phrase=phrase);session$elapse(200)
  stopifnot(length(result()$words)==10L,result()$language==code,length(ls(app_env$model_cache))<=4L)
  message('Composition check: ',code)
  session$setInputs(composing=TRUE,phrase=paste0(phrase,'x'));session$elapse(200);stopifnot(is.null(result()))
  session$setInputs(composing=FALSE,predict=sample.int(100000,1));stopifnot(length(result()$words)==10L)
  session$setInputs(phrase=paste(rep('word',110),collapse=' '),predict=sample.int(100000,1));stopifnot(grepl('500',error()))
  session$setInputs(phrase='');session$elapse(200)
 }
 message('Final stale check')
 session$setInputs(language='en_US',phrase='');session$elapse(200)
 session$setInputs(phrase='I am looking forward to ');session$elapse(200);stopifnot(!is.null(result()))
 session$setInputs(language='nl_NL');session$elapse(200);stopifnot(is.null(result()))
})
models<-c('en_US',names(app_env$language_labels)[-1]);memory<-vapply(models,function(code){m<-app_env$get_model(code);as.numeric(object.size(m))/1024^2},numeric(1))
cache_bound<-memory[['en_US']]+sum(head(sort(memory[names(memory)!='en_US'],decreasing=TRUE),3))
check<-list(languages=12,english_exact_600=TRUE,english_reserved_test_opened=FALSE,
 unicode_hindi_korean_chinese=TRUE,automatic_and_manual_all_languages=TRUE,
 composition_pause=TRUE,language_switch_clears_results=TRUE,cache_maximum_four=TRUE,
 prepared_model_objects_maximum_mib=cache_bound,model_memory_mib=as.list(memory),
 english_model_md5=unname(tools::md5sum(file.path(app,'model.rds'))),production_model_md5=unname(tools::md5sum('final_project/app/model.rds')))
stopifnot(check$english_model_md5==check$production_model_md5)
write_json(check,file.path(out,'checks.json'),auto_unbox=TRUE,pretty=TRUE,digits=10)
print(check[c('languages','english_exact_600','unicode_hindi_korean_chinese','composition_pause','prepared_model_objects_maximum_mib')])
