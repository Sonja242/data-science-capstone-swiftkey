library(data.table);library(jsonlite);library(shiny)
out<-'final_project/research/multilingual-20260926';app<-'final_project/phraseflow_multilingual'
source(file.path(app,'predictor.R'))
stopifnot(normalize_phrase('f\u00fcr sch\u00f6ne Gr\u00fc\u00dfe','unicode')=='f\u00fcr sch\u00f6ne gr\u00fc\u00dfe',
 normalize_phrase('Hyv\u00e4\u00e4 p\u00e4iv\u00e4\u00e4','unicode')=='hyv\u00e4\u00e4 p\u00e4iv\u00e4\u00e4',normalize_phrase('\u042f \u043b\u044e\u0431\u043b\u044e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u044f\u0437\u044b\u043a','unicode')=='\u044f \u043b\u044e\u0431\u043b\u044e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u044f\u0437\u044b\u043a',
 normalize_phrase('e\u0301cole','unicode')==normalize_phrase('\u00e9cole','unicode'))
baseline<-new.env();sys.source('final_project/app/predictor.R',envir=baseline)
en<-prepare_predictor(readRDS(file.path(app,'model.rds')))
old<-baseline$prepare_predictor(readRDS('final_project/app/model.rds'))
dev<-fread('data/student_context/development.csv')
for(phrase in dev$prefix){a<-predict_word(en,phrase,10L);b<-baseline$predict_word(old,phrase,10L);stopifnot(identical(a$words,b$words),identical(a$scores,b$scores))}
rm(en,old);gc(FALSE)
app_env<-new.env()
local({previous<-setwd(app);on.exit(setwd(previous));sys.source('app.R',envir=app_env)})
testServer(app_env$server,{
 session$setInputs(language='en_US',phrase='',automatic=TRUE);session$flushReact()
 for(code in names(app_env$language_labels)){
  session$setInputs(language=code);session$flushReact();stopifnot(is.null(result()))
  session$setInputs(phrase='');session$elapse(200)
  phrase<-paste0(app_env$language_examples[[code]][1],' ')
  session$setInputs(phrase=phrase);session$elapse(200)
  stopifnot(length(result()$words)==10L,identical(result()$language,code),identical(last_language(),code))
  reference<-predict_word(app_env$get_model(code),phrase,10L)
  stopifnot(identical(result()$words,reference$words))
  session$setInputs(phrase=paste0(phrase,'x'));session$elapse(200);stopifnot(is.null(result()))
  session$setInputs(predict=sample.int(100000,1));stopifnot(length(result()$words)==10L)
  session$setInputs(phrase=paste(rep('word',110),collapse=' '),predict=sample.int(100000,1));stopifnot(grepl('500',error()))
  session$setInputs(phrase='');session$elapse(200)
 }
 session$setInputs(language='en_US',phrase='I am looking forward to ');session$elapse(200)
 session$setInputs(phrase='I am looking forward to meet',language='ru_RU');session$elapse(200)
 stopifnot(is.null(result()))
})
mem<-sum(vapply(ls(app_env$model_cache),function(code)as.numeric(object.size(app_env$get_model(code))),numeric(1)))/1024^2
stopifnot(mem<400)
record<-list(english_exact_600=TRUE,english_reserved_test_opened=FALSE,unicode_accents_and_cyrillic=TRUE,
 automatic_and_manual_all_four_languages=TRUE,language_switch_invalidates_old_predictions=TRUE,
 partial_word_pause=TRUE,character_limit=TRUE,combined_prepared_model_objects_mib=mem,
 english_model_md5=unname(tools::md5sum(file.path(app,'model.rds'))),production_model_md5=unname(tools::md5sum('final_project/app/model.rds')))
stopifnot(record$english_model_md5==record$production_model_md5)
write_json(record,file.path(out,'checks.json'),auto_unbox=TRUE,pretty=TRUE,digits=10)
print(record)
