# Sonja PhraseFlow | Author: Sonja Sahebzad | Sonja Projects
library(shiny)
source('predictor.R',local=TRUE)
source('continuations.R',local=TRUE)
`%||%`<-function(a,b)if(is.null(a))b else a
language_config<-jsonlite::fromJSON('language-config.json',simplifyVector=FALSE)
language_labels<-vapply(language_config,function(x)x$label,character(1))
language_examples<-lapply(language_config,function(x)unlist(x$examples))
language_tokenizers<-vapply(language_config,function(x)x$tokenizer,character(1))
metrics<-jsonlite::fromJSON('metrics.json');runtime<-jsonlite::fromJSON('runtime-metrics.json')
language_metrics<-lapply(names(language_labels)[-1],function(code)jsonlite::fromJSON(file.path('languages',paste0(code,'_metrics.json'))))
names(language_metrics)<-names(language_labels)[-1]
model_cache<-new.env(parent=emptyenv())
cache_order<-character()
app_dir<-getwd()
get_model<-function(code){
 if(!code %in% names(language_labels))stop('Unsupported language')
 if(!exists(code,envir=model_cache,inherits=FALSE)){
  path<-if(code=='en_US')'model.rds' else file.path('languages',paste0(code,'.rds'))
  model<-prepare_predictor(readRDS(file.path(app_dir,path)))
  model$completion_index<-readRDS(file.path(app_dir,'continuations',paste0(code,'.rds')))
  assign(code,model,envir=model_cache)
 }
 cache_order<<-c(setdiff(cache_order,code),code)
 while(length(cache_order)>4L){
  discard<-setdiff(cache_order,c('en_US',code))[1L]
  rm(list=discard,envir=model_cache);cache_order<<-setdiff(cache_order,discard)
 }
 get(code,envir=model_cache,inherits=FALSE)
}
invisible(get_model('en_US'))
pct<-function(x)sprintf('%.1f%%',100*x)
ui<-fluidPage(title='Sonja PhraseFlow',
 tags$head(tags$link(rel='stylesheet',type='text/css',href='styles.css?v=1.4.1'),tags$meta(name='viewport',content='width=device-width, initial-scale=1'),tags$script(src='input.js?v=1.4.1')),
 tags$div(class='shell',
  tags$header(class='masthead',tags$a(class='brand',href='#','SONJA PROJECTS'),tags$span(class='edition','DATA SCIENCE CAPSTONE')),
  tags$div(class='hero',tags$p(class='eyebrow','SONJA PHRASEFLOW'),tags$h1('Your next word, in your language.'),
   tags$p(class='intro','Twelve language choices. Select a suggestion and keep writing.'),
   tags$p(class='hero-author','Sonja Sahebzad',tags$br(),tags$span('Utrecht, the Netherlands')),
   tags$a(class='btn btn-guide',href='Multilingual_Guide.html',target='_blank','User guide / Handleiding')),
  tabsetPanel(id='view',type='tabs',
   tabPanel('Write',value='write',
    tags$div(class='writing-grid',
     tags$section(class='editor-panel',tags$h2('Start with a phrase'),
      selectInput('language','Prediction language',choices=setNames(names(language_labels),language_labels),selected='en_US',selectize=FALSE,width='100%'),
      tags$p(class='fine-print','Changing language clears the phrase. Other languages can be added after training and validation.'),
      textAreaInput('phrase','Your phrase',value='',rows=4,placeholder='The weather today is',width='100%'),
      tags$div(class='input-meta',textOutput('input_hint',inline=TRUE),textOutput('counter',inline=TRUE)),
      checkboxInput('automatic','Update suggestions while writing',TRUE),
      selectInput('suggestion_count','Word choices',choices=c('5 suggestions'=5,'10 suggestions'=10,'20 suggestions'=20),selected=10,selectize=FALSE,width='100%'),
      tags$div(class='actions',actionButton('predict','Predict next word',class='btn-primary'),actionButton('clear','Clear',class='btn-quiet')),
      tags$p(class='keyboard-hint','Finish a word, then pause. Chinese updates after committed text without requiring a space. Ctrl + Enter (Command + Enter on Mac) also predicts.'),
      tags$p(class='example-label','Try a starting point'),
      tags$div(class='examples',actionButton('example1',language_examples$en_US[1]),actionButton('example2',language_examples$en_US[2]),actionButton('example3',language_examples$en_US[3])),
      tags$p(class='privacy-note','This app processes text on the Shiny server for your session. It does not save your phrases or send them to an external AI service.')),
     tags$section(class='prediction-panel',`aria-live`='polite',`aria-atomic`='true',uiOutput('prediction'))),
    tags$div(class='under-note',tags$strong('You decide what fits. '),'Suggestions reflect learned word patterns. The new language models are experimental and use at most four recent words.')),
   tabPanel('Results',value='results',tags$section(class='content-panel',tags$p(class='eyebrow','EVIDENCE FOR THE SELECTED LANGUAGE'),
    uiOutput('quality'),tags$h3('Performance by source'),plotOutput('source_plot',height='250px'),tableOutput('source_table'),
    tags$p(class='fine-print','A 95% interval describes sampling uncertainty. It is not confidence in an individual suggestion. Different languages use different samples and training sizes, so these scores are not a language league table.'),
    tags$p('English retains its previously evaluated model. The additional models use fixed settings and held-out sentences excluded from training. Their corpora and prediction units differ. No quiz answers enter training. Individual confidence has not been calibrated.'))),
   tabPanel('How to use',value='guide',tags$section(class='content-panel guide',tags$h2('Choose a language, then keep writing'),
    tags$ol(tags$li('Select the language in which you will write. The app does not translate existing text.'),
     tags$li('Enter a phrase. Add a space after a complete word, or select Predict next word. Chinese does not require spaces.'),
     tags$li('Click a suggestion to append it. Choose 5, 10 or 20 visible word suggestions. Short continuations can add up to three words at once. Continue with your own words whenever they fit better.')),
    tags$h3('Twelve language choices'),tags$p('English, Dutch, French, German, Finnish, Russian, Korean, Mandarin Chinese, Portuguese, Italian, Hindi and Spanish. English is unchanged; the other languages are experimental.'),
    tags$p('German, Finnish and Russian use the course corpus. The eight latest languages use Tatoeba example sentences. Their results do not measure performance on everyday messages or news.'),
    tags$p('Mandarin Chinese uses dictionary-based word segmentation and appends suggestions without spaces. Korean predicts a space-delimited unit, which may include grammatical endings. These units are not identical across languages.'),
    tags$p('The interface uses English labels. Your selected prediction language determines the model, examples and Results table. Accents and Cyrillic letters are retained in the new models.'),
    tags$h3('Short continuations'),tags$p('When a recent phrase matches stored training text, extra buttons offer an observed ending of up to three units. With only a two-unit match, just the next unit is offered. The matched context is shown. These experimental suggestions do not understand the meaning of the whole sentence and are not included in the next-word accuracy figures.'),
    tags$h3('How prediction works'),tags$p('Each model blends word patterns up to five words long using Kneser-Ney smoothing. Shorter patterns provide a fallback when a longer pattern is unfamiliar. Predictions use at most the last four normalized words, even when a complete sentence is entered.'),
    tags$p('Models load when first needed. A bounded cache retains English and up to three additional models, so revisiting an evicted language reloads it. The first request in a new language can take longer. The model does not learn from text entered during a session. Choosing a different language does not translate the text or improve the English model.'),
    tags$h3('Documentation'),tags$p(tags$a(class='btn btn-guide',href='Multilingual_Guide.html',target='_blank','Open the illustrated user guide')),
    tags$p(tags$a(href='Third_Party_Notices.html',target='_blank','Training-data sources, licences and attribution')),
    tags$p(class='fine-print','Author: Sonja Sahebzad. Sonja Projects. Utrecht, the Netherlands. Multilingual preview 1.4. 26 September 2026.')))
  ),tags$footer(tags$span('Sonja PhraseFlow'),tags$span('Author: Sonja Sahebzad',tags$br(),'Utrecht, the Netherlands'),tags$span('Sonja Projects \u00b7 Preview 1.4 \u00b7 2026'))))

server<-function(input,output,session){
 set_phrase<-function(value)session$sendCustomMessage('phraseflow-set-text',list(text=value,language=input$language %||% 'en_US'))
 result<-reactiveVal(NULL);error<-reactiveVal(NULL);last_phrase<-reactiveVal('');last_language<-reactiveVal('')
 active_language<-reactiveVal('en_US')
 awaiting_language_reset<-reactiveVal(FALSE);reset_phrase<-reactiveVal('')
 output$input_hint<-renderText(if(identical(input$language,'zh_CN'))'Up to 500 characters. Chinese does not require spaces.' else 'Up to 500 characters. Add a space after a complete word.')
 observeEvent(input$composing,{if(isTRUE(input$composing)){result(NULL);error(NULL)}},priority=25)
 output$counter<-renderText(sprintf('%d / 500',nchar(input$phrase %||% '')))
 observeEvent(input$language,{
  code<-input$language
  if(!code %in% names(language_labels))return()
  if(identical(code,active_language()))return()
  reset_phrase(input$phrase %||% '');awaiting_language_reset(FALSE)
  active_language(code);result(NULL);error(NULL);last_phrase('');last_language('')
  updateTextAreaInput(session,'phrase',placeholder=language_examples[[code]][1])
  for(j in 1:3)updateActionButton(session,paste0('example',j),label=language_examples[[code]][j])
 },priority=30)
 observeEvent(input$phrase,{
  if(isTRUE(awaiting_language_reset())&&!identical(input$phrase %||% '',reset_phrase()))awaiting_language_reset(FALSE)
  result(NULL);error(NULL)},ignoreInit=TRUE,priority=10)
 settled<-debounce(reactive(list(phrase=input$phrase %||% '',language=input$language %||% 'en_US')),150)
 predict_for<-function(phrase,explicit=FALSE){
  if(isTRUE(input$composing)||isTRUE(awaiting_language_reset()))return()
  code<-input$language %||% 'en_US';if(!code %in% names(language_labels))return()
  if(nchar(phrase)>500L){result(NULL);error('Please shorten the phrase to 500 characters or fewer.');return()}
  ctx<-input$phrase_context
  if(!is.null(ctx)&&(!identical(ctx$language,code)||!identical(ctx$text,phrase)))return()
  tokenizer<-language_tokenizers[[code]]
  if(!nzchar(normalize_phrase(phrase,tokenizer))){result(NULL);error(if(explicit)'Enter a word or phrase in the selected language.' else NULL);return()}
  if(identical(phrase,last_phrase())&&identical(code,last_language())&&!is.null(result()))return()
  start<-as.numeric(Sys.time());model<-get_model(code);p<-predict_word(model,phrase,20L)
  p$completions<-predict_continuations(model$completion_index,phrase,tokenizer,limit=8L)
  p$elapsed<-1000*(as.numeric(Sys.time())-start);p$language<-code
  last_phrase(phrase);last_language(code);error(NULL);result(p)
 }
 observeEvent(input$predict,{predict_for(input$phrase %||% '',TRUE)})
 observeEvent(list(settled(),input$automatic,input$composing),{
  s<-settled();if(!isTRUE(input$automatic)||isTRUE(input$composing)||!identical(s$phrase,input$phrase %||% '')||!identical(s$language,input$language %||% 'en_US'))return()
  if(nchar(s$phrase)>500L||identical(s$language,'zh_CN')||grepl('[[:space:].!?;:,]$',s$phrase))predict_for(s$phrase)
 })
 observeEvent(input$clear,{set_phrase('');result(NULL);error(NULL)})
 for(i in 1:3)local({j<-i;observeEvent(input[[paste0('example',j)]],{set_phrase(paste0(language_examples[[input$language %||% 'en_US']][j],if(identical(input$language,'zh_CN'))'' else ' '));result(NULL);error(NULL)})})
 for(i in 1:20)local({j<-i;observeEvent(input[[paste0('choose',j)]],{
  p<-result();req(p,!is.null(p$words[j]),identical(input$phrase,last_phrase()),identical(input$language %||% 'en_US',last_language()))
  gap<-if(identical(p$language,'zh_CN'))'' else ' '
  value<-paste0(trimws(last_phrase()),gap,p$words[j],gap)
  if(nchar(value)>500L){error('Adding this word would exceed 500 characters.');return()}
  set_phrase(value);result(NULL);error(NULL)
 })})
 for(i in 1:8)local({j<-i;observeEvent(input[[paste0('complete',j)]],{
  p<-result();req(p,identical(input$phrase,last_phrase()),identical(input$language %||% 'en_US',last_language()),!isTRUE(input$composing))
  choices<-p$completions$text
  req(j<=length(choices))
  gap<-if(identical(p$language,'zh_CN'))'' else ' '
  value<-paste0(trimws(last_phrase()),gap,choices[j],gap)
  if(nchar(value)>500L){error('Adding this continuation would exceed 500 characters.');return()}
  set_phrase(value);result(NULL);error(NULL)
 })})
 output$prediction<-renderUI({p<-result()
  if(!is.null(error()))return(tagList(tags$h2('Ready when you are.'),tags$p(class='input-error',error())))
  if(is.null(p))return(tagList(tags$p(class='eyebrow','YOUR NEXT WORD'),tags$div(class='empty-mark','Aa'),tags$h2('A little help with what comes next.'),tags$p('Choose a language, then finish a word and add a space, or select Predict next word.')))
  visible<-as.integer(input$suggestion_count %||% 10L);if(!visible %in% c(5L,10L,20L))visible<-10L
  alternatives<-seq.int(2L,min(visible,length(p$words)))
  continuation_ids<-head(which(!p$completions$text %in% head(p$words,visible)),3L)
  continuations<-p$completions$text[continuation_ids]
  unit<-if(p$language %in% c('zh_CN','ko_KR'))'unit' else 'word'
  tags$div(`data-prediction-phrase`=last_phrase(),`data-prediction-language`=p$language,
   tags$p(class='eyebrow',paste('FIRST SUGGESTION',language_labels[[p$language]],sep=' \u00b7 ')),
   actionButton('choose1',p$words[1],class='word-primary',title='Add this word to your phrase'),tags$p(class='choose-hint','Select a word to add it to your phrase.'),
   tags$p(class='example-label',sprintf('%d alternatives',length(alternatives))),
   tags$div(class='alternatives',lapply(alternatives,function(j)actionButton(paste0('choose',j),p$words[j],title=sprintf('Suggestion %d. Add this word.',j)))),
   if(length(continuations))tags$section(class='phrase-completions',tags$h3('Short continuations'),
    tags$p(class='fine-print','Choose an extra word or a short ending.'),
    tags$div(class='completion-options',lapply(seq_along(continuations),function(j)actionButton(paste0('complete',continuation_ids[j]),continuations[j],title='Append this continuation'))),
    tags$p(class='fine-print',paste('Matched phrase:',p$completions$context)),
    tags$p(class='fine-print','Experimental suggestions from training phrases. Choose what fits your sentence.')),

   tags$div(class='evidence',tags$strong(if(p$order==1L)'Frequent-word fallback' else sprintf('Pattern with %d recent %s',p$context_words,if(p$context_words==1L)unit else paste0(unit,'s'))),
    tags$p(if(p$order==1L)'This context was not retained in the compact model.' else paste('Context:',p$context)),
    tags$p(class='context-note',sprintf('%d input %s. At most 4 recent %s guide prediction.',p$input_words,if(p$input_words==1L)unit else paste0(unit,'s'),paste0(unit,'s'))),
    tags$small(sprintf('%.0f ms on this server, including a first model load if needed. Network time excluded.',p$elapsed))))
 })
 chosen_metrics<-reactive({code<-input$language %||% 'en_US';if(code=='en_US')metrics else language_metrics[[code]]})
 output$quality<-renderUI({code<-input$language %||% 'en_US';d<-chosen_metrics();m<-d$overall
  tagList(tags$h2(paste('Measured results:',language_labels[[code]])),
   tags$p(if(code=='en_US')'Archived independent evaluation: 900 English examples, 300 per text source. The English model is unchanged.' else sprintf('Experimental model. Initial held-out evaluation: %d cases from %s. Fixed settings were chosen before evaluation.',m$cases,language_config[[code]]$corpus)),
   tags$div(class='metric-grid',
    tags$div(class='metric',tags$span('First suggestion correct'),tags$strong(pct(m$top1)),tags$small(sprintf('%d / %d; 95%% interval %s to %s',m$top1_count,m$cases,pct(m$top1_low),pct(m$top1_high)))),
    tags$div(class='metric',tags$span('Correct within three'),tags$strong(pct(m$top3)),tags$small(sprintf('%d / %d; 95%% interval %s to %s',m$top3_count,m$cases,pct(m$top3_low),pct(m$top3_high)))),
    tags$div(class='metric',tags$span(if(code=='en_US')'Archived three-word timing' else 'Ten-word computation, median'),tags$strong(sprintf('%.2f ms',if(code=='en_US')runtime$median_ms else d$median_ms)),
     tags$small(if(code=='en_US')'Earlier 3,000-call benchmark; not a benchmark of this multilingual preview.' else sprintf('95th percentile %.2f ms; 1,200 warm local calls.',d$p95_ms)))),
   if(code!='en_US')tags$p(sprintf('Reference using only common words: %.1f%% first-word accuracy and %.1f%% within three. Training: %s lines. Vocabulary: %s words. Model: %.1f MiB on disk.',100*m$unigram_top1,100*m$unigram_top3,format(d$training_lines,big.mark=','),format(d$vocabulary,big.mark=','),d$model_mib)),
   if(code %in% c('ko_KR','zh_CN'))tags$p(class='fine-print',if(code=='zh_CN')'The target is one ICU-segmented Chinese word. It is not an independently annotated word-boundary benchmark.' else 'The prediction unit is a Korean eojeol: a space-delimited unit that can include grammatical endings.'),
   tags$p('Computation times exclude network, display and the 150 ms typing pause. These archived times cover ten word suggestions (three for English). The current screen can show twenty and retrieve short continuations, so its total time differs. Short continuations have not had an independent accuracy evaluation.'))
 })
 output$source_plot<-renderPlot({d<-chosen_metrics()$by_source
  par(mar=c(4,4,1,1),family='sans',fg='#173b59',col.axis='#173b59')
  bp<-barplot(rbind(100*d$top1,100*d$top3),beside=TRUE,names.arg=d$source,col=c('#173f66','#75aacf'),border=NA,ylim=c(0,max(d$top3_high*100)+10),ylab='Correct predictions (%)',legend.text=c('First suggestion','Within three'),args.legend=list(x='topleft',bty='n',horiz=TRUE,cex=.85))
  arrows(bp[1,],100*d$top1_low,bp[1,],100*d$top1_high,angle=90,code=3,length=.04);arrows(bp[2,],100*d$top3_low,bp[2,],100*d$top3_high,angle=90,code=3,length=.04)
 },res=110,alt='Accuracy by source for the selected language. Exact values appear in the following table.')
 output$source_table<-renderTable({d<-chosen_metrics()$by_source;data.frame(Source=d$source,Cases=d$cases,First=pct(d$top1),`Within three`=pct(d$top3),check.names=FALSE)},striped=TRUE,spacing='s')
}
shinyApp(ui,server)
