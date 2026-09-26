# Sonja PhraseFlow | Author: Sonja Sahebzad | Sonja Projects
library(shiny)
source('predictor.R',local=TRUE)
`%||%`<-function(a,b)if(is.null(a))b else a
language_labels<-c(en_US='English',de_DE='Deutsch',fi_FI='Suomi',ru_RU='\u0420\u0443\u0441\u0441\u043a\u0438\u0439')
language_examples<-list(en_US=c('The weather today is','I am looking forward to','She put the book on'),
 de_DE=c('Ich freue mich auf','Das Wetter ist heute','Vielen Dank f\u00fcr'),
 fi_FI=c('Min\u00e4 haluan menn\u00e4','T\u00e4n\u00e4\u00e4n on kaunis','Kiitos paljon'),
 ru_RU=c('\u042f \u0445\u043e\u0447\u0443 \u043f\u043e\u0439\u0442\u0438 \u0432','\u0421\u0435\u0433\u043e\u0434\u043d\u044f \u0445\u043e\u0440\u043e\u0448\u0430\u044f','\u0421\u043f\u0430\u0441\u0438\u0431\u043e \u0431\u043e\u043b\u044c\u0448\u043e\u0435 \u0437\u0430'))
metrics<-jsonlite::fromJSON('metrics.json');runtime<-jsonlite::fromJSON('runtime-metrics.json')
language_metrics<-lapply(names(language_labels)[-1],function(code)jsonlite::fromJSON(file.path('languages',paste0(code,'_metrics.json'))))
names(language_metrics)<-names(language_labels)[-1]
model_cache<-new.env(parent=emptyenv())
app_dir<-getwd()
get_model<-function(code){
 if(!code %in% names(language_labels))stop('Unsupported language')
 if(!exists(code,envir=model_cache,inherits=FALSE)){
  path<-if(code=='en_US')'model.rds' else file.path('languages',paste0(code,'.rds'))
  assign(code,prepare_predictor(readRDS(file.path(app_dir,path))),envir=model_cache)
 }
 get(code,envir=model_cache,inherits=FALSE)
}
invisible(get_model('en_US'))
pct<-function(x)sprintf('%.1f%%',100*x)
ui<-fluidPage(title='Sonja PhraseFlow',
 tags$head(tags$link(rel='stylesheet',type='text/css',href='styles.css?v=1.2'),tags$meta(name='viewport',content='width=device-width, initial-scale=1'),tags$script(src='input.js?v=1.2')),
 tags$div(class='shell',
  tags$header(class='masthead',tags$a(class='brand',href='#','SONJA PROJECTS'),tags$span(class='edition','DATA SCIENCE CAPSTONE')),
  tags$div(class='hero',tags$p(class='eyebrow','SONJA PHRASEFLOW'),tags$h1('Your next word, in your language.'),
   tags$p(class='intro','Choose English, Deutsch, Suomi or \u0420\u0443\u0441\u0441\u043a\u0438\u0439. Select a suggestion and keep writing.')),
  tabsetPanel(id='view',type='tabs',
   tabPanel('Write',value='write',
    tags$div(class='writing-grid',
     tags$section(class='editor-panel',tags$h2('Start with a phrase'),
      selectInput('language','Prediction language',choices=setNames(names(language_labels),language_labels),selected='en_US',selectize=FALSE,width='100%'),
      tags$p(class='fine-print','Changing language clears the phrase. Other languages can be added after training and validation.'),
      textAreaInput('phrase','Your phrase',value='',rows=4,placeholder='The weather today is',width='100%'),
      tags$div(class='input-meta',tags$span('Up to 500 characters. Add a space after a complete word.'),textOutput('counter',inline=TRUE)),
      checkboxInput('automatic','Update suggestions while writing',TRUE),
      tags$div(class='actions',actionButton('predict','Predict next word',class='btn-primary'),actionButton('clear','Clear',class='btn-quiet')),
      tags$p(class='keyboard-hint','Automatic updates follow a space or punctuation and a short pause. You can also use Ctrl + Enter (Command + Enter on Mac).'),
      tags$p(class='example-label','Try a starting point'),
      tags$div(class='examples',actionButton('example1',language_examples$en_US[1]),actionButton('example2',language_examples$en_US[2]),actionButton('example3',language_examples$en_US[3])),
      tags$p(class='privacy-note','This app processes text on the Shiny server for your session. It does not save your phrases or send them to an external AI service.')),
     tags$section(class='prediction-panel',`aria-live`='polite',`aria-atomic`='true',uiOutput('prediction'))),
    tags$div(class='under-note',tags$strong('You decide what fits. '),'Suggestions reflect learned word patterns. The new language models are experimental and use at most four recent words.')),
   tabPanel('Results',value='results',tags$section(class='content-panel',tags$p(class='eyebrow','EVIDENCE FOR THE SELECTED LANGUAGE'),
    uiOutput('quality'),tags$h3('Performance by source'),plotOutput('source_plot',height='250px'),tableOutput('source_table'),
    tags$p(class='fine-print','A 95% interval describes sampling uncertainty. It is not confidence in an individual suggestion. Different languages use different samples and training sizes, so these scores are not a language league table.'),
    tags$p('English retains its previously evaluated model. The three new models use fixed settings and held-out sentences that were excluded from training. No quiz answers enter training. Individual confidence has not been calibrated.'))),
   tabPanel('How to use',value='guide',tags$section(class='content-panel guide',tags$h2('Choose a language, then keep writing'),
    tags$ol(tags$li('Select the language in which you will write. The app does not translate existing text.'),
     tags$li('Enter a phrase. Add a space after a complete word, or select Predict next word.'),
     tags$li('Click a suggestion to append it. More suggestions opens ranks 4 through 10. Continue with your own words whenever they fit better.')),
    tags$h3('Four supported languages'),tags$p('English uses the unchanged validated model. German, Finnish and Russian use separately trained experimental models from the official course corpus. This release does not support every language.'),
    tags$p('The interface uses English labels. Your selected prediction language determines the model, examples and Results table. Accents and Cyrillic letters are retained in the new models.'),
    tags$h3('How prediction works'),tags$p('Each model blends word patterns up to five words long using Kneser-Ney smoothing. Shorter patterns provide a fallback when a longer pattern is unfamiliar. Predictions use at most the last four normalized words, even when a complete sentence is entered.'),
    tags$p('Models are loaded once per server process when first needed. The first request in a new language can take longer. The model does not learn from text entered during a session. Choosing a different language does not translate the text or improve the English model.'),
    tags$h3('Documentation'),tags$p(tags$a(href='Multilingual_Guide.html',target='_blank','Read the multilingual guide and evaluation')),
    tags$p(class='fine-print','Author: Sonja Sahebzad. Sonja Projects. Multilingual preview 1.2. 26 September 2026.')))
  ),tags$footer(tags$span('Sonja PhraseFlow'),tags$span('Author: Sonja Sahebzad'),tags$span('Sonja Projects \u00b7 Preview 1.2 \u00b7 2026'))))

server<-function(input,output,session){
 result<-reactiveVal(NULL);error<-reactiveVal(NULL);last_phrase<-reactiveVal('');last_language<-reactiveVal('')
 active_language<-reactiveVal('en_US')
 output$counter<-renderText(sprintf('%d / 500',nchar(input$phrase %||% '')))
 observeEvent(input$language,{
  code<-input$language
  if(!code %in% names(language_labels))return()
  if(identical(code,active_language()))return()
  active_language(code);result(NULL);error(NULL);last_phrase('');last_language('')
  updateTextAreaInput(session,'phrase',value='',placeholder=language_examples[[code]][1])
  for(j in 1:3)updateActionButton(session,paste0('example',j),label=language_examples[[code]][j])
 },priority=30)
 observeEvent(input$phrase,{result(NULL);error(NULL)},ignoreInit=TRUE,priority=10)
 settled<-debounce(reactive(list(phrase=input$phrase %||% '',language=input$language %||% 'en_US')),150)
 predict_for<-function(phrase,explicit=FALSE){
  code<-input$language %||% 'en_US';if(!code %in% names(language_labels))return()
  if(nchar(phrase)>500L){result(NULL);error('Please shorten the phrase to 500 characters or fewer.');return()}
  tokenizer<-if(code=='en_US')'english' else 'unicode'
  if(!nzchar(normalize_phrase(phrase,tokenizer))){result(NULL);error(if(explicit)'Enter a word or phrase in the selected language.' else NULL);return()}
  if(identical(phrase,last_phrase())&&identical(code,last_language())&&!is.null(result()))return()
  start<-as.numeric(Sys.time());model<-get_model(code);p<-predict_word(model,phrase,10L)
  p$elapsed<-1000*(as.numeric(Sys.time())-start);p$language<-code
  last_phrase(phrase);last_language(code);error(NULL);result(p)
 }
 observeEvent(input$predict,{predict_for(input$phrase %||% '',TRUE)})
 observeEvent(list(settled(),input$automatic),{
  s<-settled();if(!isTRUE(input$automatic)||!identical(s$phrase,input$phrase %||% '')||!identical(s$language,input$language %||% 'en_US'))return()
  if(nchar(s$phrase)>500L||grepl('[[:space:].!?;:,]$',s$phrase))predict_for(s$phrase)
 })
 observeEvent(input$clear,{updateTextAreaInput(session,'phrase',value='');result(NULL);error(NULL)})
 for(i in 1:3)local({j<-i;observeEvent(input[[paste0('example',j)]],{updateTextAreaInput(session,'phrase',value=paste0(language_examples[[input$language %||% 'en_US']][j],' '));result(NULL);error(NULL)})})
 for(i in 1:10)local({j<-i;observeEvent(input[[paste0('choose',j)]],{
  p<-result();req(p,!is.null(p$words[j]),identical(input$phrase,last_phrase()),identical(input$language %||% 'en_US',last_language()))
  value<-paste0(trimws(last_phrase()),' ',p$words[j],' ')
  if(nchar(value)>500L){error('Adding this word would exceed 500 characters.');return()}
  updateTextAreaInput(session,'phrase',value=value);result(NULL);error(NULL)
 })})
 output$prediction<-renderUI({p<-result()
  if(!is.null(error()))return(tagList(tags$h2('Ready when you are.'),tags$p(class='input-error',error())))
  if(is.null(p))return(tagList(tags$p(class='eyebrow','YOUR NEXT WORD'),tags$div(class='empty-mark','Aa'),tags$h2('A little help with what comes next.'),tags$p('Choose a language, then finish a word and add a space, or select Predict next word.')))
  tags$div(`data-prediction-phrase`=last_phrase(),`data-prediction-language`=p$language,
   tags$p(class='eyebrow',paste('FIRST SUGGESTION',language_labels[[p$language]],sep=' \u00b7 ')),
   actionButton('choose1',p$words[1],class='word-primary',title='Add this word to your phrase'),tags$p(class='choose-hint','Select a word to add it to your phrase.'),
   tags$p(class='example-label','Two alternatives'),tags$div(class='alternatives',actionButton('choose2',p$words[2]),actionButton('choose3',p$words[3])),
   tags$details(class='more-words',tags$summary('More suggestions'),tags$p(class='fine-print','Ranks 4 to 10 from the selected language model.'),tags$ol(start='4',lapply(4:10,function(j)tags$li(actionButton(paste0('choose',j),p$words[j]))))),
   tags$div(class='evidence',tags$strong(if(p$order==1L)'Frequent-word fallback' else sprintf('Pattern with %d recent %s',p$context_words,if(p$context_words==1L)'word' else 'words')),
    tags$p(if(p$order==1L)'This context was not retained in the compact model.' else paste('Context:',p$context)),
    tags$p(class='context-note',sprintf('%d input words. At most 4 recent words guide prediction.',p$input_words)),
    tags$small(sprintf('%.0f ms on this server, including a first model load if needed. Network time excluded.',p$elapsed))))
 })
 chosen_metrics<-reactive({code<-input$language %||% 'en_US';if(code=='en_US')metrics else language_metrics[[code]]})
 output$quality<-renderUI({code<-input$language %||% 'en_US';d<-chosen_metrics();m<-d$overall
  tagList(tags$h2(paste('Measured results:',language_labels[[code]])),
   tags$p(if(code=='en_US')'Archived independent evaluation: 900 English examples, 300 per text source. The English model is unchanged.' else 'Experimental model. Initial held-out evaluation: 600 examples, 200 per text source. Fixed settings were chosen before this evaluation.'),
   tags$div(class='metric-grid',
    tags$div(class='metric',tags$span('First suggestion correct'),tags$strong(pct(m$top1)),tags$small(sprintf('%d / %d; 95%% interval %s to %s',m$top1_count,m$cases,pct(m$top1_low),pct(m$top1_high)))),
    tags$div(class='metric',tags$span('Correct within three'),tags$strong(pct(m$top3)),tags$small(sprintf('%d / %d; 95%% interval %s to %s',m$top3_count,m$cases,pct(m$top3_low),pct(m$top3_high)))),
    tags$div(class='metric',tags$span(if(code=='en_US')'Archived three-word timing' else 'Ten-word computation, median'),tags$strong(sprintf('%.2f ms',if(code=='en_US')runtime$median_ms else d$median_ms)),
     tags$small(if(code=='en_US')'Earlier 3,000-call benchmark; not a benchmark of this multilingual preview.' else sprintf('95th percentile %.2f ms; 1,200 warm local calls.',d$p95_ms)))),
   if(code!='en_US')tags$p(sprintf('Reference using only common words: %.1f%% first-word accuracy and %.1f%% within three. Training: %s lines. Vocabulary: %s words. Model: %.1f MiB on disk.',100*m$unigram_top1,100*m$unigram_top3,format(d$training_lines,big.mark=','),format(d$vocabulary,big.mark=','),d$model_mib)),
   tags$p('Computation times exclude network, display and the 150 ms typing pause. Extra language support does not establish higher English accuracy.'))
 })
 output$source_plot<-renderPlot({d<-chosen_metrics()$by_source
  par(mar=c(4,4,1,1),family='sans',fg='#173b59',col.axis='#173b59')
  bp<-barplot(rbind(100*d$top1,100*d$top3),beside=TRUE,names.arg=d$source,col=c('#173f66','#75aacf'),border=NA,ylim=c(0,max(d$top3_high*100)+10),ylab='Correct predictions (%)',legend.text=c('First suggestion','Within three'),args.legend=list(x='topleft',bty='n',horiz=TRUE,cex=.85))
  arrows(bp[1,],100*d$top1_low,bp[1,],100*d$top1_high,angle=90,code=3,length=.04);arrows(bp[2,],100*d$top3_low,bp[2,],100*d$top3_high,angle=90,code=3,length=.04)
 },res=110,alt='Accuracy by source for the selected language. Exact values appear in the following table.')
 output$source_table<-renderTable({d<-chosen_metrics()$by_source;data.frame(Source=d$source,Cases=d$cases,First=pct(d$top1),`Within three`=pct(d$top3),check.names=FALSE)},striped=TRUE,spacing='s')
}
shinyApp(ui,server)
