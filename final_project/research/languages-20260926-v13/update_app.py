from pathlib import Path
import json,shutil
ROOT=Path.cwd();APP=ROOT/'final_project/phraseflow_multilingual';OUT=ROOT/'final_project/research/languages-20260926-v13'
extras=json.loads((APP/'extension-languages.json').read_text(encoding='utf-8'))
registry={
 'en_US':dict(name='English',label='English',tokenizer='english',examples=['The weather today is','I am looking forward to','She put the book on']),
 'de_DE':dict(name='German',label='Deutsch',tokenizer='unicode',examples=['Ich freue mich auf','Das Wetter ist heute','Vielen Dank für']),
 'fi_FI':dict(name='Finnish',label='Suomi',tokenizer='unicode',examples=['Minä haluan mennä','Tänään on kaunis','Kiitos paljon']),
 'ru_RU':dict(name='Russian',label='Русский',tokenizer='unicode',examples=['Я хочу пойти в','Сегодня хорошая','Спасибо большое за']),**extras}
for code,d in registry.items():
 d['corpus']='Original course corpus' if code not in extras else 'Tatoeba example sentences'
 d['experimental']=code!='en_US'
(APP/'language-config.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
p=APP/'app.R';s=p.read_text(encoding='utf-8');start=s.index('language_labels<-');end=s.index('metrics<-',start)
s=s[:start]+'''language_config<-jsonlite::fromJSON('language-config.json',simplifyVector=FALSE)
language_labels<-vapply(language_config,function(x)x$label,character(1))
language_examples<-lapply(language_config,function(x)unlist(x$examples))
language_tokenizers<-vapply(language_config,function(x)x$tokenizer,character(1))
'''+s[end:]
s=s.replace('model_cache<-new.env(parent=emptyenv())','model_cache<-new.env(parent=emptyenv())\ncache_order<-character()')
s=s.replace(" get(code,envir=model_cache,inherits=FALSE)",""" cache_order<<-c(setdiff(cache_order,code),code)
 while(length(cache_order)>4L){
  discard<-setdiff(cache_order,c('en_US',code))[1L]
  rm(list=discard,envir=model_cache);cache_order<<-setdiff(cache_order,discard)
 }
 get(code,envir=model_cache,inherits=FALSE)""")
s=s.replace('?v=1.2','?v=1.3')
start=s.index("   tags$p(class='intro',");end=s.index("  tabsetPanel",start)
s=s[:start]+'''   tags$p(class='intro','Twelve language choices. Select a suggestion and keep writing.'),
   tags$p(class='hero-author','Sonja Sahebzad',tags$br(),tags$span('Utrecht, the Netherlands')),
   tags$a(class='btn btn-guide',href='Multilingual_Guide.html',target='_blank','User guide / Handleiding')),
'''+s[end:]
s=s.replace("tags$span('Up to 500 characters. Add a space after a complete word.')","textOutput('input_hint',inline=TRUE)")
s=s.replace("tags$p(class='keyboard-hint','Automatic updates follow a space or punctuation and a short pause. You can also use Ctrl + Enter (Command + Enter on Mac).')","tags$p(class='keyboard-hint','Finish a word, then pause. Chinese updates after committed text without requiring a space. Ctrl + Enter (Command + Enter on Mac) also predicts.')")
s=s.replace('The three new models use fixed settings and held-out sentences that were excluded from training.','The additional models use fixed settings and held-out sentences excluded from training. Their corpora and prediction units differ.')
s=s.replace("tags$h3('Four supported languages'),tags$p('English uses the unchanged validated model. German, Finnish and Russian use separately trained experimental models from the official course corpus. This release does not support every language.')", """tags$h3('Twelve language choices'),tags$p('English, Dutch, French, German, Finnish, Russian, Korean, Mandarin Chinese, Portuguese, Italian, Hindi and Spanish. English is unchanged; the other languages are experimental.'),
    tags$p('German, Finnish and Russian use the course corpus. The eight latest languages use Tatoeba example sentences. Their results do not measure performance on everyday messages or news.'),
    tags$p('Mandarin Chinese uses dictionary-based word segmentation and appends suggestions without spaces. Korean predicts a space-delimited unit, which may include grammatical endings. These units are not identical across languages.')""")
s=s.replace("tags$li('Enter a phrase. Add a space after a complete word, or select Predict next word.')","tags$li('Enter a phrase. Add a space after a complete word, or select Predict next word. Chinese does not require spaces.')")
s=s.replace('Models are loaded once per server process when first needed.','Models load when first needed. A bounded cache retains English and up to three additional models, so revisiting an evicted language reloads it.')
s=s.replace("tags$h3('Documentation'),tags$p(tags$a(href='Multilingual_Guide.html',target='_blank','Read the multilingual guide and evaluation'))", """tags$h3('Documentation'),tags$p(tags$a(class='btn btn-guide',href='Multilingual_Guide.html',target='_blank','Open the illustrated user guide')),
    tags$p(tags$a(href='Third_Party_Notices.html',target='_blank','Training-data sources, licences and attribution'))""")
s=s.replace('Multilingual preview 1.2. 26 September 2026.','Utrecht, the Netherlands. Multilingual preview 1.3. 26 September 2026.')
s=s.replace("tags$span('Author: Sonja Sahebzad')","tags$span('Author: Sonja Sahebzad',tags$br(),'Utrecht, the Netherlands')")
s=s.replace('Preview 1.2','Preview 1.3')
s=s.replace(" output$counter<-renderText", """ output$input_hint<-renderText(if(identical(input$language,'zh_CN'))'Up to 500 characters. Chinese does not require spaces.' else 'Up to 500 characters. Add a space after a complete word.')
 observeEvent(input$composing,{if(isTRUE(input$composing)){result(NULL);error(NULL)}},priority=25)
 output$counter<-renderText""")
s=s.replace("  code<-input$language %||% 'en_US';if(!code %in% names(language_labels))return()", "  if(isTRUE(input$composing))return()\n  code<-input$language %||% 'en_US';if(!code %in% names(language_labels))return()")
s=s.replace("tokenizer<-if(code=='en_US')'english' else 'unicode'","tokenizer<-language_tokenizers[[code]]")
s=s.replace("observeEvent(list(settled(),input$automatic),{", "observeEvent(list(settled(),input$automatic,input$composing),{")
s=s.replace("!isTRUE(input$automatic)||", "!isTRUE(input$automatic)||isTRUE(input$composing)||")
s=s.replace("if(nchar(s$phrase)>500L||grepl('[[:space:].!?;:,]$',s$phrase))", "if(nchar(s$phrase)>500L||identical(s$language,'zh_CN')||grepl('[[:space:].!?;:,]$',s$phrase))")
s=s.replace("  value<-paste0(trimws(last_phrase()),' ',p$words[j],' ')", "  gap<-if(identical(p$language,'zh_CN'))'' else ' '\n  value<-paste0(trimws(last_phrase()),gap,p$words[j],gap)")
s=s.replace("value=paste0(language_examples[[input$language %||% 'en_US']][j],' ')","value=paste0(language_examples[[input$language %||% 'en_US']][j],if(identical(input$language,'zh_CN'))'' else ' ')")
s=s.replace("else 'Experimental model. Initial held-out evaluation: 600 examples, 200 per text source. Fixed settings were chosen before this evaluation.'", "else sprintf('Experimental model. Initial held-out evaluation: %d cases from %s. Fixed settings were chosen before evaluation.',m$cases,language_config[[code]]$corpus)")
s=s.replace("   tags$p('Computation times exclude", """   if(code %in% c('ko_KR','zh_CN'))tags$p(class='fine-print',if(code=='zh_CN')'The target is one ICU-segmented Chinese word. It is not an independently annotated word-boundary benchmark.' else 'The prediction unit is a Korean eojeol: a space-delimited unit that can include grammatical endings.'),
   tags$p('Computation times exclude""")
# Keep R source portable even under Windows C locale.
s=''.join(c if ord(c)<128 else '\\u%04x'%ord(c) for c in s)
p.write_text(s,encoding='ascii');(OUT/'app.R').write_text(s,encoding='ascii')
js=APP/'www/input.js';t=js.read_text(encoding='utf-8')
t=t.replace("  function phraseBox()", "  let composing = false;\n  function phraseBox()")
t=t.replace("if (box && window.Shiny)","if (box && window.Shiny && !composing)")
t=t.replace("  document.addEventListener('input',", """  document.addEventListener('compositionstart', function (event) {
    if (event.target.id !== 'phrase') return;
    composing = true;
    if (window.Shiny) Shiny.setInputValue('composing', true, {priority:'event'});
    guardSuggestions();
  }, true);
  document.addEventListener('compositionend', function (event) {
    if (event.target.id !== 'phrase') return;
    composing = false;
    if (window.Shiny) Shiny.setInputValue('composing', false, {priority:'event'});
    guardSuggestions(); sendCurrentText();
  }, true);
  document.addEventListener('input',""")
t=t.replace("    if (phraseBox()) phraseBox().value = '';", "    composing = false;\n    if (window.Shiny) Shiny.setInputValue('composing', false, {priority:'event'});\n    if (phraseBox()) phraseBox().value = '';")
js.write_text(t,encoding='utf-8')
css=APP/'www/styles.css';t=css.read_text(encoding='utf-8')
t+='\n.hero-author{font-size:14px;color:var(--muted);margin:14px 0 10px}.hero-author span{font-size:12px}.btn-guide{font-weight:600;font-size:14px;border-color:#94b8d1;background:#e9f3fa}.hero .btn-guide{margin:0 0 2px}\n'
css.write_text(t,encoding='utf-8')
print('Twelve language configurations, guide buttons, author location and input handling installed.')
