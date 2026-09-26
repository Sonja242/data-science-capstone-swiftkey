from pathlib import Path
import shutil
root=Path.cwd();app=root/'final_project/phraseflow_multilingual';out=root/'final_project/research/continuations-20260926-v14'
stage=Path(__file__).parent
shutil.copy2(stage/'continuations.R',app/'continuations.R')
p=app/'app.R';s=p.read_text(encoding='utf-8')
s=s.replace("source('predictor.R',local=TRUE)","source('predictor.R',local=TRUE)\nsource('continuations.R',local=TRUE)")
s=s.replace("assign(code,prepare_predictor(readRDS(file.path(app_dir,path))),envir=model_cache)","model<-prepare_predictor(readRDS(file.path(app_dir,path)))\n  model$completion_index<-readRDS(file.path(app_dir,'continuations',paste0(code,'.rds')))\n  assign(code,model,envir=model_cache)")
s=s.replace('v=1.3.1','v=1.4.0').replace('Preview 1.3','Preview 1.4').replace('preview 1.3','preview 1.4')
s=s.replace("checkboxInput('automatic','Update suggestions while writing',TRUE),","checkboxInput('automatic','Update suggestions while writing',TRUE),\n      selectInput('suggestion_count','Word choices',choices=c('5 suggestions'=5,'10 suggestions'=10,'20 suggestions'=20),selected=10,selectize=FALSE,width='100%'),")
s=s.replace('More suggestions opens ranks 4 through 10.','Choose 5, 10 or 20 visible word suggestions. Short continuations can add up to three words at once.')
s=s.replace("tags$h3('How prediction works'),", "tags$h3('Short continuations'),tags$p('When a recent phrase matches stored training text, extra buttons offer an observed ending of up to three units. With only a two-unit match, just the next unit is offered. The matched context is shown. These experimental suggestions do not understand the meaning of the whole sentence and are not included in the next-word accuracy figures.'),\n    tags$h3('How prediction works'),")
s=s.replace("p<-predict_word(model,phrase,10L)","p<-predict_word(model,phrase,20L)\n  p$completions<-predict_continuations(model$completion_index,phrase,tokenizer,limit=8L)")
s=s.replace('for(i in 1:10)local','for(i in 1:20)local')
marker=" output$prediction<-renderUI({p<-result()"
assert marker in s
s=s.replace(marker,""" for(i in 1:3)local({j<-i;observeEvent(input[[paste0('complete',j)]],{
  p<-result();req(p,identical(input$phrase,last_phrase()),identical(input$language %||% 'en_US',last_language()),!isTRUE(input$composing))
  visible<-as.integer(input$suggestion_count %||% 10L)
  choices<-head(setdiff(p$completions$text,head(p$words,visible)),3L)
  req(j<=length(choices))
  gap<-if(identical(p$language,'zh_CN'))'' else ' '
  value<-paste0(trimws(last_phrase()),gap,choices[j],gap)
  if(nchar(value)>500L){error('Adding this continuation would exceed 500 characters.');return()}
  updateTextAreaInput(session,'phrase',value=value);result(NULL);error(NULL)
 })})
 output$prediction<-renderUI({p<-result()""")
s=s.replace("  unit<-if(p$language", "  visible<-as.integer(input$suggestion_count %||% 10L);if(!visible %in% c(5L,10L,20L))visible<-10L\n  alternatives<-seq.int(2L,min(visible,length(p$words)))\n  continuations<-head(setdiff(p$completions$text,head(p$words,visible)),3L)\n  unit<-if(p$language")
old="""   tags$p(class='example-label','Two alternatives'),tags$div(class='alternatives',actionButton('choose2',p$words[2]),actionButton('choose3',p$words[3])),
   tags$details(class='more-words',tags$summary('More suggestions'),tags$p(class='fine-print','Ranks 4 to 10 from the selected language model.'),tags$ol(start='4',lapply(4:10,function(j)tags$li(actionButton(paste0('choose',j),p$words[j]))))),"""
new="""   tags$p(class='example-label',sprintf('%d alternatives',length(alternatives))),
   tags$div(class='alternatives',lapply(alternatives,function(j)actionButton(paste0('choose',j),p$words[j],title=sprintf('Suggestion %d. Add this word.',j)))),
   if(length(continuations))tags$section(class='phrase-completions',tags$h3('Short continuations'),
    tags$p(class='fine-print','Choose an extra word or a short ending.'),
    tags$div(class='completion-options',lapply(seq_along(continuations),function(j)actionButton(paste0('complete',j),continuations[j],title='Append this continuation'))),
    tags$p(class='fine-print',paste('Matched phrase:',p$completions$context)),
    tags$p(class='fine-print','Experimental suggestions from training phrases. Choose what fits your sentence.')),
"""
assert old in s;s=s.replace(old,new)
s=s.replace('Extra language support does not establish higher English accuracy.','These archived times cover ten word suggestions (three for English). The current screen can show twenty and retrieve short continuations, so its total time differs. Short continuations have not had an independent accuracy evaluation.')
p.write_text(s,encoding='utf-8')
p=app/'www/styles.css';s=p.read_text(encoding='utf-8')
s+='\n/* Version 1.4: visible choices and separate short continuations. */\n.alternatives{flex-wrap:wrap;gap:8px}.alternatives .btn{font-size:17px;padding:8px 13px;max-width:100%;overflow-wrap:anywhere}.phrase-completions{border-top:1px solid #c8dce9;margin-top:22px;padding-top:16px}.phrase-completions h3{font-size:17px;margin:0 0 7px}.completion-options{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.completion-options .btn{font-size:17px;font-weight:600;padding:8px 13px}.editor-panel #suggestion_count{font-size:14px;height:40px;min-height:40px}.editor-panel .checkbox{margin-bottom:13px}\n'
p.write_text(s,encoding='utf-8')
p=app/'www/input.js';s=p.read_text(encoding='utf-8');s=s.replace("event.target.closest('button[id^=choose]')","event.target.closest('button[id^=choose],button[id^=complete]')")
p.write_text(s,encoding='utf-8')
for f in stage.iterdir():
 if f.is_file():shutil.copy2(f,out/f.name)
for name in ['app.R','predictor.R','continuations.R']:shutil.copy2(app/name,out/name)
