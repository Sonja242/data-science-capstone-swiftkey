from pathlib import Path
p=Path.cwd()/'final_project/phraseflow_multilingual/app.R';s=p.read_text(encoding='utf-8')
s=s.replace("for(i in 1:3)local({j<-i;observeEvent(input[[paste0('complete',j)]]", "for(i in 1:8)local({j<-i;observeEvent(input[[paste0('complete',j)]]")
s=s.replace("  visible<-as.integer(input$suggestion_count %||% 10L)\n  choices<-head(setdiff(p$completions$text,head(p$words,visible)),3L)","  choices<-p$completions$text")
s=s.replace("  continuations<-head(setdiff(p$completions$text,head(p$words,visible)),3L)","  continuation_ids<-head(which(!p$completions$text %in% head(p$words,visible)),3L)\n  continuations<-p$completions$text[continuation_ids]")
s=s.replace("actionButton(paste0('complete',j),continuations[j]", "actionButton(paste0('complete',continuation_ids[j]),continuations[j]")
p.write_text(s,encoding='utf-8')
