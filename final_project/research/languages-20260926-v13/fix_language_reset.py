from pathlib import Path
root=Path.cwd();app=root/'final_project/phraseflow_multilingual';out=root/'final_project/research/languages-20260926-v13'
p=app/'app.R';s=p.read_text(encoding='ascii')
s=s.replace(" active_language<-reactiveVal('en_US')", " active_language<-reactiveVal('en_US')\n awaiting_language_reset<-reactiveVal(FALSE);reset_phrase<-reactiveVal('')")
s=s.replace("  active_language(code);result(NULL)", "  reset_phrase(input$phrase %||% '');awaiting_language_reset(TRUE)\n  active_language(code);result(NULL)")
s=s.replace(" observeEvent(input$phrase,{result(NULL);error(NULL)}", " observeEvent(input$phrase,{\n  if(isTRUE(awaiting_language_reset())&&!identical(input$phrase %||% '',reset_phrase()))awaiting_language_reset(FALSE)\n  result(NULL);error(NULL)}")
s=s.replace("  if(isTRUE(input$composing))return()", "  if(isTRUE(input$composing)||isTRUE(awaiting_language_reset()))return()")
p.write_text(s,encoding='ascii');(out/'app.R').write_text(s,encoding='ascii')
