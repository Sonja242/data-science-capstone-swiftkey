from pathlib import Path
p=Path.cwd()/'final_project/phraseflow_multilingual/app.R'
s=p.read_text(encoding='utf-8')
# input.js clears old text synchronously when the language changes. A delayed
# server value='' can otherwise erase newly typed text in the new language.
s=s.replace("updateTextAreaInput(session,'phrase',value='',placeholder=language_examples[[code]][1])",
            "updateTextAreaInput(session,'phrase',placeholder=language_examples[[code]][1])")
s=s.replace("reset_phrase(input$phrase %||% '');awaiting_language_reset(TRUE)","reset_phrase(input$phrase %||% '');awaiting_language_reset(FALSE)")
s=s.replace("  tokenizer<-language_tokenizers[[code]]", "  ctx<-input$phrase_context\n  if(!is.null(ctx)&&(!identical(ctx$language,code)||!identical(ctx$text,phrase)))return()\n  tokenizer<-language_tokenizers[[code]]")
p.write_text(s,encoding='utf-8')
p=Path.cwd()/'final_project/phraseflow_multilingual/www/input.js'
s=p.read_text(encoding='utf-8')
s=s.replace("if (box && window.Shiny && !composing) Shiny.setInputValue('phrase', box.value, {priority: 'event'});", "if (box && window.Shiny && !composing) {\n      Shiny.setInputValue('phrase_context', {text:box.value, language:document.getElementById('language').value}, {priority:'event'});\n      Shiny.setInputValue('phrase', box.value, {priority: 'event'});\n    }")
p.write_text(s,encoding='utf-8')
