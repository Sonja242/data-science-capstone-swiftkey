from pathlib import Path
app=Path.cwd()/'final_project/phraseflow_multilingual'
p=app/'app.R';s=p.read_text(encoding='utf-8')
s=s.replace("server<-function(input,output,session){", "server<-function(input,output,session){\n set_phrase<-function(value)session$sendCustomMessage('phraseflow-set-text',list(text=value,language=input$language %||% 'en_US'))")
s=s.replace("updateTextAreaInput(session,'phrase',value=value)","set_phrase(value)")
s=s.replace("updateTextAreaInput(session,'phrase',value='')","set_phrase('')")
s=s.replace("updateTextAreaInput(session,'phrase',value=paste0(language_examples[[input$language %||% 'en_US']][j],if(identical(input$language,'zh_CN'))'' else ' '))", "set_phrase(paste0(language_examples[[input$language %||% 'en_US']][j],if(identical(input$language,'zh_CN'))'' else ' '))")
s=s.replace('v=1.4.0','v=1.4.1');p.write_text(s,encoding='utf-8')
p=app/'www/input.js';s=p.read_text(encoding='utf-8')
s=s.replace("  document.addEventListener('compositionstart'", """  $(document).on('shiny:connected', function () {
    Shiny.addCustomMessageHandler('phraseflow-set-text', function (message) {
      const box = phraseBox(), language = document.getElementById('language');
      if (!box || !language || language.value !== message.language) return;
      box.value = message.text;
      box.dispatchEvent(new Event('input', {bubbles:true}));
    });
  });
  document.addEventListener('compositionstart'""")
p.write_text(s,encoding='utf-8')
