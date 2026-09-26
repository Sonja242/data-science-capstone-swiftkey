from pathlib import Path
root=Path.cwd();p=root/'final_project/phraseflow_multilingual/app.R';s=p.read_text(encoding='ascii')
s=s.replace('?v=1.3\'','?v=1.3.1\'')
s=s.replace("  tags$div(`data-prediction-phrase`=last_phrase()", "  unit<-if(p$language %in% c('zh_CN','ko_KR'))'unit' else 'word'\n  tags$div(`data-prediction-phrase`=last_phrase()")
s=s.replace("if(p$context_words==1L)'word' else 'words'","if(p$context_words==1L)unit else paste0(unit,'s')")
s=s.replace("sprintf('%d input words. At most 4 recent words guide prediction.',p$input_words)","sprintf('%d input %s. At most 4 recent %s guide prediction.',p$input_words,if(p$input_words==1L)unit else paste0(unit,'s'),paste0(unit,'s'))")
p.write_text(s,encoding='ascii')
(root/'final_project/research/languages-20260926-v13/app.R').write_text(s,encoding='ascii')
