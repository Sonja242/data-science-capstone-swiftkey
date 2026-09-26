from pathlib import Path
root=Path.cwd();out=root/'final_project/research/continuations-20260926-v14'
s=(root/'final_project/research/languages-20260926-v13/test_extension.R').read_text(encoding='utf-8')
s=s.replace("out<-'final_project/research/languages-20260926-v13'","out<-'final_project/research/continuations-20260926-v14'")
s=s.replace('length(result()$words)==10L','length(result()$words)==20L')
s=s.replace("automatic=TRUE,composing=FALSE","automatic=TRUE,composing=FALSE,suggestion_count=10")
s=s.replace("session$setInputs(phrase='I am looking forward to ')","session$setInputs(phrase='I am looking forward to ',phrase_context=list(text='I am looking forward to ',language='en_US'))")
s=s.replace("check<-list(languages=12,", "check<-list(languages=12,twenty_word_candidates=TRUE,")
(out/'test_features.R').write_text(s,encoding='utf-8')
