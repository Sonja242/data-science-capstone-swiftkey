from pathlib import Path
root=Path.cwd();app=root/'final_project/phraseflow_multilingual'
p=app/'Multilingual_Guide.Rmd';s=p.read_text(encoding='utf-8').replace('Preview 1.3','Preview 1.4')
s=s.replace("checks<-fromJSON('../research/languages-20260926-v13/checks.json')","checks<-fromJSON('../research/continuations-20260926-v14/checks.json')")
s=s.replace('**More suggestions** opens ranks four through ten.','**Word choices** selects 5, 10 or 20 visible suggestions. **Short continuations** offers extra words or endings where training text matches.')
section='''## More choices and short continuations

The first suggestion stays prominent; the other selected word choices are all visible beneath it. Select **5, 10 or 20 suggestions** to adjust the amount shown. More choices help a writer find a suitable alternative, but do not improve the ranking of the first suggestion.

**Short continuations** is a separate, experimental lookup of endings from existing training sentences. It matches the last two to four normalized units and offers up to three alternatives. Three- or four-unit matches can add an observed ending of up to three units. Two-unit matches offer only the next unit because there is less context. The matched phrase appears below the buttons. No match means that no extra continuation is shown.

The lists were built from the existing training files in all twelve languages. Candidate endings are ranked by their training frequency; no quiz answers or evaluation targets were added. Only contexts with at least two retained training occurrences are kept. Selecting a button appends its words, then the predictor updates for the new text. Chinese continues without inserted spaces. The existing next-word scores are unchanged.

For example, the Dutch next-word model already ranks **elkaar** first after **we houden van**. The extra lookup can offer **gelukkig** after the suffix **lang en**, which the compact next-word ranking missed. These are inspected demonstrations, not independent accuracy measurements. A local phrase match can still be inappropriate for the meaning of the entire sentence. The app does not yet reliably understand a writer's intention or guarantee a natural complete sentence.

'''
s=s.replace('## Chinese, Korean and Hindi input',section+'## Chinese, Korean and Hindi input')
s=s.replace('Reported inference times measure ten suggestions on the local CPU after loading.','The archived inference times measure ten suggestions on the local CPU after loading. The current screen computes twenty word candidates and performs the additional ending lookup; the archived times do not benchmark that whole operation.')
s=s.replace('This excludes the R process, Shiny, cached pages and temporary allocations;','This includes the stored ending indexes but excludes the R process, Shiny, cached pages and temporary allocations;')
s=s.replace('## Discussion and next improvements','The next-word accuracy graphs apply to the unchanged word model. The supplementary ending lookup has not received an independent accuracy evaluation. Its training phrases are not an evaluation set.\n\n## Discussion and next improvements')
p.write_text(s,encoding='utf-8')
p=app/'PhraseFlow_Pitch.Rpres';s=p.read_text(encoding='utf-8').replace('Local preview 1.3','Local preview 1.4')
s=s.replace('<b>More suggestions</b> opens ranks four through ten.','Choose <b>5, 10 or 20 words</b>. Extra buttons offer short endings from matching training phrases.')
s=s.replace('The app\'s <b>User guide / Handleiding</b> button opens the same instructions. Results shows the selected language\'s evidence.','The <b>User guide / Handleiding</b> button explains both features. Short endings remain experimental; the accuracy charts measure the word model.')
p.write_text(s,encoding='utf-8')
