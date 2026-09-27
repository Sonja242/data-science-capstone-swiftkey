"""Author: Sonja Sahebzad | Sonja Projects. Preserve the live app; create a preview."""
from pathlib import Path
import shutil, json, hashlib
ROOT=Path.cwd();base=ROOT/'final_project/app';out=ROOT/'final_project/phraseflow_preview'
if out.exists():raise FileExistsError('Preserve existing preview.')
out.mkdir();(out/'www').mkdir()
for name in ['model.rds','predictor.R','metrics.json','runtime-metrics.json']:
 shutil.copy2(base/name,out/name)
for name in ['styles.css','input.js']:shutil.copy2(base/'www'/name,out/'www'/name)
text=(base/'app.R').read_text(encoding='utf-8-sig')
assert text.count('p <- predict_word(model,phrase)')==1 and text.count('for (i in 1:3)')==1
text=text.replace('Sonja Next Word','Sonja PhraseFlow')
text=text.replace('tags$h1("Find your next word.")','tags$h1("Let your next word flow.")')
text=text.replace('"Type an English phrase. Get one suggestion, plus two alternatives to keep your writing moving."',
 '"Write an English phrase. Choose a suggestion and keep your sentence moving."')
text=text.replace('p <- predict_word(model,phrase)','p <- predict_word(model,phrase,10L)')
text=text.replace('for (i in 1:3)','for (i in 1:10)')
needle='tags$div(class="evidence",tags$strong(if(p$order==1L)'
assert text.count(needle)==1
text=text.replace(needle,'''tags$details(class="more-words",tags$summary("More suggestions"),
        tags$p(class="fine-print","Alternatives 4 to 10, ordered by the same model. Select a word to add it."),
        tags$ol(start="4",lapply(4:10,function(j)tags$li(actionButton(paste0("choose",j),p$words[j]))))),
      tags$div(class="evidence",tags$strong(if(p$order==1L)''')
needle='tags$section(class="content-panel guide",tags$h2("Three steps to keep writing"),'
assert text.count(needle)==1
text=text.replace(needle,needle+'''
          tags$p(tags$a(href="PhraseFlow_User_Guide.pdf",target="_blank","Open the illustrated user guide (PDF)"),
                 " | ",tags$a(href="PhraseFlow_User_Guide.html",target="_blank","Read the HTML guide")),
          tags$p(class="fine-print","Author: Sonja Sahebzad. Product preview 1.1. Documentation: 26 September 2026."),''')
text=text.replace('tags$span("Sonja Projects · 2026")','tags$span("Sonja Projects · Preview 1.1 · 2026")')
(out/'app.R').write_text(text,encoding='utf-8')
js=(out/'www/input.js').read_text()
assert js.count("#choose1, #choose2, #choose3")==1
js=js.replace("#choose1, #choose2, #choose3","button[id^=choose]")
(out/'www/input.js').write_text(js,encoding='utf-8')
with (out/'www/styles.css').open('a',encoding='utf-8') as f:
 f.write('\n.more-words{margin:22px 0 0;border-top:1px solid var(--line);padding-top:14px}.more-words summary{color:var(--navy);font-size:14px;font-weight:600;cursor:pointer}.more-words ol{display:grid;grid-template-columns:1fr 1fr;gap:7px 28px;padding-left:24px;margin-top:12px}.more-words li{font-size:12px;color:var(--muted)}.more-words .btn{padding:5px 12px;font-size:15px;text-align:left;width:100%}\n')
shutil.copy2(ROOT/'final_project/Sonja Next Word.Rproj',out/'Sonja PhraseFlow.Rproj')
(out/'preview_manifest.json').write_text(json.dumps({'author':'Sonja Sahebzad','brand':'Sonja Projects',
 'title':'Sonja PhraseFlow','version':'1.1 preview','date':'2026-09-26','status':'Local preview; not deployed',
 'model':'Unchanged validated CPU n-gram model; experimental reranker is separate',
 'model_md5':hashlib.md5((out/'model.rds').read_bytes()).hexdigest(),
 'new_features':['Expandable ranks4-10 with clickable words','PDF and HTML user guides','Product name and document metadata'],
 'production_changed':False},indent=2),encoding='utf-8')
print(out)
