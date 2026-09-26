from pathlib import Path
from html.parser import HTMLParser
import json, hashlib, shutil, re

root=Path('C:/Users/csj50/OneDrive/Documents/Sonja report/Data Science Capstone')
app=root/'final_project/phraseflow_multilingual'
pub=root/'final_project/publication-20260927'
site='https://sonja242.github.io/data-science-capstone-swiftkey/'
shutil.copy2(app/'Multilingual_Guide.html',root/'docs/phraseflow-guide.html')
shutil.copy2(app/'PhraseFlow_Pitch.html',root/'docs/phraseflow-pitch.html')
s=(app/'Third_Party_Notices.html').read_text(encoding='utf-8').replace('www/attribution/','phraseflow-attribution/').replace('href="Multilingual_Guide.html"','href="phraseflow-guide.html"')
(root/'docs/phraseflow-data-credits.html').write_text(s,encoding='utf-8')
shutil.copytree(app/'www/attribution',root/'docs/phraseflow-attribution',dirs_exist_ok=True)

runtime=['app.R','predictor.R','continuations.R','language-config.json','metrics.json','runtime-metrics.json','model.rds']
runtime += [p.relative_to(app).as_posix() for d in ['languages','continuations'] for p in sorted((app/d).glob('*')) if p.is_file()]
runtime += ['www/input.js','www/styles.css','www/Multilingual_Guide.html','www/Third_Party_Notices.html']
runtime += [p.relative_to(app).as_posix() for d in ['www/attribution','www/figures'] for p in sorted((app/d).glob('*')) if p.is_file()]
runtime=sorted(set(runtime))
assert all((app/p).is_file() for p in runtime)
(pub/'app-files.txt').write_text('\n'.join(runtime)+'\n',encoding='utf-8')
(pub/'portfolio-files.txt').write_text('app.R\nwww/styles.css\nwww/favicon.svg\n',encoding='utf-8')

class Audit(HTMLParser):
    def __init__(self):super().__init__();self.slides=0;self.images=[];self.links=[]
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if t=='section':self.slides+=1
        if t=='img':self.images.append(a.get('src',''))
        if t=='a':self.links.append(a.get('href',''))
a=Audit();deck=(app/'PhraseFlow_Pitch.html').read_text(encoding='utf-8');a.feed(deck)
assert a.slides==5 and len(a.images)==3 and all(s.startswith('data:image/') for s in a.images)
assert 'Local preview' not in deck and 'Version 1.4' in deck
assert site+'phraseflow-guide.html' in a.links and site+'phraseflow-data-credits.html' in a.links
assert 'Multilingual_Guide.html' not in a.links
assert (app/'Multilingual_Guide.html').read_bytes()==(app/'www/Multilingual_Guide.html').read_bytes()
assert hashlib.md5((app/'model.rds').read_bytes()).hexdigest()=='bd58c642df1759c4ce1c39b9ace8927c'
assert not any('rsconnect' in p or 'test_cases' in p for p in runtime)
checks={'version':'1.4','native_presenter_slides':a.slides,'embedded_images':len(a.images),'public_documentation_links':True,'guide_knitted_and_copied':True,'english_model_unchanged':True,'languages':12,'bundle_files':len(runtime),'bundle_mib':sum((app/p).stat().st_size for p in runtime)/2**20,'reserved_english_test_opened':False,'course_submitted':False,'peer_message_sent':False}
(pub/'release-checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
(pub/'app-sha256.json').write_text(json.dumps({p:hashlib.sha256((app/p).read_bytes()).hexdigest() for p in runtime},indent=2)+'\n',encoding='utf-8')

paths=['.gitignore','README.md','docs/index.html','final_project/README.md','final_project/Submission_Details.md']
paths += [p.relative_to(root).as_posix() for p in sorted((root/'docs').glob('phraseflow-*')) if p.is_file()]
paths += [p.relative_to(root).as_posix() for p in sorted((root/'docs/phraseflow-attribution').glob('*')) if p.is_file()]
paths += [(app/p).relative_to(root).as_posix() for p in runtime]
paths += [(app/p).relative_to(root).as_posix() for p in ['README.md','Sonja PhraseFlow.Rproj','Multilingual_Guide.Rmd','Multilingual_Guide.html','PhraseFlow_Pitch.Rpres','PhraseFlow_Pitch.md','PhraseFlow_Pitch.html','pitch.css','Third_Party_Notices.html','preview_manifest.json','www/multilingual-demo.png']]
for d in ['multilingual-20260926','languages-20260926-v13','continuations-20260926-v14']:
    for p in sorted((root/'final_project/research'/d).iterdir()):
        if p.is_file() and not p.name.endswith('_index.tsv'):
            paths.append(p.relative_to(root).as_posix())
# This file list is refreshed just before staging to include verified publication results.
paths += [p.relative_to(root).as_posix() for p in sorted(pub.iterdir()) if p.is_file() and p.name!='git-files.txt']
paths=sorted(set(paths))
assert not any('rsconnect' in p or p.startswith('data/') or '_index.tsv' in p for p in paths)
(pub/'git-files.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))
