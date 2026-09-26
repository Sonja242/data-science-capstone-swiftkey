from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,shutil
root=Path.cwd();app=root/'final_project/phraseflow_multilingual';out=root/'final_project/research/continuations-20260926-v14';old=root/'final_project/phraseflow_multilingual_v13_20260926'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Audit(HTMLParser):
 def __init__(self):super().__init__();self.slides=0;self.images=[]
 def handle_starttag(self,tag,attrs):
  if tag=='section':self.slides+=1
  if tag=='img':self.images.append(dict(attrs).get('src',''))
a=Audit();deck=(app/'PhraseFlow_Pitch.html').read_text(encoding='utf-8');a.feed(deck)
assert a.slides==5 and len(a.images)==3 and all(x.startswith('data:image/') for x in a.images)
assert 'Local preview 1.4' in deck and '5, 10 or 20 words' in deck
assert sha(app/'Multilingual_Guide.html')==sha(app/'www/Multilingual_Guide.html')
models=[app/'model.rds',*sorted((app/'languages').glob('*.rds'))]
assert len(models)==12 and all(sha(p)==sha(old/p.relative_to(app)) for p in models)
checks=json.loads((out/'checks.json').read_text(encoding='utf-8'))
assert checks['english_exact_600'] and checks['twenty_word_candidates'] and not checks['english_reserved_test_opened']
for f in Path(__file__).parent.iterdir():
 if f.is_file() and f.resolve()!=(out/f.name).resolve():shutil.copy2(f,out/f.name)
for name in ['app.R','predictor.R','continuations.R','Multilingual_Guide.Rmd','PhraseFlow_Pitch.Rpres']:shutil.copy2(app/name,out/name)
for name in ['input.js','styles.css']:shutil.copy2(app/'www'/name,out/name)
result={'version':'1.4','model_files_unchanged':12,'native_presenter_slides':5,'embedded_images':3,
 'knitted_guide_matches_app_copy':True,'automated_checks':'checks.json and continuation_checks.json',
 'browser_checks':['5/10/20 visible word choices','Dutch single-word insertion and refreshed predictions','Dutch two-word insertion and refreshed predictions','Chinese insertion without spaces','Immediate typing after language change preserved','New guide and slide illustration inspected'],
 'language_models':12,'quality_limitation':'No independent accuracy evaluation of supplementary endings. Existing word rankings unchanged.',
 'deployed':False}
(out/'release_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(out/'release_sha256.json').write_text(json.dumps({str(p.relative_to(app)).replace('\\','/'):sha(p) for p in sorted(app.rglob('*')) if p.is_file()},indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['version','model_files_unchanged','native_presenter_slides','knitted_guide_matches_app_copy','deployed']},indent=2))
