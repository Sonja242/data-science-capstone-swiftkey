"""Audit saved development evidence and document the local preview; no test input."""
from pathlib import Path
import csv, json, hashlib, collections
ROOT=Path.cwd()
OUT=ROOT/'final_project/research/phraseflow-20260926'
APP=ROOT/'final_project/phraseflow_preview'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(p,kind='sha256'):return hashlib.new(kind,p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
manifest=read(OUT/'training_manifest.json')
assert digest(OUT/'protocol.json')==manifest['protocol_sha256']
assert digest(APP/'model.rds','md5')==manifest['model_md5']
assert digest(ROOT/'final_project/app/model.rds','md5')==manifest['model_md5']
assert digest(APP/'predictor.R')==digest(ROOT/'final_project/app/predictor.R')
allrows=rows(OUT/'development_cases.csv');summary=rows(OUT/'selected_comparison.csv')
assert len(allrows)==3600 and len(summary)==6
for s in summary:
 r=[x for x in allrows if x['objective']==s['objective'] and x['seed']==s['seed']]
 assert len(r)==600 and len({x['case_id'] for x in r})==600
 for key in ['top1','top3']:assert sum(int(x[key]) for x in r)==int(s[key])
 assert sum(int(x['cpu_top1']) for x in r)==106
 assert sum(int(x['cpu_top3']) for x in r)==164
 assert sum(int(x['covered']) for x in r)==332
groups=collections.defaultdict(list)
for x in allrows:
 context='1-4 words' if int(x['context_words'])<=4 else ('5-12 words' if int(x['context_words'])<=12 else '13+ words')
 for dimension,value in [('source',x['source']),('context_length',context)]:
  groups[(x['objective'],x['seed'],dimension,value)].append(x)
out=[]
for (objective,seed,dimension,value),r in groups.items():
 out.append(dict(objective=objective,seed=seed,dimension=dimension,group=value,cases=len(r),
  **{k:sum(int(x[k]) for x in r) for k in ['covered','cpu_top1','cpu_top3','top1','top3']}))
with (OUT/'subgroup_results.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
pitch=(APP/'PhraseFlow_Pitch.Rpres').read_text(encoding='utf-8')
assert sum(line.startswith('==========') for line in pitch.splitlines())==5
appfile=APP/'app.R';app=appfile.read_text(encoding='utf-8')
app=app.replace('"Local computation, median"','"Archived three-word timing"')
app=app.replace('"These are unrestricted next-word results, not multiple-choice quiz scores. Local timing uses five randomized rounds on the same 600 development inputs.',
 '"These are unrestricted next-word results, not multiple-choice quiz scores. The archived timing above measures the three-suggestion version; it does not benchmark this ten-suggestion preview. Local timing uses five randomized rounds on the same 600 development inputs.')
appfile.write_text(app,encoding='utf-8')
save(OUT/'final_audit.json',dict(all_six_runs_reconciled=True,protocol_unchanged=True,
 production_model_unchanged=True,preview_model_identical=True,preview_predictor_identical=True,
 reserved_test_opened=False,pitch_slides=5,quiz_used_for_selection=False,
 public_deployment_changed=False))
save(OUT/'code_fingerprints.json',{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.suffix in ['.py','.R','.Rmd']})
print('Evidence reconciled; model and protocol unchanged; five slides; subgroups saved.')
