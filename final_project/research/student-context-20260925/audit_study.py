"""Recompute public comparisons from saved case records. No final-test access."""
from pathlib import Path
import csv,json,hashlib,platform,importlib.metadata
from collections import Counter
ROOT=Path.cwd();OUT=ROOT/'final_project/research/student-context-20260925';DATA=ROOT/'data/student_context'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
d=rows(OUT/'development_cases.csv');summary=rows(OUT/'development_comparison.csv')
py=rows(OUT/'python_development_cases.csv')
assert len(d)==len(py)==7800
key=lambda r:(r['name'],r['source'],r['line_hash'])
assert {key(r):r['rank'] for r in d}=={key(r):r['rank'] for r in py}
for row in summary:
 subset=[r for r in d if r['name']==row['name']]
 assert len(subset)==600 and Counter(r['source']for r in subset)=={'blogs':200,'news':200,'twitter':200}
 assert sum(int(r['rank'])==1 for r in subset)==int(row['top1'])
 assert sum(int(r['rank'])>0 for r in subset)==int(row['top3'])
train=rows(OUT/'training_hashes.csv');dev=rows(DATA/'development.csv')
reserved=rows(ROOT/'final_project/research/speed-quality-20260925/reserved_hashes.csv')
assert len({r['line_hash']for r in train})==90000
assert not ({r['line_hash']for r in train}&{r['line_hash']for r in dev+reserved})
teacher=rows(DATA/'teacher_training.csv')
assert {r['line_hash']for r in teacher}<={r['line_hash']for r in train}
for name in ['CE_control','Distill25','Distill50']:
 spec=read(DATA/name/'weights.json')
 assert all(sha(DATA/name/s['file'])==s['sha256'] for s in spec.values())
checks=read(OUT/'export_checks.json')
assert all(r['passed'] for r in checks.values())
decision=read(OUT/'decision.json')
eligible=[r for r in summary if r['qualified']=='TRUE']
assert len(eligible)==decision['qualifying_candidates']
if not eligible:assert decision['selected']=='Current'and not decision['final_test_used']
else:
 expected=sorted(eligible,key=lambda r:(-int(r['top3']),-int(r['top1']),float(r['median_ms'])))[0]['name']
 assert expected==decision['selected']
timing=rows(OUT/'timing.csv');assert len(timing)==23400
assert all(v==1800 for v in Counter(r['name']for r in timing).values())
result={'author':'Sonja Sahebzad','passed':True,'case_comparisons':len(d),'python_R_ranks_identical':True,
 'train_lines':len(train),'teacher_cases':len(teacher),'no_train_dev_or_reserved_overlap':True,
 'timed_calls':len(timing),'reserved_test_opened_by_audit':False,'selection_recomputed':True,
 'package_versions':{p:importlib.metadata.version(p)for p in ['torch','numpy','transformers','peft']},
 'files_sha256':{p.name:sha(p)for p in OUT.iterdir()if p.suffix in ['.py','.R','.json','.csv']and p.name!='audit.json'},
 'base_model_manifest':read(ROOT/'models/neural/Qwen3-1.7B-Base/download_manifest.json')}
(OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items()if k not in ['files_sha256','package_versions','base_model_manifest']},indent=2))
