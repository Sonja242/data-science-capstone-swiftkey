"""Author: Sonja Sahebzad | Sonja Projects. Prefix-only frozen teacher scoring."""
from pathlib import Path
import json, hashlib, sys, time
import numpy as np
import torch
ROOT=Path.cwd();OUT=ROOT/'final_project/research/phraseflow-20260926';DATA=ROOT/'data/phraseflow_20260926'
sys.path.insert(0,str(ROOT/'python'))
from adaptation_experiment import AdaptedRanker
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
assert (OUT/'training_manifest.json').exists()
if (OUT/'teacher_training_cost.json').exists():raise FileExistsError('Preserve completed scoring.')
req=[json.loads(s) for s in (DATA/'training_requests.jsonl').read_text(encoding='utf-8-sig').splitlines()]
assert len(req)==3000
assert all(set(r)=={'case_id','source','line_hash','prefix','candidate_ids','candidate_words'} for r in req)
prior=read(ROOT/'final_project/research/candidate-diagnosis-20260926/teacher_fingerprint.json')
assert sha(ROOT/'models/neural/adaptation_local/adapter_model.safetensors')==prior['adapter_sha256']
assert sha(ROOT/'python/neural_predictor.py')==prior['neural_code_sha256']
assert sha(ROOT/'python/adaptation_experiment.py')==prior['adapted_code_sha256']
save(OUT/'teacher_training_fingerprint.json',{'teacher':prior,'requests_sha256':sha(DATA/'training_requests.jsonl'),
 'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json')})
ranker=AdaptedRanker('local256');assert ranker.max_context==128 and ranker.batch_size==64
ranker.check_scoring();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
p=DATA/'teacher_training_scores.jsonl';done=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
assert all(a['line_hash']==b['line_hash'] and a['word_ids']==b['candidate_ids'] for a,b in zip(done,req))
start=time.perf_counter()
with p.open('a',encoding='utf-8') as f:
 for i in range(len(done),len(req)):
  r=req[i];torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode():
   prefix,prefill=ranker._prefill(r['prefix']);scores=ranker._score_candidates(prefix,prefill,r['candidate_words'])
  torch.cuda.synchronize();ms=1000*(time.perf_counter()-t)
  assert set(scores)==set(r['candidate_words']) and np.isfinite(list(scores.values())).all()
  f.write(json.dumps({'case_id':r['case_id'],'source':r['source'],'line_hash':r['line_hash'],
   'word_ids':r['candidate_ids'],'scores':[scores[w] for w in r['candidate_words']],'milliseconds':ms})+'\n');f.flush()
  del prefill
  if (i+1)%100==0:print(f'Training teacher: {i+1}/3000; {time.perf_counter()-start:.1f}s this run',flush=True)
rs=[json.loads(s) for s in p.read_text().splitlines()];ms=np.array([r['milliseconds'] for r in rs])
save(OUT/'teacher_training_cost.json',{'cases':len(rs),'device':'GPU','gpu':torch.cuda.get_device_name(),
 'seconds':float(ms.sum()/1000),'median_ms':float(np.median(ms)),'p95_ms':float(np.quantile(ms,.95)),
 'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'scores_sha256':sha(p),'reserved_test_opened':False})
