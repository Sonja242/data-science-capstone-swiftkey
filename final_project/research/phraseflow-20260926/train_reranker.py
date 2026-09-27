"""Sonja Projects | Author: Sonja Sahebzad.
Bounded, registered candidate-conditional reranker. No test or quiz inputs.
"""
from pathlib import Path
import csv, json, hashlib, sys, time, math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path.cwd();OUT=ROOT/'final_project/research/phraseflow-20260926';DATA=ROOT/'data/phraseflow_20260926'
OLD=ROOT/'final_project/research/student-context-20260925';DIAG=ROOT/'final_project/research/candidate-diagnosis-20260926'
sys.path.insert(0,str(OLD))
from student_study import Student, inputs, WORDS, LOOKUP
WORD_ARRAY=np.array(WORDS)
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jl(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines()]
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def csvout(p,rs):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(OUT/'protocol.json');cfg=protocol['training']
if (OUT/'selection.json').exists():raise FileExistsError('Preserve completed training.')
assert (OUT/'teacher_training_cost.json').exists()
student=Student().cuda().eval()
initial=ROOT/'data/student_context/supervised_initial.pt'
student.load_state_dict(torch.load(initial,map_location='cuda',weights_only=True))
for p in student.parameters():p.requires_grad_(False)
base=np.array([float(r['base_probability']) for r in rows(ROOT/'data/student_context/vocabulary.csv')])
char_lengths=np.array([len(w) for w in WORDS])

def dataset(split):
 training=split=='training'
 cases=rows(ROOT/('data/student_context/teacher_training.csv' if training else 'data/student_context/development.csv'))
 candidates=rows(DATA/'training_candidates.csv' if training else DIAG/'cpu_candidates.csv')
 teacher=jl(DATA/'teacher_training_scores.jsonl' if training else DIAG/'teacher_scores.jsonl')
 n=len(cases);assert n==(3000 if training else 600) and len(candidates)==n*128 and len(teacher)==n
 ids=np.array([int(r['word_id']) for r in candidates]).reshape(n,128)
 cpu=np.array([float(r['cpu_probability']) for r in candidates],dtype=np.float32).reshape(n,128)
 assert (cpu>0).all()
 assert all(c['line_hash']==t['line_hash'] and list(ids[i])==t['word_ids'] for i,(c,t) in enumerate(zip(cases,teacher)))
 target=np.array([LOOKUP.get(c['actual'],-1) for c in cases]);y=np.full(n,-100,dtype=np.int64)
 for i in range(n):
  at=np.where(ids[i]==target[i])[0]
  if len(at):y[i]=at[0]
 x,l=inputs([c['prefix'] for c in cases]);zs=[]
 with torch.inference_mode():
  for at in range(0,n,128):
   xx=torch.tensor(x[at:at+128],device='cuda');ll=torch.tensor(l[at:at+128],device='cuda')
   hh,_=student.gru(student.embedding(xx));h=hh[torch.arange(len(xx),device='cuda'),ll-1]
   zs.append(torch.tanh(student.projection(h)).cpu().numpy())
 z=np.concatenate(zs);embedding=student.embedding.weight.detach().cpu().numpy()[2:][ids]
 zwide=np.broadcast_to(z[:,None,:],embedding.shape)
 logits=(embedding*zwide).sum(2)+student.bias.detach().cpu().numpy()[ids]
 seen=(ids[:,:,None]+2==x[:,None,:]).any(2).astype(np.float32)
 lengths=np.array([len(c['prefix'].split()) for c in cases])
 scalars=np.stack([np.log(cpu),np.log(base[ids]),logits,np.log1p(char_lengths[ids]),seen,
  np.broadcast_to(np.log1p(lengths)[:,None],ids.shape)],axis=2)
 features=np.concatenate([zwide,embedding,zwide*embedding,scalars],axis=2).astype(np.float32)
 return {'cases':cases,'ids':ids,'cpu':cpu,'y':y,'target':target,'features':features,
  'teacher':np.array([r['scores'] for r in teacher],np.float32)}

t0=time.perf_counter();train=dataset('training');dev=dataset('development')
assert not set(c['line_hash'] for c in train['cases'])&set(c['line_hash'] for c in dev['cases'])
mean=train['features'].mean(axis=(0,1),dtype=np.float64).astype(np.float32)
scale=np.maximum(train['features'].std(axis=(0,1),dtype=np.float64),.0001).astype(np.float32)
for d in [train,dev]:
 d['x']=torch.tensor((d['features']-mean)/scale,device='cuda');del d['features']
 d['logcpu']=torch.tensor(np.log(d['cpu']),device='cuda')
 d['ty']=torch.tensor(d['y'],device='cuda')
 d['tlogq']=F.log_softmax(torch.tensor(d['teacher'],device='cuda')/2,dim=1)
 d['tq']=d['tlogq'].exp()
feature_seconds=time.perf_counter()-t0

class Head(nn.Module):
 def __init__(self):
  super().__init__();self.hidden=nn.Linear(198,64);self.output=nn.Linear(64,1)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,x,logcpu):return logcpu+self.output(torch.tanh(self.hidden(x))).squeeze(-1)

def predictions(score,d):
 # Use alphabetical tie handling, matching native R and the original CPU engine.
 order=np.array([np.lexsort((WORD_ARRAY[ids],-r))[:3] for ids,r in zip(d['ids'],score)])
 selected=np.take_along_axis(d['ids'],order,1)
 return selected,selected[:,0]==d['target'],(selected==d['target'][:,None]).any(1)

def evaluate(head,d):
 head.eval();scores=[]
 with torch.inference_mode():
  for at in range(0,len(d['y']),128):scores.append(head(d['x'][at:at+128],d['logcpu'][at:at+128]).cpu().numpy())
 s=np.concatenate(scores);logp=torch.log_softmax(torch.tensor(s,device='cuda')/2,dim=1)
 covered=d['ty']>=0
 ce=float(F.cross_entropy(torch.tensor(s,device='cuda')[covered],d['ty'][covered]))
 kl=float((d['tq']*(d['tlogq']-logp)).sum(1).mean())
 top,a,b=predictions(s,d)
 return {'top1':int(a.sum()),'top3':int(b.sum()),'covered':int(covered.sum()),'target_ce_covered':ce,'conditional_teacher_kl_T2':kl},s,top

check=Head().cuda();initial_result,initial_scores,_=evaluate(check,dev)
assert initial_result['top1']==106 and initial_result['top3']==164
curve=[];cost=[];models={};checkpoint_dir=DATA/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
for alpha in cfg['teacher_weight']:
 name='TargetOnly' if alpha==0 else 'ConditionalTeacher25'
 for seed in cfg['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);head=Head().cuda()
  opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
  start=time.perf_counter()
  for epoch in range(cfg['epochs']+1):
   if epoch:
    head.train();order=rng.permutation(len(train['y']))
    for at in range(0,len(order),cfg['batch_size']):
     ix=order[at:at+cfg['batch_size']];s=head(train['x'][ix],train['logcpu'][ix]);y=train['ty'][ix]
     mask=y>=0;ce=F.cross_entropy(s[mask],y[mask]) if mask.any() else s.sum()*0
     logp=F.log_softmax(s/2,dim=1);kl=(train['tq'][ix]*(train['tlogq'][ix]-logp)).sum(1).mean()
     loss=(1-alpha)*ce+alpha*4*kl
     opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
   tr,_,_=evaluate(head,train);dr,_,_=evaluate(head,dev)
   for split,metrics in [('training',tr),('development',dr)]:
    curve.append({'objective':name,'teacher_weight':alpha,'seed':seed,'epoch':epoch,'split':split,**metrics,
     'target_loss_weighted':(1-alpha)*metrics['target_ce_covered'],'teacher_loss_weighted':alpha*4*metrics['conditional_teacher_kl_T2']})
   torch.save(head.state_dict(),checkpoint_dir/f'{name}_{seed}_epoch{epoch}.pt')
   print(f'{name} seed{seed} epoch{epoch}: train {tr["top1"]}/{tr["top3"]}; dev {dr["top1"]}/{dr["top3"]}',flush=True)
  torch.cuda.synchronize();cost.append({'objective':name,'seed':seed,'device':'GPU','seconds_training_and_evaluation':time.perf_counter()-start})
csvout(OUT/'learning_curves.csv',curve);save(OUT/'training_cost.json',{'feature_seconds':feature_seconds,'runs':cost,
 'gpu':torch.cuda.get_device_name(),'frozen_encoder_sha256':sha(initial),'trainable_parameters':sum(p.numel() for p in head.parameters()),
 'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(Path(__file__)),'reserved_test_opened':False})
averages=[]
for name in ['TargetOnly','ConditionalTeacher25']:
 for epoch in range(1,cfg['epochs']+1):
  rs=[r for r in curve if r['objective']==name and r['epoch']==epoch and r['split']=='development']
  averages.append({'objective':name,'epoch':epoch,'mean_top1':float(np.mean([r['top1'] for r in rs])),'mean_top3':float(np.mean([r['top3'] for r in rs]))})
best_per_objective=[sorted([r for r in averages if r['objective']==name],key=lambda r:(-r['mean_top3'],-r['mean_top1'],r['epoch']))[0] for name in ['TargetOnly','ConditionalTeacher25']]
chosen=sorted(best_per_objective,key=lambda r:(-r['mean_top3'],-r['mean_top1'],r['objective']!='TargetOnly'))[0]
summary=[];case_results=[];score_selected=None;head_selected=None
baseline1=dev['y']==0;baseline3=(dev['y']>=0)&(dev['y']<3)
source=np.array([r['source'] for r in dev['cases']]);groups=[np.where(source==s)[0] for s in sorted(set(source))]
def paired_ci(delta):
 rng=np.random.default_rng(20260926);boot=np.empty(10000)
 for b in range(len(boot)):boot[b]=np.concatenate([delta[rng.choice(g,len(g),replace=True)] for g in groups]).mean()
 return np.quantile(boot,[.025,.975]).tolist()
for selection in best_per_objective:
 for seed in cfg['seeds']:
  name=selection['objective'];epoch=selection['epoch'];head=Head().cuda()
  head.load_state_dict(torch.load(checkpoint_dir/f'{name}_{seed}_epoch{epoch}.pt',weights_only=True))
  metrics,s,top=evaluate(head,dev);a=top[:,0]==dev['target'];b=(top==dev['target'][:,None]).any(1)
  lo,hi=paired_ci(b.astype(int)-baseline3.astype(int))
  summary.append({'objective':name,'epoch':epoch,'seed':seed,**metrics,'top3_gained':int((b&~baseline3).sum()),'top3_lost':int((~b&baseline3).sum()),
   'top3_difference':float((b.astype(int)-baseline3.astype(int)).mean()),'top3_difference_low':lo,'top3_difference_high':hi})
  for i,c in enumerate(dev['cases']):
   case_results.append({'objective':name,'epoch':epoch,'seed':seed,'case_id':i+1,'source':c['source'],'line_hash':c['line_hash'],
    'context_words':len(c['prefix'].split()),'covered':int(dev['y'][i]>=0),'cpu_top1':int(baseline1[i]),'cpu_top3':int(baseline3[i]),
    'top1':int(a[i]),'top3':int(b[i])})
  if name==chosen['objective'] and seed==cfg['seeds'][0]:score_selected=s;head_selected=head.state_dict();selected_ids=top
csvout(OUT/'selected_comparison.csv',summary);csvout(OUT/'development_cases.csv',case_results);csvout(OUT/'epoch_means.csv',averages)
gate_rows=[r for r in summary if r['objective']==chosen['objective']]
quality_gate=chosen['mean_top3']>=170 and chosen['mean_top1']>=106 and sum(r['top3']>164 for r in gate_rows)>=2 and min(r['top1'] for r in gate_rows)>=106
selection={**chosen,'representative_seed':cfg['seeds'][0],'quality_gate_passed':bool(quality_gate),'production_changed':False,
 'reserved_test_opened':False,'study_status':'Exploratory development selection; CPU export checks follow. No independent validation or calibrated confidence.',
 'training_covered':int((train['y']>=0).sum()),'development_covered':int((dev['y']>=0).sum())}
export=DATA/'export';export.mkdir(exist_ok=True)
def export_weights(directory,state):
 directory.mkdir(exist_ok=True);manifest={}
 for name,t in state.items():
  arr=t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t,np.float32)
  f=name.replace('.','_')+'.f32';arr.astype('<f4').tofile(directory/f);manifest[name]={'file':f,'shape':list(arr.shape),'sha256':sha(directory/f)}
 save(directory/'weights.json',manifest)
export_weights(export/'encoder',student.state_dict())
export_weights(export/'head',{**head_selected,'feature_mean':mean,'feature_scale':scale})
score_selected.astype('<f4').tofile(DATA/'selected_development_scores.f32')
selected_ids.astype('<i4').tofile(DATA/'selected_development_ids.i32')
save(OUT/'selection.json',selection)
print(json.dumps(selection,indent=2),flush=True)
