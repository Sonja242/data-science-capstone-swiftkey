"""Sonja Projects. Compact context and teacher-distillation pilot.

Author: Sonja Sahebzad. Run from the corpus project root. Public outputs contain
only aggregate results and hashed case IDs. Raw text and weights stay in data/.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys, time, copy, platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path.cwd()
OUT = ROOT/'final_project/research/student-context-20260925'
DATA = ROOT/'data/student_context'
sys.path.insert(0, str(ROOT/'python'))
SEED=20261007
torch.set_num_threads(4)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark=False

def readjson(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rs):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
VOCAB=rows(DATA/'vocabulary.csv') if (DATA/'vocabulary.csv').exists() else []
WORDS=[r['word'] for r in VOCAB]
LOOKUP={w:i for i,w in enumerate(WORDS)}

def inputs(phrases):
    x=np.zeros((len(phrases),48),np.int64);lens=np.empty(len(phrases),np.int64)
    for i,s in enumerate(phrases):
        words=s.split()[-48:] or ['<unknown>'];lens[i]=len(words)
        x[i,:len(words)]=[LOOKUP.get(w,-1)+2 if w in LOOKUP else 1 for w in words]
    return x,lens

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(50002,64,padding_idx=0)
        nn.init.normal_(self.embedding.weight,std=.08)
        with torch.no_grad():self.embedding.weight[0].zero_()
        self.gru=nn.GRU(64,96,batch_first=True)
        self.projection=nn.Linear(96,64)
        self.bias=nn.Parameter(torch.tensor(np.log(np.maximum([float(r['base_probability']) for r in VOCAB],1e-12)),dtype=torch.float32))
    def forward(self,x,lengths):
        hidden,_=self.gru(self.embedding(x))
        h=hidden[torch.arange(len(x),device=x.device),lengths-1]
        z=torch.tanh(self.projection(h))
        return F.linear(z,self.embedding.weight[2:],self.bias)

def teacher():
    from adaptation_experiment import AdaptedRanker
    fingerprint={'protocol_sha256':sha(OUT/'protocol.json'),
                 'code_sha256':sha(__file__),
                 'adapter_sha256':sha(ROOT/'models/neural/adaptation_local/adapter_model.safetensors'),
                 'vocabulary_sha256':sha(DATA/'vocabulary.csv')}
    if (OUT/'teacher_fingerprint.json').exists():
        assert readjson(OUT/'teacher_fingerprint.json')==fingerprint
    else: save(OUT/'teacher_fingerprint.json',fingerprint)
    ranker=AdaptedRanker('local256');ranker.shortlist=64
    ranker.check_scoring()
    torch.cuda.reset_peak_memory_stats()
    for split in ['teacher_training','development']:
        cases=rows(DATA/(split+'.csv'))
        path=DATA/('teacher_'+split+'.jsonl')
        done=[]
        if path.exists():
            done=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            assert all(d['line_hash']==c['line_hash'] for d,c in zip(done,cases))
        started=time.perf_counter()
        with path.open('a',encoding='utf-8') as f:
            for i in range(len(done),len(cases)):
                case=cases[i]
                # Actual next word never enters this candidate/scoring block.
                phrase=case['prefix'];ngram=json.loads(case['ngram_candidates_json'])
                torch.cuda.synchronize();begin=time.perf_counter()
                with torch.inference_mode():
                    prefix,prefill=ranker._prefill(phrase)
                    idx=ranker.eligible[torch.topk(prefill.logits[0,-1,ranker.eligible],64).indices].tolist()
                    words=list(dict.fromkeys([ranker.decoded[k].strip().lower() for k in idx]+ngram))
                    words=[w for w in words if w in LOOKUP]
                    assert len(words)>=32 and len(words)<=96
                    scores=ranker._score_candidates(prefix,prefill,words)
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-begin)
                order=sorted(words,key=lambda w:(-scores[w],w))
                result={'source':case['source'],'line_hash':case['line_hash'],
                        'word_ids':[LOOKUP[w] for w in order],
                        'scores':[scores[w] for w in order],'milliseconds':elapsed}
                f.write(json.dumps(result,separators=(',',':'))+'\n');f.flush()
                if (i+1)%100==0:print(f'Teacher {split}: {i+1}/{len(cases)}, {(time.perf_counter()-started):.0f}s this run',flush=True)
        details=[json.loads(line) for line in path.read_text().splitlines()]
        if split=='development':
            public=[]
            for c,d in zip(cases,details):
                wid=LOOKUP.get(c['actual'],-1)
                rank=d['word_ids'].index(wid)+1 if wid in d['word_ids'] else 0
                public.append({'source':c['source'],'line_hash':c['line_hash'],'rank':rank,
                               'milliseconds':d['milliseconds'],'candidate_count':len(d['word_ids'])})
            csvout(OUT/'teacher_development.csv',public)
    save(OUT/'teacher_cost.json',{'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,
        'training_cases':3000,'development_cases':600,
        'seconds_training_scoring':sum(json.loads(s)['milliseconds'] for s in (DATA/'teacher_teacher_training.jsonl').read_text().splitlines())/1000,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})

def prepare_arrays():
    if (DATA/'supervised.npz').exists():return
    rng=np.random.default_rng(20261006)
    train=rows(DATA/'training_lines.csv');xs=[];ys=[];lengths=[]
    for row in train:
        ww=row['text'].split()
        positions=rng.choice(np.arange(1,len(ww)),size=min(6,len(ww)-1),replace=False)
        for pos in positions:
            y=LOOKUP.get(ww[pos],-100)
            if y<0:continue
            seq=[LOOKUP[w]+2 if w in LOOKUP else 1 for w in ww[max(0,pos-48):pos]]
            xs.append(seq+[0]*(48-len(seq)));ys.append(y);lengths.append(len(seq))
    np.savez(DATA/'supervised.npz',x=np.asarray(xs,dtype=np.int32),y=np.asarray(ys,dtype=np.int64),lengths=np.asarray(lengths,np.int64))
    save(OUT/'training_sample.json',{'lines':len(train),'supervised_positions':len(ys),'max_context_words':48})

def evaluate_model(model,split='development'):
    cases=rows(DATA/(split+'.csv'));x,l=inputs([r['prefix'] for r in cases])
    ps=[]
    model.eval()
    with torch.inference_mode():
        for at in range(0,len(x),32):
            logits=model(torch.tensor(x[at:at+32],device='cuda'),torch.tensor(l[at:at+32],device='cuda'))
            ps.append(torch.softmax(logits.float(),-1).cpu().numpy())
    return np.concatenate(ps)

def train():
    prepare_arrays()
    a=np.load(DATA/'supervised.npz');n=len(a['y']);cost=[];curve=[]
    teacher_rows=[json.loads(s) for s in (DATA/'teacher_teacher_training.jsonl').read_text().splitlines()]
    cases=rows(DATA/'teacher_training.csv')
    assert [x['line_hash'] for x in teacher_rows]==[x['line_hash'] for x in cases]
    tx,tl=inputs([r['prefix'] for r in cases]);ty=np.array([LOOKUP.get(r['actual'],-100) for r in cases])
    ti=np.zeros((len(cases),96),np.int64);tp=np.zeros((len(cases),96),np.float32)
    for i,r in enumerate(teacher_rows):
        logits=np.array(r['scores'])/2;prob=np.exp(logits-logits.max());prob/=prob.sum()
        ti[i,:len(prob)]=r['word_ids'];tp[i,:len(prob)]=prob
    model=Student().cuda();scaler=torch.amp.GradScaler('cuda')
    initial=DATA/'supervised_initial.pt'
    if not initial.exists():
        opt=torch.optim.AdamW(model.parameters(),lr=.002)
        start=time.perf_counter();rng=np.random.default_rng(SEED)
        for epoch in range(2):
            model.train();order=rng.permutation(n);tot=0.;steps=0
            for at in range(0,n,256):
                ix=order[at:at+256]
                x=torch.tensor(a['x'][ix],dtype=torch.long,device='cuda');l=torch.tensor(a['lengths'][ix],device='cuda');y=torch.tensor(a['y'][ix],device='cuda')
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(x,l);loss=F.cross_entropy(logits,y)
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(opt);scaler.update();tot+=float(loss.detach());steps+=1
                if steps%200==0:print(f'Pretrain epoch{epoch+1}: {at+len(ix)}/{n}, loss{tot/steps:.3f}, elapsed{time.perf_counter()-start:.0f}s',flush=True)
            curve.append({'phase':'supervised_pretrain','alpha':0.,'epoch':epoch+1,'training_loss':tot/steps})
        torch.cuda.synchronize();cost.append({'phase':'pretrain','seconds':time.perf_counter()-start,'examples':n*2})
        torch.save(model.state_dict(),initial)
        csvout(OUT/'pretraining_curve.csv',curve)
        save(OUT/'pretraining_cost.json',cost[-1])
    base=torch.load(initial,map_location='cpu',weights_only=True)
    for name,alpha in [('CE_control',0.),('Distill25',.25),('Distill50',.5)]:
        torch.manual_seed(SEED+1);model=Student().cuda();model.load_state_dict(base)
        ckpt=DATA/(name+'.pt')
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True))
        else:
            opt=torch.optim.AdamW(model.parameters(),lr=.0005)
            rng=np.random.default_rng(SEED+1);start=time.perf_counter()
            for epoch in range(5):
                model.train();order=rng.permutation(len(tx));tot=0.;steps=0
                for at in range(0,len(tx),256):
                    ix=order[at:at+256]
                    x=torch.tensor(tx[ix],device='cuda');l=torch.tensor(tl[ix],device='cuda');y=torch.tensor(ty[ix],device='cuda')
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        logits=model(x,l);ce=F.cross_entropy(logits,y,ignore_index=-100)
                        if alpha:
                            logp=F.log_softmax(logits.float()/2,-1)
                            soft=-(logp.gather(1,torch.tensor(ti[ix],device='cuda'))*torch.tensor(tp[ix],device='cuda')).sum(1).mean()*4
                            loss=(1-alpha)*ce+alpha*soft
                        else:loss=ce
                    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    scaler.step(opt);scaler.update();tot+=float(loss.detach());steps+=1
                curve.append({'phase':name,'alpha':alpha,'epoch':epoch+1,'training_loss':tot/steps})
                print(f'{name} epoch{epoch+1}: loss{tot/steps:.3f}',flush=True)
            torch.cuda.synchronize();cost.append({'phase':name,'seconds':time.perf_counter()-start,'examples':len(tx)*5})
            torch.save(model.state_dict(),ckpt)
        prob=evaluate_model(model)
        prob.astype('<f4').tofile(DATA/(name+'_dev.f32'))
        export=DATA/name;export.mkdir(exist_ok=True)
        manifest={}
        for k,t in model.state_dict().items():
            arr=t.detach().cpu().float().numpy();namefile=k.replace('.','_')+'.f32'
            arr.astype('<f4').tofile(export/namefile)
            manifest[k]={'file':namefile,'shape':list(arr.shape),'sha256':sha(export/namefile)}
        save(export/'weights.json',manifest)
        save(OUT/(name+'_model.json'),{'parameters':sum(p.numel() for p in model.parameters()),
            'parameter_mib_fp32':sum(p.numel()*4 for p in model.parameters())/2**20,
            'checkpoint_sha256':sha(ckpt),'distillation_weight':alpha,'context_words':48})
    csvout(OUT/'finetuning_curve.csv',[r for r in curve if r['phase']!='supervised_pretrain'])
    save(OUT/'training_cost.json',cost)
    save(OUT/'python-environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'seed':SEED,
        'study_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
        'private_input_sha256':{p.name:sha(p) for p in [DATA/'supervised.npz',DATA/'teacher_teacher_training.jsonl',DATA/'training_lines.csv']}})
    compare()

def compare():
    cases=rows(DATA/'development.csv');actual=np.array([LOOKUP.get(r['actual'],-1) for r in cases]);allrows=[];details=[]
    base=np.fromfile(DATA/'ngram_dev.f32',dtype='<f4').reshape(600,50000)
    candidates=[('Current',base)]
    for name in ['CE_control','Distill25','Distill50']:
        p=np.fromfile(DATA/(name+'_dev.f32'),dtype='<f4').reshape(600,50000)
        assert np.allclose(p.sum(1),1,atol=1e-5)
        for weight in [.1,.25,.5,1.]:candidates.append((name+'_'+str(weight),base*(1-weight)+p*weight))
    for name,p in candidates:
        correct1=correct3=0
        for i,row in enumerate(p):
            ids=np.argpartition(-row,3)[:3];ids=sorted(ids,key=lambda k:(-row[k],WORDS[k]))
            rank=ids.index(actual[i])+1 if actual[i] in ids else 0
            correct1+=rank==1;correct3+=rank>0
            details.append({'name':name,'source':cases[i]['source'],'line_hash':cases[i]['line_hash'],'rank':rank})
        allrows.append({'name':name,'cases':600,'top1':correct1,'top3':correct3})
    csvout(OUT/'python_development_comparison.csv',allrows)
    csvout(OUT/'python_development_cases.csv',details)
    print(json.dumps(allrows,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['teacher','train','compare']);args=parser.parse_args()
    assert (OUT/'protocol.json').exists() and len(WORDS)==50000
    if args.phase=='teacher':teacher()
    elif args.phase=='train':train()
    else:compare()
