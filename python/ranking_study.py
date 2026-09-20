"""Controlled ranking ablations and separate confidence calibration.

Author: Sonja Sahebzad. Candidate generation never receives the observed word.
"""
import argparse
import copy
import csv
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from adaptation_experiment import AdaptedRanker
from neural_predictor import ROOT,WORD
from learning_curve_experiment import sha,code_sha,read_json,inference_fingerprint
from ranking_rules import VARIANTS,COLUMNS,prediction,metrics,fit_sigmoid,calibrate,confidence_metrics,wilson

P=ROOT/'models/ranking_study_protocol.json'


def save(name,value):
    with (ROOT/'models'/('ranking_study_'+name+'.json')).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,separators=(',',':'),allow_nan=False)


def load(name):return read_json(ROOT/'models'/('ranking_study_'+name+'.json'))


def fingerprint():
    old=read_json(ROOT/'models/word_generation_implementation.json')['fingerprint']
    assert inference_fingerprint()==old['inference_inputs_sha256']
    assert code_sha(ROOT/'python/adaptation_experiment.py')==old['scorer_canonical_sha256']
    assert sha(ROOT/'python/neural_predictor.py')==old['base_code_sha256']
    return {'study_code':code_sha(__file__),'rules_code':code_sha(ROOT/'python/ranking_rules.py'),
        'inference_inputs':old['inference_inputs_sha256'],'scorer_code':old['scorer_canonical_sha256'],
        'prior_development_sha256':sha(ROOT/'models/word_generation_development.json'),
        'manifest_sha256':sha(ROOT/'models/ranking_study_data_manifest.json')}


def frozen():
    p=load('protocol')
    assert p['fingerprint']==fingerprint()
    return p


def cases(split):
    p=load('protocol');m=load('data_manifest')
    file=ROOT/('data/adaptation/development.csv' if split=='development' else 'data/ranking_study/'+split+'.csv')
    assert sha(file)==m[split+'_csv_sha256']
    with file.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
    assert len(r)==p['cases'][split]
    return r


class RankStudy(AdaptedRanker):
    def __init__(self):super().__init__('local256')

    def candidate_words(self,prefill,ngram):
        ids=self.eligible[self.torch.topk(prefill.logits[0,-1,self.eligible],256).indices].tolist()
        return [w for w in dict.fromkeys([self.decoded[i].strip().lower() for i in ids]+list(ngram)) if WORD.fullmatch(w)]

    def components(self,prefix,prefill,names,sequences):
        t=self.torch
        initial=t.log_softmax(prefill.logits[0,-1].float(),dim=-1)
        values={}
        for start in range(0,len(names),self.batch_size):
            words=names[start:start+self.batch_size];seqs=sequences[start:start+self.batch_size]
            lengths=t.tensor([len(s) for s in seqs],device='cuda');width=max(map(len,seqs));n=len(words)
            padded=np.full((n,width),self.tokenizer.eos_token_id,dtype=np.int64)
            for i,s in enumerate(seqs):padded[i,:len(s)]=s
            ids=t.from_numpy(padded).to('cuda')
            mask=(t.arange(width,device='cuda')[None,:]<lengths[:,None]).long()
            cache=copy.deepcopy(prefill.past_key_values);cache.batch_repeat_interleave(n)
            attention=t.cat((t.ones((n,len(prefix)),device='cuda',dtype=t.long),mask),dim=1)
            out=self.model(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True,logits_to_keep=0)
            logits=out.logits.float();normalizers=t.logsumexp(logits,dim=-1)
            token=initial[ids[:,0]]
            if width>1:
                part=logits[:,:-1,:].gather(2,ids[:,1:,None]).squeeze(-1)-normalizers[:,:-1]
                token=token+(part*mask[:,1:]).sum(dim=1)
            row=t.arange(n,device='cuda');last=lengths-1
            last_logits=logits[row,last,:]
            boundary_sum=t.logsumexp(last_logits[:,self.boundary],dim=-1)
            boundary=boundary_sum-normalizers[row,last]
            # Preserve the exact sequence of the archived FP16 scorer operations.
            total=token+boundary_sum-normalizers[row,last]
            combined=t.stack((token,boundary,total),dim=1).cpu().tolist()
            values.update(zip(words,combined))
            del out,cache,logits,normalizers,last_logits
        return values

    def features(self,phrase,ngram,alternate=False):
        with self.torch.inference_mode():
            prefix,prefill=self._prefill(phrase)
            words=self.candidate_words(prefill,ngram)
            canonical=[self._tokens(w) for w in words]
            values=self.components(prefix,prefill,words,canonical)
            alt={};different={}
            if alternate:
                space=self.tokenizer.encode(' ',add_special_tokens=False)
                for w,seq in zip(words,canonical):
                    other=space+self.tokenizer.encode(w,add_special_tokens=False)
                    assert self.tokenizer.decode(other,clean_up_tokenization_spaces=False)==' '+w
                    if other!=seq:different[w]=other
                alt=self.components(prefix,prefill,list(different),list(different.values()))
            return [[w,*values[w],len(seq),len(w),alt[w][2] if w in alt else values[w][2],w in alt]
                    for w,seq in zip(words,canonical)]


def freeze():
    if P.exists():raise FileExistsError('Keep the existing ranking registration')
    m=load('data_manifest')
    p={'author':'Sonja Sahebzad','registered_utc':datetime.now(timezone.utc).isoformat(),
        'fingerprint':fingerprint(),'variants':list(VARIANTS),'columns':list(COLUMNS),
        'cases':{'development':600,'calibration':900,'final':900},
        'fixed':'Operational local256 merged FP16 model; 128-token context; identical target-blind eligible256 plus R20 candidate pool',
        'rules':{'baseline':'Original canonical log word likelihood plus boundary mass, archived operation order',
            'no_boundary':'Canonical token log likelihood only', 'half_boundary':'Token log likelihood +0.5*boundary log mass',
            'token_mean':'Token log likelihood/token count + boundary log mass',
            'token_mean_no_boundary':'Token log likelihood/token count; planned boundary-by-length interaction',
            'char_sqrt':'Token log likelihood/sqrt(character count) + boundary log mass',
            'two_paths':'Log-sum-exp of canonical complete-word score and distinct space-token+word-token score; exact same decoded surface, not all segmentations'},
        'selection':'Development integer top3, then integer top1, then registered variant order favoring baseline; no outcome-dependent extra variants',
        'calibration':'First-suggestion correctness only. Fixed penalized logistic map of logit shortlist softmax share: sum binary log loss +0.5*slope^2. Fit separately for baseline and frozen selected variant on900 new validation examples, including missed targets. No ranking is trained.',
        'thresholds':[0.,.5,.7,.85,.95],
        'final':'900 untouched examples after variant and confidence map are frozen. Accuracy over all cases; selected-subset top1/top3 and coverage; Wilson95 intervals for answered cases; Brier, log loss,10-bin reliability/ECE. Thresholds not optimized on final.',
        'promotion':'Only ranking promotion allowed after primary top3 paired source-stratified bootstrap95 lower>0 and exact two-sided McNemar p<.05. Confidence remains research-only pending calibration assessment; no automatic95% guarantee.',
        'bootstrap_seed':20261690,'bootstrap_repetitions':10000,
        'timing':'Standalone original versus selected on first20 final cases/source, after loading, alternating order, CUDA sync. Excludes R shortlist generation, model loading and choices. Match recorded outputs exactly.',
        'numerical':'Canonical extraction must match every archived development score exactly. Neutral alternative/full-sequence FP16 check<=0.06; report observed error. All inference remains FP16.',
        'limitations':'Reused development set; fixed corpus and candidate pool; incomplete segmentation marginalization; unknown pretraining overlap; no fitted reranker or new base model in this study.'}
    save('protocol',p)
    ranker=RankStudy();checks=[]
    with ranker.torch.inference_mode():
        for phrase in ('we are talking about','the scientific explanation is'):
            prefix,prefill=ranker._prefill(phrase)
            words=['the','science','uncharacteristically']
            canonical=[ranker._tokens(w) for w in words]
            comp=ranker.components(prefix,prefill,words,canonical)
            old=ranker._score_candidates(prefix,prefill,words)
            assert all(comp[w][2]==old[w] for w in words)
            seqs=[ranker.tokenizer.encode(' ',add_special_tokens=False)+ranker.tokenizer.encode(w,add_special_tokens=False) for w in words]
            alt=ranker.components(prefix,prefill,words,seqs)
            for w,seq in zip(words,seqs):
                ids=ranker.torch.tensor([prefix+seq],device='cuda')
                out=ranker.model(input_ids=ids,attention_mask=ranker.torch.ones_like(ids),use_cache=False)
                lp=ranker.torch.log_softmax(out.logits[0].float(),dim=-1)
                direct=sum(lp[len(prefix)-1+i,k] for i,k in enumerate(seq))+ranker.torch.logsumexp(lp[-1,ranker.boundary],dim=-1)
                error=abs(float(direct)-alt[w][2]);assert error<=.06,(w,error)
                checks.append({'phrase':phrase,'word':w,'canonical_identical':True,'alternate_formula_error':error})
    save('neutral_checks',{'passed':True,'checks':checks,'protocol_sha256':sha(P)})
    print('Registered seven variants; neutral scoring checks passed',flush=True)


def evaluate(split):
    p=frozen();assert load('neutral_checks')['passed']
    path=ROOT/'models'/f'ranking_study_{split}.json'
    if path.exists():raise FileExistsError(path)
    if split=='development':variants=VARIANTS
    else:
        selection=load('selection');assert selection['development_sha256']==sha(ROOT/'models/ranking_study_development.json')
        variants=tuple(dict.fromkeys(('baseline',selection['variant'])))
        if split=='final':assert load('calibrators')['selection_sha256']==sha(ROOT/'models/ranking_study_selection.json')
    ranker=RankStudy();details=[];source_cases=cases(split)
    previous=read_json(ROOT/'models/word_generation_development.json')['details'] if split=='development' else None
    started=time.perf_counter()
    for i,case in enumerate(source_cases):
        features=ranker.features(case['prefix'],json.loads(case['ngram_candidates_json']),'two_paths' in variants)
        if previous:
            old=previous[i]
            assert old['line_hash']==case['line_hash']
            assert [[r[0],r[3]] for r in features]==old['baseline']['scores'],'Archived baseline extraction differs'
        predictions={v:prediction(features,v) for v in variants}
        details.append({'source':case['source'],'line_hash':case['line_hash'],'actual':case['actual'],
            'target_tokens':len(ranker._tokens(case['actual'])),'covered':case['actual'] in {r[0] for r in features},
            'features':features,'predictions':predictions})
        if (i+1)%100==0:print(f'{split} ranking cases {i+1}/{len(source_cases)}',flush=True)
    artifact={'split':split,'columns':list(COLUMNS),'protocol_sha256':sha(P),'details':details,
        'summaries':{v:metrics(details,v) for v in variants},'joint_collection_seconds':time.perf_counter()-started}
    if split!='development':artifact['selection_sha256']=sha(ROOT/'models/ranking_study_selection.json')
    if split=='final':
        artifact['calibrators_sha256']=sha(ROOT/'models/ranking_study_calibrators.json')
        timing={v:[] for v in variants};seen={s:0 for s in ('blogs','news','twitter')}
        for i,(case,d) in enumerate(zip(source_cases,details)):
            if seen[case['source']]>=20:continue
            j=sum(seen.values());seen[case['source']]+=1
            order=list(variants)[j%len(variants):]+list(variants)[:j%len(variants)]
            for v in order:
                ranker.torch.cuda.synchronize();t0=time.perf_counter()
                f=ranker.features(case['prefix'],json.loads(case['ngram_candidates_json']),v=='two_paths')
                pred=prediction(f,v)
                ranker.torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-t0)
                assert pred==d['predictions'][v]
                timing[v].append({'source':case['source'],'line_hash':case['line_hash'],'milliseconds':elapsed})
        artifact['timings']=timing
    save(split,artifact)
    print(json.dumps(artifact['summaries'],indent=2),flush=True)


def select():
    frozen();dev=load('development');assert dev['protocol_sha256']==sha(P)
    selected=min(VARIANTS,key=lambda v:(-sum(d['actual'] in d['predictions'][v]['words'] for d in dev['details']),
        -sum(d['actual']==d['predictions'][v]['words'][0] for d in dev['details']),VARIANTS.index(v)))
    save('selection',{'variant':selected,'selected_utc':datetime.now(timezone.utc).isoformat(),
        'development_sha256':sha(ROOT/'models/ranking_study_development.json'),'protocol_sha256':sha(P)})
    print('Development choice frozen:',selected,flush=True)


def fit():
    frozen();data=load('calibration');selected=load('selection')
    assert data['selection_sha256']==sha(ROOT/'models/ranking_study_selection.json')
    models={}
    for v in data['summaries']:
        raw=[d['predictions'][v]['raw_shortlist_share'] for d in data['details']]
        y=[d['actual']==d['predictions'][v]['words'][0] for d in data['details']]
        models[v]=fit_sigmoid(raw,y)
    save('calibrators',{'models':models,'event':'First suggestion equals observed next word',
        'selection_sha256':sha(ROOT/'models/ranking_study_selection.json'),
        'calibration_sha256':sha(ROOT/'models/ranking_study_calibration.json'),
        'fitted_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(P)})
    print(json.dumps(models,indent=2),flush=True)


def summarize():
    p=frozen();test=load('final');dev=load('development');selection=load('selection');cal=load('calibrators')
    assert test['calibrators_sha256']==sha(ROOT/'models/ranking_study_calibrators.json')
    v=selection['variant'];details=test['details']
    from learning_curve_audit import confirmatory_comparison
    def comparison_arm(name):
        rows=[]
        for d in details:
            words=d['predictions'][name]['words'];r=words.index(d['actual'])+1 if d['actual'] in words else 0
            rows.append({'source':d['source'],'line_hash':d['line_hash'],'rank':r,'choice_correct':False,'shortlist_contains_target':d['covered']})
        return {'summary':test['summaries'][name],'details':rows}
    comparison=confirmatory_comparison(comparison_arm('baseline'),comparison_arm(v),bootstrap_seed=p['bootstrap_seed'])['primary_top3']
    confidence=[];selective=[];cases_csv=[];groups=[]
    for name,summary in test['summaries'].items():
        preds=[d['predictions'][name] for d in details]
        y=np.array([d['actual']==q['words'][0] for d,q in zip(details,preds)])
        y3=np.array([d['actual'] in q['words'] for d,q in zip(details,preds)])
        raw=np.array([q['raw_shortlist_share'] for q in preds]);adjusted=calibrate(raw,cal['models'][name])
        for kind,prob in (('raw_shortlist_share',raw),('calibrated',adjusted)):
            confidence.append({'variant':name,'kind':kind,**confidence_metrics(y,prob)})
        for threshold in p['thresholds']:
            keep=adjusted>=threshold;n=int(keep.sum());k=int(y[keep].sum())
            selective.append({'variant':name,'threshold':threshold,'answered':n,'total':len(y),'coverage':n/len(y),
                'correct_first':k,'accuracy_first':k/n if n else None,'accuracy_top3':float(y3[keep].mean()) if n else None,
                'wilson95':wilson(k,n)})
        for d,q,prob,correct,correct3 in zip(details,preds,adjusted,y,y3):
            cases_csv.append({'variant':name,'source':d['source'],'line_hash':d['line_hash'],'actual':d['actual'],
                'first':q['words'][0],'second':q['words'][1],'third':q['words'][2],'correct_first':bool(correct),
                'correct_top3':bool(correct3),'covered':d['covered'],'raw_shortlist_share':q['raw_shortlist_share'],
                'calibrated_first_probability':float(prob)})
    for split,data in (('development',dev),('final',test)):
        for name in data['summaries']:
            for group in ('single_token','multiple_tokens','blogs','news','twitter'):
                ds=[d for d in data['details'] if (group=='single_token' and d['target_tokens']==1) or
                    (group=='multiple_tokens' and d['target_tokens']>1) or d['source']==group]
                groups.append({'split':split,'variant':name,'group':group,**metrics(ds,name)})
    paired=[]
    for name in dev['summaries']:
        wins=sum(d['actual'] not in d['predictions']['baseline']['words'] and d['actual'] in d['predictions'][name]['words'] for d in dev['details'])
        losses=sum(d['actual'] in d['predictions']['baseline']['words'] and d['actual'] not in d['predictions'][name]['words'] for d in dev['details'])
        paired.append({'variant':name,'gains':wins,'losses':losses})
    outcome={'author':'Sonja Sahebzad','selected':v,'primary':comparison,'promote':comparison['promote'],
        'development':[{'variant':k,**z} for k,z in dev['summaries'].items()],
        'development_paired':paired,'final':[{'variant':k,**z} for k,z in test['summaries'].items()],
        'confidence':confidence,'selective':selective,'groups':groups,
        'timings':[{'variant':k,'cases':len(z),'mean_ms':float(np.mean([x['milliseconds'] for x in z]))} for k,z in test['timings'].items()],
        'input_sha256':{name:sha(ROOT/'models'/('ranking_study_'+name+'.json')) for name in
            ('protocol','development','selection','calibration','calibrators','final')},
        'confidence_not_a_guarantee':True}
    save('summary',outcome)
    with (ROOT/'models/ranking_study_final_cases.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases_csv[0]));w.writeheader();w.writerows(cases_csv)
    print(json.dumps({'selected':v,'primary':comparison,'final':outcome['final'],'selective':selective},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--evaluate',choices=('development','calibration','final'))
    g.add_argument('--select',action='store_true');g.add_argument('--fit-calibration',action='store_true');g.add_argument('--summarize',action='store_true')
    args=a.parse_args()
    if args.freeze:freeze()
    elif args.evaluate:evaluate(args.evaluate)
    elif args.select:select()
    elif args.fit_calibration:fit()
    else:summarize()
