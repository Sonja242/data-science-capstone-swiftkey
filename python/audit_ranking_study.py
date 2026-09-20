"""Independent CPU reconstruction of ranks, confidence and paired decisions."""
import argparse,csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'models'
ORDER=('baseline','no_boundary','half_boundary','token_mean','token_mean_no_boundary','char_sqrt','two_paths')

def load(name):return json.loads((M/('ranking_study_'+name+'.json')).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(p):return hashlib.sha256(Path(p).read_text(encoding='utf-8-sig').encode()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-10),(a,b)

def reconstruct(rows,name):
    data={}
    for w,l,b,t,n,c,alt,distinct in rows:
        assert n>=1 and c==len(w) and math.isfinite(t)
        value={'baseline':t,'no_boundary':l,'half_boundary':l+b/2,
            'token_mean':l/n+b,'token_mean_no_boundary':l/n,'char_sqrt':l/math.sqrt(c)+b}
        maximum=max(t,alt)
        value['two_paths']=maximum+math.log(math.exp(t-maximum)+math.exp(alt-maximum)) if distinct else t
        data[w]=value[name]
    assert len(data)==len(rows)
    ordered=sorted(data,key=lambda w:(-data[w],w))
    top=data[ordered[0]]
    share=1/sum(math.exp(x-top) for x in data.values())
    return ordered[:3],share,top,top-data[ordered[1]]

def inspect(split):
    artifact=load(split);manifest=load('data_manifest');protocol=load('protocol')
    path=ROOT/('data/adaptation/development.csv' if split=='development' else 'data/ranking_study/'+split+'.csv')
    assert sha(path)==manifest[split+'_csv_sha256']
    with path.open(encoding='utf-8-sig',newline='') as f:cases=list(csv.DictReader(f))
    assert artifact['protocol_sha256']==sha(M/'ranking_study_protocol.json')
    assert artifact['split']==split and len(cases)==len(artifact['details'])==protocol['cases'][split]
    assert Counter(r['source'] for r in cases)=={s:len(cases)//3 for s in ('blogs','news','twitter')}
    names=list(artifact['summaries'])
    if split=='development':assert names==list(ORDER)
    else:assert names==list(dict.fromkeys(('baseline',load('selection')['variant'])))
    previous=json.loads((M/'word_generation_development.json').read_text())['details'] if split=='development' else None
    results={name:[] for name in names}
    for i,(case,d) in enumerate(zip(cases,artifact['details'])):
        assert (case['source'],case['line_hash'],case['actual'])==(d['source'],d['line_hash'],d['actual'])
        pool={r[0] for r in d['features']}
        assert d['covered']==(d['actual'] in pool)
        if previous:
            assert previous[i]['line_hash']==d['line_hash']
            assert [[r[0],r[3]] for r in d['features']]==previous[i]['baseline']['scores']
        for name in names:
            words,share,score,gap=reconstruct(d['features'],name)
            pred=d['predictions'][name]
            assert words==pred['words'];close(share,pred['raw_shortlist_share']);close(score,pred['top_score']);close(gap,pred['score_gap'])
            results[name].append(words.index(d['actual'])+1 if d['actual'] in words else 0)
    for name,ranks in results.items():
        summary=artifact['summaries'][name];n=len(ranks)
        for key,value in {'cases':n,'top1':sum(r==1 for r in ranks)/n,'top3':sum(r>0 for r in ranks)/n,
            'mrr_at3':sum(1/r if r else 0 for r in ranks)/n,'coverage':sum(d['covered'] for d in artifact['details'])/n}.items():close(summary[key],value)
    return artifact,results

def main(final=False):
    protocol=load('protocol');fp=protocol['fingerprint']
    assert fp['study_code']==canonical(ROOT/'python/ranking_study.py')
    assert fp['rules_code']==canonical(ROOT/'python/ranking_rules.py')
    dev,devr=inspect('development')
    chosen=min(ORDER,key=lambda n:(-sum(x>0 for x in devr[n]),-sum(x==1 for x in devr[n]),ORDER.index(n)))
    result={'author':'Sonja Sahebzad','passed':True,'selected_recomputed':chosen,'development_cases':600,
        'all_development_candidates_and_baseline_scores_unchanged':True}
    if final:
        selection=load('selection');caldata,calr=inspect('calibration');test,testr=inspect('final');fit=load('calibrators');summary=load('summary')
        assert selection['variant']==chosen and summary['selected']==chosen
        assert selection['development_sha256']==sha(M/'ranking_study_development.json')
        assert fit['calibration_sha256']==sha(M/'ranking_study_calibration.json')
        assert fit['selection_sha256']==sha(M/'ranking_study_selection.json')
        assert test['calibrators_sha256']==sha(M/'ranking_study_calibrators.json')
        assert caldata['selection_sha256']==test['selection_sha256']==fit['selection_sha256']
        for name,h in summary['input_sha256'].items():assert sha(M/('ranking_study_'+name+'.json'))==h
        assert set(d['line_hash'] for d in caldata['details']).isdisjoint(d['line_hash'] for d in test['details'])
        assert load('partition_audit')['passed']
        a=np.array([r>0 for r in testr['baseline']],dtype=int);b=np.array([r>0 for r in testr[chosen]],dtype=int)
        delta=b-a;gains=int((delta==1).sum());losses=int((delta==-1).sum());n=gains+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/(2**n)) if n else 1.
        rng=np.random.default_rng(protocol['bootstrap_seed']);boot=np.zeros(10000)
        for source in ('blogs','news','twitter'):
            group=delta[[i for i,d in enumerate(test['details']) if d['source']==source]]
            for start in range(0,10000,500):boot[start:start+500]+=rng.choice(group,(500,len(group)),replace=True).mean(axis=1)/3
        interval=np.quantile(boot,[.025,.975]).tolist()
        close(summary['primary']['difference'],delta.mean());close(summary['primary']['exact_paired_p'],p)
        for x,y in zip(interval,summary['primary']['interval']):close(x,y)
        assert summary['promote']==summary['primary']['promote']==(interval[0]>0 and p<.05)
        checks=[]
        for name,model in fit['models'].items():
            raw=np.clip([d['predictions'][name]['raw_shortlist_share'] for d in caldata['details']],1e-6,1-1e-6)
            x=np.column_stack((np.ones(len(raw)),np.log(raw/(1-raw))))
            beta=np.array([model['intercept'],model['slope']]);y=np.array([r==1 for r in calr[name]])
            predicted=1/(1+np.exp(-(x@beta)));gradient=x.T@(predicted-y)+[0.,beta[1]]
            assert np.max(np.abs(gradient))<1e-6
            y=np.array([r==1 for r in testr[name]]);y3=np.array([r>0 for r in testr[name]])
            raw=np.array([d['predictions'][name]['raw_shortlist_share'] for d in test['details']])
            clipped=np.clip(raw,1e-6,1-1e-6)
            calibrated=1/(1+np.exp(-(beta[0]+beta[1]*np.log(clipped/(1-clipped)))))
            for kind,prob in (('raw_shortlist_share',raw),('calibrated',calibrated)):
                row=next(x for x in summary['confidence'] if x['variant']==name and x['kind']==kind)
                bounded=np.clip(prob,1e-12,1-1e-12)
                close(row['brier'],np.mean((bounded-y)**2));close(row['log_loss'],-np.mean(y*np.log(bounded)+(1-y)*np.log1p(-bounded)))
                ece=0
                for i,r in enumerate(row['reliability']):
                    mask=(bounded>=i/10)&((bounded<(i+1)/10) if i<9 else bounded<=1)
                    assert r['cases']==int(mask.sum())
                    if mask.any():
                        close(r['mean_probability'],bounded[mask].mean());close(r['observed_accuracy'],y[mask].mean())
                        ece+=mask.sum()*abs(bounded[mask].mean()-y[mask].mean())/len(y)
                close(row['ece_10_equal_width'],ece)
            for row in (r for r in summary['selective'] if r['variant']==name):
                keep=calibrated>=row['threshold'];n=int(keep.sum());k=int(y[keep].sum())
                assert row['answered']==n and row['correct_first']==k and row['total']==900
                close(row['coverage'],n/900)
                if n:
                    close(row['accuracy_first'],k/n);close(row['accuracy_top3'],y3[keep].mean())
                    z=1.959963984540054;center=(k/n+z*z/(2*n))/(1+z*z/n)
                    radius=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
                    for x,v in zip(row['wilson95'],[max(0.,center-radius),min(1.,center+radius)]):close(x,v)
                else:assert row['accuracy_first'] is None and row['wilson95']==[None,None]
            checks.append({'variant':name,'calibrator_max_abs_gradient':float(np.abs(gradient).max()),'final_confidence_recomputed':True})
        for timing in summary['timings']:
            rows=test['timings'][timing['variant']]
            assert len(rows)==timing['cases']==60 and Counter(r['source'] for r in rows)=={s:20 for s in ('blogs','news','twitter')}
            close(timing['mean_ms'],sum(r['milliseconds'] for r in rows)/60)
        result.update({'final_cases':900,'calibration_cases':900,'gains':gains,'losses':losses,'paired_interval':interval,'exact_p':p,'promotion_passed':summary['promote'],'calibration_checks':checks,
            'summary_sha256':sha(M/'ranking_study_summary.json')})
    result['audit_code_canonical_sha256']=canonical(__file__)
    path=M/('ranking_study_independent_audit.json' if final else 'ranking_study_development_audit.json')
    if path.exists():assert json.loads(path.read_text())==result
    else:
        with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args();main(args.final)
