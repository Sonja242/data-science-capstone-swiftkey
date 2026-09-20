"""Independent CPU recomputation of saved diagnostic ranks. Sonja Sahebzad."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'models'


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check():
    raw=load(M/'word_generation_development.json')
    first=load(M/'target_rank_fp16.json')
    second=load(M/'target_rank_fp32.json')
    summary=load(M/'target_rank_summary.json')
    protocol=load(M/'target_rank_protocol.json')
    assert raw['split']=='development' and protocol['no_final_files_opened']
    assert first['protocol_sha256']==second['protocol_sha256']==sha(M/'target_rank_protocol.json')
    assert len(first['details'])==len(raw['details'])==600
    assert summary['diagnostic_not_predictive_performance']
    for name,h in summary['input_sha256'].items():assert sha(M/name)==h
    for name,h in protocol['input_sha256'].items():assert sha(ROOT/name)==h
    assert hashlib.sha256((ROOT/'python/target_rank_diagnostic.py').read_text(encoding='utf-8-sig').encode()).hexdigest()==protocol['analysis_canonical_sha256']
    checked={r['line_hash']:r for r in second['details']}
    assert len(checked)==len(second['details'])
    counts={arm:Counter() for arm in ('baseline','beam16')}
    selected=[];changes=[];computed=[];closest=[]
    for saved,row in zip(raw['details'],first['details']):
        target=saved['actual']
        assert (saved['source'],saved['line_hash'],target)==(row['source'],row['line_hash'],row['target'])
        assert row['target_tokens']==saved['target_tokens']
        base=dict(saved['baseline']['scores'])
        union={**base,**dict(saved['candidates']['beam16']['scores'])}
        why=[]
        for arm,pool in (('baseline',base),('beam16',union)):
            source_scores=dict(pool)
            present=target in pool
            if not present:pool[target]=row['injected_fp16_score']
            ordered=sorted(pool,key=lambda w:(-pool[w],w))
            pos=ordered.index(target)+1
            other=[w for w in ordered if w!=target]
            cutoff=other[2]
            category=('already_top3' if pos<=3 else 'covered_below_top3') if present else ('missing_reaches_top3' if pos<=3 else 'missing_still_below_top3')
            x=row['arms'][arm]
            assert x['diagnostic_rank']==pos and x['target_present']==present and x['category']==category
            assert x['original_rank']==(pos if present else None)
            assert x['target_score']==pool[target] and x['third_other_word']==cutoff and x['third_other_score']==pool[cutoff]
            assert x['margin_to_top3']==pool[target]-pool[cutoff]
            assert x['candidate_count']==len(source_scores) and x['diagnostic_candidate_count']==len(pool)
            counts[arm][category]+=1
            counts[arm]['diagnostic_top1']+=pos==1
            counts[arm]['diagnostic_top3']+=pos<=3
            if abs(x['margin_to_top3'])<=protocol['near_top3_log_margin']:why.append(arm+':near_top3')
            if category=='missing_reaches_top3':why.append(arm+':potential_recovery')
            if arm=='baseline' and not present:closest.append({'word':target,'rank':pos,'margin':x['margin_to_top3']})
            if row['line_hash'] in checked:
                item=checked[row['line_hash']];scores=dict(item['scores'])
                pool32={w:scores[w] for w in pool}
                order32=sorted(pool32,key=lambda w:(-pool32[w],w));pos32=order32.index(target)+1
                r32=item['arms'][arm]
                assert r32['diagnostic_rank']==pos32
                cutoff32=[w for w in order32 if w!=target][2]
                assert r32['third_other_word']==cutoff32
                assert r32['margin_to_top3']==scores[target]-scores[cutoff32]
                counts[arm]['fp32_checked']+=1
                if (pos<=3)!=(pos32<=3):
                    counts[arm]['membership_changes_in_checked_subset']+=1
                    changes.append({'arm':arm,'source':row['source'],'target':target,'line_hash':row['line_hash'],
                        'present_originally':present,'fp16_rank':pos,'fp32_rank':pos32,
                        'fp16_margin':x['margin_to_top3'],'fp32_margin':r32['margin_to_top3']})
                counts[arm]['recoveries_confirmed_fp32']+=category=='missing_reaches_top3' and pos32<=3
            computed.append((arm,row['line_hash'],pos))
        assert why==row['fp32_reasons']
        if why:selected.append(row['line_hash'])
    assert selected==[r['line_hash'] for r in second['details']]
    for entry in summary['groups']:
        if entry['group']=='All':
            assert entry['cases']==600
            for field in ('already_top3','covered_below_top3','missing_reaches_top3','missing_still_below_top3','diagnostic_top1','diagnostic_top3','fp32_checked','membership_changes_in_checked_subset','recoveries_confirmed_fp32'):
                assert entry[field]==counts[entry['arm']][field]
    for row in second['details']:
        scores=dict(row['scores'])
        errors={w:abs(scores[w]-v) for w,v in row['full_sequence_scores'].items()}
        assert errors==row['absolute_formula_errors']
        assert row['formula_passed']==(max(errors.values())<=1e-4)
    assert summary['fp32_cases']==len(checked)
    assert summary['fp32_all_formula_checks_passed']==all(r['formula_passed'] for r in second['details'])
    assert summary['fp32_max_formula_error']==max(v for r in second['details'] for v in r['absolute_formula_errors'].values())
    with (M/'target_rank_cases.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
    assert [(r['arm'],r['line_hash'],int(r['diagnostic_rank'])) for r in table]==computed
    result={'passed':True,'author':'Sonja Sahebzad','scope':'Independent CPU rank/count/selection recomputation from saved scores; not a new model-performance estimate',
        'cases':600,'fp32_cases':len(checked),'counts':counts,'precision_membership_changes':changes,
        'checked_missing_targets':sum(not next(r for r in first['details'] if r['line_hash']==h)['arms']['baseline']['target_present'] for h in checked),
        'closest_missing_target':max(closest,key=lambda x:x['margin']),
        'input_sha256':{p.name:sha(p) for p in (M/'target_rank_protocol.json',M/'target_rank_summary.json',M/'target_rank_fp16.json',M/'target_rank_fp32.json',M/'target_rank_cases.csv')},
        'audit_code_canonical_sha256':hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode()).hexdigest()}
    path=M/'target_rank_independent_audit.json'
    if path.exists():assert load(path)==result
    else:
        with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':check()
