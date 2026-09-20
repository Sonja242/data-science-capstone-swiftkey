"""Recompute report values from published per-case scores. Sonja Sahebzad."""
import csv
import json
from pathlib import Path
from word_generation_experiment import ROOT, frozen, read_json, sha, summarize
from learning_curve_audit import confirmatory_comparison


def main():
    frozen()
    dev = read_json(ROOT / 'models/word_generation_development.json')
    test = read_json(ROOT / 'models/word_generation_final.json')
    selected = read_json(ROOT / 'models/word_generation_selection.json')
    name = selected['candidate']
    assert test['selection_sha256'] == sha(ROOT / 'models/word_generation_selection.json')
    assert selected['development_artifact_sha256'] == sha(ROOT / 'models/word_generation_development.json')
    assert set(test['summaries']) == {'baseline',name}
    a, rows_a = summarize(test['details'],'baseline')
    b, rows_b = summarize(test['details'],name)
    comparison = confirmatory_comparison({'summary':a,'details':rows_a},
        {'summary':b,'details':rows_b},bootstrap_seed=20261590)
    primary = comparison['primary_top3']
    groups = []
    predicates = {'All':lambda d:True,'single_token':lambda d:d['target_tokens']==1,
                  'multiple_tokens':lambda d:d['target_tokens']>1}
    predicates.update({s:lambda d,s=s:d['source']==s for s in ('blogs','news','twitter')})
    mechanism = None
    for label, predicate in predicates.items():
        indices = [i for i,d in enumerate(test['details']) if predicate(d)]
        x,y = [rows_a[i] for i in indices],[rows_b[i] for i in indices]
        n=len(x)
        gains=sum(not u['rank'] and v['rank']>0 for u,v in zip(x,y))
        losses=sum(u['rank']>0 and not v['rank'] for u,v in zip(x,y))
        gain_missing=sum(not u['rank'] and v['rank']>0 and not u['shortlist_contains_target'] for u,v in zip(x,y))
        coverage_gains=sum(not u['shortlist_contains_target'] and v['shortlist_contains_target'] for u,v in zip(x,y))
        coverage_losses=sum(u['shortlist_contains_target'] and not v['shortlist_contains_target'] for u,v in zip(x,y))
        assert coverage_losses==0 and gain_missing==gains
        groups.append({'group':label,'cases':n,'baseline_top3':sum(d['rank']>0 for d in x)/n if n else None,
            'expanded_top3':sum(d['rank']>0 for d in y)/n if n else None,
            'baseline_coverage':sum(d['shortlist_contains_target'] for d in x)/n if n else None,
            'expanded_coverage':sum(d['shortlist_contains_target'] for d in y)/n if n else None,
            'gains':gains,'losses':losses})
        if label=='All':
            mechanism={'gains':gains,'losses':losses,'coverage_gains':coverage_gains,
                'coverage_losses':coverage_losses,'gains_from_previously_missing':gain_missing}
    development=[]
    for candidate, metrics in dev['summaries'].items():
        computed,_=summarize(dev['details'],candidate)
        for key,value in computed.items(): assert abs(metrics[key]-value)<1e-12
        development.append({'candidate':candidate,**metrics})
    final=[]
    for candidate, metrics in test['summaries'].items():
        computed,_=summarize(test['details'],candidate)
        for key,value in computed.items(): assert abs(metrics[key]-value)<1e-12
        final.append({'candidate':candidate,**metrics})
    result={'author':'Sonja Sahebzad','promote':primary['promote'],'selection':{'candidate':name},
        'development':development,'final':final,'primary':primary,'mechanism':mechanism,'groups':groups,
        'input_sha256':{file:sha(ROOT/'models'/file) for file in (
            'word_generation_protocol.json','word_generation_implementation.json',
            'word_generation_selection.json','word_generation_development.json','word_generation_final.json')},
        'limitations':['Repeatedly reused development data; one fresh final test.',
          'Exact normalized-line deduplication does not exclude near duplicates or pretrained-model overlap.',
          'Equal weighting of blogs/news/twitter; single observed next word, not every plausible continuation.',
          'Synthetic choices are a fixed scorer diagnostic, not quiz results.',
          'Bounded approximate vocabulary search, canonical paths and heuristic boundary mass remain limitations.']}
    path=ROOT/'models/word_generation_summary.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for filename,rows in (('word_generation_development.csv',development),
                          ('word_generation_final.csv',final),('word_generation_groups.csv',groups)):
        with (ROOT/'models'/filename).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # One row per case/configuration, with baseline paired in the same row.
    # These small CSVs complement complete candidate/score JSON, not replace it.
    for split,raw in (('development',dev),('final',test)):
        records=[]
        for case in raw['details']:
            actual=case['actual'];old=case['baseline'];old_scores=dict(old['scores'])
            old_rank=old['words'].index(actual)+1 if actual in old['words'] else 0
            for candidate,expanded in case['candidates'].items():
                scores=dict(old_scores,**dict(expanded['scores']))
                rank=expanded['words'].index(actual)+1 if actual in expanded['words'] else 0
                records.append({'candidate':candidate,'source':case['source'],'line_hash':case['line_hash'],
                    'observed_word':actual,'canonical_target_tokens':case['target_tokens'],
                    'baseline_rank_at3':old_rank,'expanded_rank_at3':rank,
                    'baseline_covered':actual in old_scores,'expanded_covered':actual in scores,
                    'gain':not old_rank and rank>0,'loss':old_rank>0 and not rank,
                    'baseline_first':old['words'][0],'baseline_second':old['words'][1],'baseline_third':old['words'][2],
                    'expanded_first':expanded['words'][0],'expanded_second':expanded['words'][1],'expanded_third':expanded['words'][2],
                    'baseline_target_log_score':old_scores.get(actual,''),'expanded_target_log_score':scores.get(actual,''),
                    'synthetic_choice_correct':case['options']['words'][0]==actual})
        with (ROOT/'models'/f'word_generation_{split}_cases.csv').open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
