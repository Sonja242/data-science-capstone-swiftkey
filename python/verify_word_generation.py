"""Independent reconstruction and provenance checks. Author: Sonja Sahebzad."""
import argparse
import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from neural_predictor import ROOT
from learning_curve_experiment import read_json, sha, code_sha, save_new
from word_generation_experiment import frozen
from word_generation_audit import derive_artifacts, select_generation, primary_comparison


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def provenance(raw, split):
    frozen()
    assert raw['split']==split
    assert raw['implementation_sha256']==sha(ROOT/'models/word_generation_implementation.json')
    assert raw['protocol_sha256']==sha(ROOT/'models/word_generation_protocol.json')
    filename='data/adaptation/development.csv' if split=='development' else 'data/word_generation/final_test.csv'
    cases=read_csv(ROOT/filename)
    expected=[(r['source'],r['line_hash']) for r in cases]
    assert expected==[(r['source'],r['line_hash']) for r in raw['details']]
    expected_arms={'baseline','beam16','beam64'} if split=='development' else {
        'baseline',read_json(ROOT/'models/word_generation_selection.json')['candidate']}
    assert set(raw['timings'])==set(raw['summaries'])==expected_arms
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/neural/Qwen3-1.7B-Base',local_files_only=True,trust_remote_code=False)
    vocabulary={r['word'] for r in read_csv(ROOT/'data/neural_evaluation/vocabulary.csv')}
    for case,row in zip(cases,raw['details']):
        target=case['actual']
        assert row['actual']==target
        assert row['target_tokens']==len(tokenizer.encode(' '+target,add_special_tokens=False))
        assert row['target_in_vocabulary']==(target in vocabulary)
        import json
        assert set(dict(row['options']['scores']))==set(json.loads(case['options_json']))
        for candidate in row['candidates'].values():
            proposed=candidate['proposal_words']
            added=dict(candidate['scores'])
            assert len(proposed)<=128 and len(set(proposed))==len(proposed)
            assert set(added)==set(proposed)-set(dict(row['baseline']['scores']))
            assert all(w in vocabulary for w in proposed)
            assert all(2<=len(tokenizer.encode(' '+w,add_special_tokens=False))<=6 for w in proposed)
    n_per_source=20 if split=='development' else 300
    seen=Counter();timing_keys=[]
    for case in cases:
        if seen[case['source']]<n_per_source:
            timing_keys.append((case['source'],case['line_hash']));seen[case['source']]+=1
    for candidate,times in raw['timings'].items():
        assert timing_keys==[(t['source'],t['line_hash']) for t in times]
        assert all(math.isfinite(t['milliseconds']) and t['milliseconds']>0 for t in times)
        assert raw['summaries'][candidate]['timing_cases']==len(times)
        assert abs(raw['summaries'][candidate]['milliseconds']-np.mean([t['milliseconds'] for t in times]))<1e-9
    return expected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    if args.development==args.final: raise ValueError('Choose one audit stage')
    split='development' if args.development else 'final'
    raw=read_json(ROOT/'models'/f'word_generation_{split}.json')
    expected=provenance(raw,split)
    selection=read_json(ROOT/'models/word_generation_selection.json')
    assert selection['implementation_sha256']==sha(ROOT/'models/word_generation_implementation.json')
    assert selection['protocol_sha256']==sha(ROOT/'models/word_generation_protocol.json')
    assert selection['development_artifact_sha256']==sha(ROOT/'models/word_generation_development.json')
    if args.development:
        metrics={name:raw['summaries'][name]['milliseconds'] for name in ('beam16','beam64')}
        outcome=select_generation(raw,metrics)
        assert outcome['candidate']==selection['candidate']
        artifacts=derive_artifacts(raw,expected_keys=expected,expected_split='development')
        for name,artifact in artifacts.items():
            for key,value in artifact['summary'].items():
                assert abs(value-raw['summaries'][name][key])<1e-12
        filename='word_generation_development_audit.json'
    else:
        dev_audit=read_json(ROOT/'models/word_generation_development_audit.json')
        assert dev_audit['passed']
        assert dev_audit['raw_artifact_sha256']==sha(ROOT/'models/word_generation_development.json')
        development=read_json(ROOT/'models/word_generation_development.json')
        provenance(development,'development')
        assert development['implementation_sha256']==sha(ROOT/'models/word_generation_implementation.json')
        assert development['protocol_sha256']==sha(ROOT/'models/word_generation_protocol.json')
        dev_selection=select_generation(development,{name:development['summaries'][name]['milliseconds'] for name in ('beam16','beam64')})
        dev_truth={'baseline':dev_selection['baseline'],**{r['candidate']:r for r in dev_selection['development']}}
        for name,metrics in dev_truth.items():
            for key,value in metrics.items():
                if key in development['summaries'][name]:assert abs(value-development['summaries'][name][key])<1e-12
        assert dev_selection['candidate']==selection['candidate']
        assert selection['development_summaries']==development['summaries']
        partition=read_json(ROOT/'models/word_generation_partition_audit.json')
        assert partition['passed']
        assert partition['reserved_hash_manifest_sha256']==sha(ROOT/'models/word_generation_reserved_case_hashes.csv')
        assert partition['partition_rds_sha256']==sha(ROOT/'models/expanded_partitions_v3.rds')
        assert raw['selection_sha256']==sha(ROOT/'models/word_generation_selection.json')
        outcome=primary_comparison(raw,selection['candidate'],expected_keys=expected)
        report=read_json(ROOT/'models/word_generation_summary.json')
        expected_inputs={'word_generation_protocol.json','word_generation_implementation.json',
                         'word_generation_selection.json','word_generation_development.json','word_generation_final.json'}
        assert set(report['input_sha256'])==expected_inputs
        for file,digest in report['input_sha256'].items():assert digest==sha(ROOT/'models'/file)
        assert report['selection']['candidate']==selection['candidate']
        primary=outcome['primary_top3']
        assert report['promote']==primary['promote']
        assert abs(report['primary']['difference']-primary['difference'])<1e-12
        assert report['primary']['interval']==primary['interval']
        assert report['primary']['exact_paired_p']==primary['exact_paired_p']
        assert report['mechanism']['gains']==primary['gains']
        assert report['mechanism']['losses']==primary['losses']
        assert report['mechanism']['coverage_gains']==primary['expanded_coverage']-primary['baseline_coverage']
        assert report['mechanism']['coverage_losses']==primary['lost_coverage']==0
        assert report['mechanism']['gains_from_previously_missing']==primary['gains_from_previously_missing_target']
        derived=derive_artifacts(raw,(selection['candidate'],),expected_keys=expected,expected_split='final')
        for dataset,table in ((development,report['development']),(raw,report['final'])):
            assert len(table)==len(dataset['summaries'])
            assert {r['candidate'] for r in table}==set(dataset['summaries'])
            for row in table:
                assert {k:v for k,v in row.items() if k!='candidate'}==dataset['summaries'][row['candidate']]
        for name,artifact in derived.items():
            for key,value in artifact['summary'].items():assert abs(value-raw['summaries'][name][key])<1e-12
        expected_groups={row['group']:row for row in outcome['descriptive_groups']}
        expected_groups['multiple_tokens']=expected_groups.pop('multi_token')
        expected_groups['All']=primary
        assert len(report['groups'])==len(expected_groups)
        assert {row['group'] for row in report['groups']}==set(expected_groups)
        for row in report['groups']:
            actual=expected_groups[row['group']];n=actual['cases']
            assert row['cases']==n
            if n==0:
                assert row['gains']==row['losses']==0
                assert all(row[key] is None for key in ('baseline_top3','expanded_top3','baseline_coverage','expanded_coverage'))
                continue
            for key in ('gains','losses'):assert row[key]==actual[key]
            for key,count_name in (('baseline_top3','baseline_top3_correct'),('expanded_top3','expanded_top3_correct'),
                                   ('baseline_coverage','baseline_coverage'),('expanded_coverage','expanded_coverage')):
                assert abs(row[key]-actual[count_name]/n)<1e-12
        # Check the compact public prediction tables against the full evidence.
        for label,dataset in (('development',development),('final',raw)):
            exported=read_csv(ROOT/'models'/f'word_generation_{label}_cases.csv')
            expected_records=[(case,name,item) for case in dataset['details'] for name,item in case['candidates'].items()]
            assert len(exported)==len(expected_records)
            for record,(case,name,item) in zip(exported,expected_records):
                actual=case['actual'];base=case['baseline'];scores=dict(base['scores']);union=dict(scores,**dict(item['scores']))
                old_rank=base['words'].index(actual)+1 if actual in base['words'] else 0
                rank=item['words'].index(actual)+1 if actual in item['words'] else 0
                assert (record['candidate'],record['source'],record['line_hash'])==(name,case['source'],case['line_hash'])
                assert record['observed_word']==actual and int(record['canonical_target_tokens'])==case['target_tokens']
                assert int(record['baseline_rank_at3'])==old_rank and int(record['expanded_rank_at3'])==rank
                for key,value in (('baseline_covered',actual in scores),('expanded_covered',actual in union),
                    ('gain',not old_rank and rank>0),('loss',old_rank>0 and not rank),
                    ('synthetic_choice_correct',case['options']['words'][0]==actual)):
                    assert record[key]==str(bool(value))
                for side,words in (('baseline',base['words']),('expanded',item['words'])):
                    assert [record[side+'_'+position] for position in ('first','second','third')]==words
                for side,mapping in (('baseline',scores),('expanded',union)):
                    value=record[side+'_target_log_score']
                    assert abs(float(value)-mapping[actual])<1e-12 if actual in mapping else value==''
        filename='word_generation_independent_audit.json'
    result={'passed':True,'author':'Sonja Sahebzad','split':split,
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'outcome':outcome,
        'raw_artifact_sha256':sha(ROOT/'models'/f'word_generation_{split}.json'),
        'auditor_canonical_sha256':code_sha(ROOT/'python/word_generation_audit.py'),
        'verification_code_canonical_sha256':code_sha(__file__),
        'data_identity_targets_options_vocabulary_token_counts_verified':True,
        'all_proposed_words_are_complete_training_words_within_frozen_depth':True,
        'timings_independently_recomputed':True}
    if args.final:
        result['prediction_csv_sha256']={label:sha(ROOT/'models'/f'word_generation_{label}_cases.csv') for label in ('development','final')}
    save_new(ROOT/'models'/filename,result)
    print(f'PASS: {split} independent metrics, identity, target encodings, timings and selection/paired test',flush=True)


if __name__=='__main__':main()
