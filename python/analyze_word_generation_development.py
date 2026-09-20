"""CPU-only structural reach versus bounded proposal coverage. Sonja Sahebzad."""
import csv
import json
from transformers import AutoTokenizer
from neural_predictor import ROOT, WORD
from learning_curve_experiment import read_json, save_new
from word_generation_audit import derive_artifacts, paired_counts

def main():
    raw=read_json(ROOT/'models/word_generation_development.json')
    artifacts=derive_artifacts(raw)
    with (ROOT/'data/adaptation/development.csv').open(encoding='utf-8-sig',newline='') as f:
        cases=list(csv.DictReader(f))
    with (ROOT/'data/neural_evaluation/vocabulary.csv').open(encoding='utf-8-sig',newline='') as f:
        vocabulary={r['word'] for r in csv.DictReader(f)}
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/neural/Qwen3-1.7B-Base',local_files_only=True)
    decoded=tokenizer.batch_decode([[i] for i in range(len(tokenizer))],clean_up_tokenization_spaces=False)
    eligible_words={token.strip().lower() for token in decoded if token.startswith(' ') and WORD.fullmatch(token.strip().lower()) and token.strip().lower() in vocabulary}
    structural=[]
    for case,row in zip(cases,raw['details']):
        assert case['line_hash']==row['line_hash']
        target=row['actual']
        old=target in eligible_words or target in json.loads(case['ngram_candidates_json'])
        new=old or (target in vocabulary and 2<=row['target_tokens']<=6)
        structural.append({'source':row['source'],'line_hash':row['line_hash'],
            'target_tokens':row['target_tokens'],'old_structurally_reachable':old,
            'new_structurally_reachable':new,
            'baseline_covered':target in dict(row['baseline']['scores']),
            'beam16_covered':target in dict(row['baseline']['scores']) or target in dict(row['candidates']['beam16']['scores']),
            'beam64_covered':target in dict(row['baseline']['scores']) or target in dict(row['candidates']['beam64']['scores'])})
    output={'scope':'Development only; structural lexicon reach assumes unrestricted proposal search and is not actual beam coverage',
        'cases':len(structural),'old_structurally_unreachable':sum(not x['old_structurally_reachable'] for x in structural),
        'new_structurally_unreachable':sum(not x['new_structurally_reachable'] for x in structural),
        'newly_structurally_reachable':sum(not x['old_structurally_reachable'] and x['new_structurally_reachable'] for x in structural),
        'transitions':{name:paired_counts(artifacts['baseline']['details'],artifacts[name]['details']) for name in ('beam16','beam64')},
        'details':structural}
    save_new(ROOT/'models/word_generation_development_mechanism.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='details'},indent=2))

if __name__=='__main__':main()
