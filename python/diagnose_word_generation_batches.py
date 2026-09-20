"""Investigate the largest development FP16 batch discrepancy, no final data."""
import csv
import json
from adaptation_experiment import AdaptedRanker
from neural_predictor import ROOT
from learning_curve_experiment import read_json, save_new
audit=read_json(ROOT/'models/word_generation_development_audit.json')
maximum=audit['outcome']['numerical_diagnostics']['maximum_comparison']
with (ROOT/'data/adaptation/development.csv').open(encoding='utf-8-sig',newline='') as stream:
    case=next(r for r in csv.DictReader(stream) if r['line_hash']==maximum['line_hash'])
raw=read_json(ROOT/'models/word_generation_development.json')
detail=next(d for d in raw['details'] if d['line_hash']==maximum['line_hash'])
ranker=AdaptedRanker('local256')
torch=ranker.torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
ranker.model.float()
torch.cuda.empty_cache()
word=maximum['word']
scores={}
with torch.inference_mode():
    prefix,prefill=ranker._prefill(case['prefix'])
    for candidate in ('beam16','beam64'):
        names=[w for w,_ in detail['candidates'][candidate]['scores']]
        # Reproduce the exact original batch containing the identified word.
        index=names.index(word)
        batch=names[(index//64)*64:(index//64+1)*64]
        scores[candidate]=ranker._score_candidates(prefix,prefill,batch)[word]
    seq=ranker._tokens(word)
    ids=torch.tensor([prefix+seq],device='cuda')
    out=ranker.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    logp=torch.log_softmax(out.logits[0].float(),dim=-1)
    direct=sum(logp[len(prefix)-1+k,token] for k,token in enumerate(seq))
    direct+=torch.logsumexp(logp[-1,ranker.boundary],dim=-1)
    scores['full_sequence']=float(direct)
gap=abs(scores['beam16']-scores['beam64'])
assert gap<.0001
assert max(abs(scores[c]-scores['full_sequence']) for c in ('beam16','beam64'))<.0001
result={'scope':'Largest development cross-batch discrepancy; no final cases or changed predictions',
    'same_merged_weights_promoted_to_fp32':True,'maximum_saved_fp16_comparison':maximum,
    'fp32_scores':scores,'fp32_cross_batch_absolute_difference':gap,
    'fp32_formula_check_passed':True,'scoring_or_selection_changed':False,
    'limitation':'One diagnostic case is not a guarantee for other corpus cases; published outcomes remain the frozen FP16 implementation.'}
save_new(ROOT/'models/word_generation_batch_precision_diagnostic.json',result)
print(json.dumps(result,indent=2),flush=True)
