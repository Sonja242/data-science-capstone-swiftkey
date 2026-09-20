"""Neutral numerical diagnostic before performance evaluation. Sonja Sahebzad."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptation_experiment import AdaptedRanker
from neural_predictor import ROOT
ranker = AdaptedRanker('local256')
torch = ranker.torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rows = []
for precision in ('float16', 'float32'):
    if precision == 'float32':
        torch.cuda.empty_cache()
        ranker.model.float()
    for phrase in ('we are talking about', 'the scientific explanation is', 'i will see you'):
        with torch.inference_mode():
            prefix, prefill = ranker._prefill(phrase)
            words = ['the','science','uncharacteristically',"can't"]
            cached = ranker._score_candidates(prefix,prefill,words)
            direct = {}
            for word in words:
                seq = ranker._tokens(word)
                ids = torch.tensor([prefix+seq],device='cuda')
                output = ranker.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
                logp = torch.log_softmax(output.logits[0].float(),dim=-1)
                value = sum(logp[len(prefix)-1+k,token] for k,token in enumerate(seq))
                value += torch.logsumexp(logp[-1,ranker.boundary],dim=-1)
                direct[word] = float(value)
                del output, logp
            row={'precision':precision,'phrase':phrase,'cached':cached,'direct':direct,
                 'max_absolute_error':max(abs(cached[w]-direct[w]) for w in words),
                 'rank_identical':sorted(words,key=lambda w:-cached[w])==sorted(words,key=lambda w:-direct[w])}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del prefill
path=ROOT/'models/word_generation_neutral_precision_diagnostic.json'
with path.open('x',encoding='utf-8') as f:
    json.dump({'initial_fp16_tolerance':0.001,'initial_failure':0.0020313262939453125,
               'no_development_or_final_outcomes_used':True,'results':rows},f,indent=2)
