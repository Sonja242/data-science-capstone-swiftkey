"""Verify numerical equivalence and option separation after GPU vectorization."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from adaptation_experiment import AdaptedRanker,ROOT
from neural_predictor import NeuralWordRanker

ranker=AdaptedRanker("base256")
ranker.check_scoring()
torch=ranker.torch
checks=[]
with torch.inference_mode():
    for phrase in ("we went to the restaurant for","she said that she would rather"):
        prefix,prefill=ranker._prefill(phrase)
        ids=ranker.eligible[torch.topk(prefill.logits[0,-1,ranker.eligible],256).indices].tolist()
        words=list(dict.fromkeys([ranker.decoded[i].strip().lower() for i in ids]+["uncharacteristically","can't"]))
        start=time.perf_counter()
        reference=NeuralWordRanker._score_candidates(ranker,prefix,prefill,words)
        old_seconds=time.perf_counter()-start
        start=time.perf_counter()
        vectorized=ranker._score_candidates(prefix,prefill,words)
        new_seconds=time.perf_counter()-start
        error=max(abs(reference[w]-vectorized[w]) for w in words)
        assert error<.0001,error
        assert sorted(words,key=lambda w:(-reference[w],w))[:3]==sorted(words,key=lambda w:(-vectorized[w],w))[:3]
        checks.append({"candidates":len(words),"max_absolute_log_score_difference":error,
                       "original_seconds":old_seconds,"vectorized_seconds":new_seconds})
    free=ranker.predict("we went to the restaurant for",["dinner"])
    offered=ranker.predict("we went to the restaurant for",["dinner"],["lunch","coffee","tea","dinner"])
    assert free["shortlist"]==offered["shortlist"] and free["words"]==offered["words"]
(ROOT/"models/adaptation_scoring_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
print("PASS: vectorized scores agree with the original implementation; top-three rankings and choice separation preserved.")
print(json.dumps(checks,indent=2))
