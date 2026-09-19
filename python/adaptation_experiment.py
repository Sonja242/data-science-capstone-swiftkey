"""Diagnose, compare and select fixed next-word candidates. Sonja Sahebzad."""
from pathlib import Path
import argparse
import csv
import copy
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
from neural_predictor import ROOT, NeuralWordRanker

CANDIDATES={
    "base64":{"shortlist":64,"adapter":None},
    "base256":{"shortlist":256,"adapter":None},
    "local256":{"shortlist":256,"adapter":"local"},
    "augmented256":{"shortlist":256,"adapter":"augmented"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_cases(split):
    with (ROOT/"data/adaptation"/f"{split}.csv").open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name,rows):
    with (ROOT/"models"/name).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class AdaptedRanker(NeuralWordRanker):
    def __init__(self,candidate):
        self.candidate=candidate
        config=CANDIDATES[candidate]
        super().__init__("1.7B",shortlist=config["shortlist"],batch_size=64 if config["shortlist"]>64 else 32)
        self.adapter_hash=None
        if config["adapter"]:
            from peft import PeftModel
            folder=ROOT/"models/neural"/("adaptation_"+config["adapter"])
            manifest=read_json(ROOT/"models"/f"adaptation_training_{config['adapter']}.json")
            self.adapter_hash=sha(folder/"adapter_model.safetensors")
            assert self.adapter_hash==manifest["adapter_sha256"]
            self.model=PeftModel.from_pretrained(self.model,folder,is_trainable=False).merge_and_unload()
            self.model.config.use_cache=True
            self.model.eval()

    def _score_candidates(self,prefix_ids,prefill,words):
        if self.candidate=="base64":
            return super()._score_candidates(prefix_ids,prefill,words)
        torch=self.torch
        initial=torch.log_softmax(prefill.logits[0,-1].float(),dim=-1)
        scores={}
        for start in range(0,len(words),self.batch_size):
            names=words[start:start+self.batch_size]
            sequences=[self._tokens(w) for w in names]
            lengths=torch.tensor([len(x) for x in sequences],device="cuda")
            width=max(map(len,sequences)); batch=len(names)
            # Pad on CPU once, then transfer one tensor instead of many tiny tensors.
            padded=np.full((batch,width),self.tokenizer.eos_token_id,dtype=np.int64)
            for i,seq in enumerate(sequences): padded[i,:len(seq)]=seq
            ids=torch.from_numpy(padded).to("cuda")
            mask=(torch.arange(width,device="cuda")[None,:]<lengths[:,None]).long()
            cache=copy.deepcopy(prefill.past_key_values)
            cache.batch_repeat_interleave(batch)
            attention=torch.cat((torch.ones((batch,len(prefix_ids)),device="cuda",dtype=torch.long),mask),dim=1)
            out=self.model(input_ids=ids,attention_mask=attention,past_key_values=cache,use_cache=True,logits_to_keep=0)
            logits=out.logits.float(); normalizers=torch.logsumexp(logits,dim=-1)
            value=initial[ids[:,0]]
            if width>1:
                token_scores=logits[:,:-1,:].gather(2,ids[:,1:,None]).squeeze(-1)-normalizers[:,:-1]
                value=value+(token_scores*mask[:,1:]).sum(dim=1)
            row=torch.arange(batch,device="cuda"); last=lengths-1
            last_logits=logits[row,last,:]
            value=value+torch.logsumexp(last_logits[:,self.boundary],dim=-1)-normalizers[row,last]
            scores.update(zip(names,value.cpu().tolist()))
            del out,cache,logits,normalizers,last_logits
        if not all(np.isfinite(x) for x in scores.values()): raise RuntimeError("Non-finite candidate scores")
        return scores


def metrics(details):
    n=len(details)
    return {"cases":n,"top1":sum(d["rank"]==1 for d in details)/n,
        "top3":sum(d["rank"]>0 for d in details)/n,
        "synthetic_choice_accuracy":sum(d["choice_correct"] for d in details)/n,
        "mrr_at3":sum(1/d["rank"] if d["rank"] else 0 for d in details)/n,
        "shortlist_recall":sum(d["shortlist_contains_target"] for d in details)/n}


def evaluate(candidate,split):
    path=ROOT/"models"/f"adaptation_{candidate}_{split}.json"
    if path.exists():
        result=read_json(path)
        verify(result,split)
        print("Reusing verified result:",path.name,flush=True)
        return
    ranker=AdaptedRanker(candidate)
    ranker.check_scoring()
    rows=read_cases(split); details=[]
    start=time.perf_counter()
    ranker.torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        result=ranker.predict(row["prefix"],json.loads(row["ngram_candidates_json"]),json.loads(row["options_json"]))
        actual=row["actual"]
        details.append({"source":row["source"],"line_hash":row["line_hash"],
            "rank":result["words"].index(actual)+1 if actual in result["words"] else 0,
            "choice_correct":result["choices"][0]==actual,
            "shortlist_contains_target":actual in result["shortlist"],
            "context_words":len(row["prefix"].split()),"target_characters":len(actual)})
        if (i+1)%100==0: print(f"{candidate} {split}: {i+1}/{len(rows)}",flush=True)
    result={"candidate":candidate,"split":split,"configuration":CANDIDATES[candidate],
        "inference_batch_size":ranker.batch_size,
        "summary":{**metrics(details),"milliseconds":1000*(time.perf_counter()-start)/len(rows),
                   "peak_gpu_mib":ranker.torch.cuda.max_memory_allocated()/1024**2},
        "details":details,"base_model":ranker.manifest,"adapter_sha256":ranker.adapter_hash,
        "code_sha256":code_sha(__file__),"base_code_sha256":sha(ROOT/"python/neural_predictor.py"),
        "data_sha256":sha(ROOT/"data/adaptation"/f"{split}.csv")}
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(candidate,split,json.dumps(result["summary"]),flush=True)


def verify(result,split):
    assert result["code_sha256"]==code_sha(__file__)
    assert result["base_code_sha256"]==sha(ROOT/"python/neural_predictor.py")
    assert result["data_sha256"]==sha(ROOT/"data/adaptation"/f"{split}.csv")
    rows=read_cases(split)
    assert [(r["source"],r["line_hash"]) for r in rows]==[(d["source"],d["line_hash"]) for d in result["details"]]
    recipe=result["configuration"]["adapter"]
    if recipe:
        assert result["adapter_sha256"]==sha(ROOT/"models/neural"/("adaptation_"+recipe)/"adapter_model.safetensors")


def diagnose():
    previous=read_json(ROOT/"models/neural_1.7B_development.json")["details"]
    with (ROOT/"data/neural_evaluation/development.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    assert [r["line_hash"] for r in rows]==[d["line_hash"] for d in previous]
    groups={"All previous development":list(range(len(rows)))}
    for source in ("blogs","news","twitter"):
        groups[source]=[i for i,r in enumerate(rows) if r["source"]==source]
    for label,lo,hi in (("1-3 context words",1,3),("4-9 context words",4,9),("10+ context words",10,100000)):
        groups[label]=[i for i,r in enumerate(rows) if lo<=len(r["prefix"].split())<=hi]
    table=[]
    for label,ids in groups.items():
        d=[previous[i] for i in ids]
        table.append({"group":label,"cases":len(d),"top3":sum(x["rank"]>0 for x in d)/len(d),
            "target_missing":sum(not x["shortlist_contains_target"] for x in d)/len(d),
            "ranking_miss":sum(x["rank"]==0 and x["shortlist_contains_target"] for x in d)/len(d)})
    write_csv("adaptation_error_analysis.csv",table)


def compare():
    development=[]
    for candidate in CANDIDATES:
        result=read_json(ROOT/"models"/f"adaptation_{candidate}_development.json")
        verify(result,"development")
        development.append({"candidate":candidate,**result["summary"]})
    write_csv("adaptation_development.csv",development)
    selected=max(development,key=lambda r:(r["top3"],r["synthetic_choice_accuracy"],-r["milliseconds"]))
    choice={"candidate":selected["candidate"],"criterion":"Development top-3, then synthetic choice accuracy, then speed",
            "code_sha256":code_sha(__file__),"development_sha256":sha(ROOT/"data/adaptation/development.csv")}
    path=ROOT/"models/adaptation_selection.json"
    if path.exists(): assert read_json(path)==choice
    else: path.write_text(json.dumps(choice,indent=2),encoding="utf-8")
    print("Selection frozen before final test:",choice,flush=True)
    # These three secondary contrasts are fixed before looking at test outcomes:
    # shortlist width, local adaptation, and external-data augmentation.
    for candidate in CANDIDATES:
        subprocess.run([sys.executable,__file__,"--evaluate",candidate,"--split","test"],check=True,cwd=ROOT)
    reference=read_json(ROOT/"models/adaptation_base64_test.json")
    chosen=read_json(ROOT/"models"/f"adaptation_{choice['candidate']}_test.json")
    verify(reference,"test"); verify(chosen,"test")
    all_results=[read_json(ROOT/"models"/f"adaptation_{candidate}_test.json") for candidate in CANDIDATES]
    for result in all_results: verify(result,"test")
    tests=[{"candidate":x["candidate"],**x["summary"]} for x in all_results]
    write_csv("adaptation_test.csv",tests)
    strata=sorted({d["source"] for d in reference["details"]})
    rng=np.random.default_rng(20261320)
    intervals={}
    for metric in ("top3","choice"):
        paired=np.array([(int(b["rank"]>0)-int(a["rank"]>0)) if metric=="top3" else
                        (int(b["choice_correct"])-int(a["choice_correct"]))
                        for a,b in zip(reference["details"],chosen["details"])])
        bootstrap=np.zeros(10000)
        for source in strata:
            group=paired[[i for i,x in enumerate(reference["details"]) if x["source"]==source]]
            bootstrap+=rng.choice(group,(10000,len(group)),replace=True).mean(axis=1)/len(strata)
        intervals[metric]={"difference":float(paired.mean()),"ci95":np.quantile(bootstrap,[.025,.975]).tolist()}
    contrasts=[]
    for label,left,right in (("Expanded shortlist",0,1),("Local adaptation",1,2),("Extra dialogue data",2,3)):
        for metric in ("top3","choice"):
            delta=np.array([(int(b["rank"]>0)-int(a["rank"]>0)) if metric=="top3" else
                (int(b["choice_correct"])-int(a["choice_correct"]))
                for a,b in zip(all_results[left]["details"],all_results[right]["details"])])
            boot=np.zeros(10000)
            for source in strata:
                group=delta[[i for i,x in enumerate(reference["details"]) if x["source"]==source]]
                boot+=rng.choice(group,(10000,len(group)),replace=True).mean(axis=1)/len(strata)
            bounds=np.quantile(boot,[.025,.975])
            contrasts.append({"contrast":label,"metric":metric,"difference":float(delta.mean()),
                              "lower":float(bounds[0]),"upper":float(bounds[1])})
    write_csv("adaptation_planned_contrasts.csv",contrasts)
    source_rows=[]
    for result in all_results:
        for source in strata:
            d=[x for x in result["details"] if x["source"]==source]
            source_rows.append({"candidate":result["candidate"],"source":source,**metrics(d)})
    write_csv("adaptation_sources.csv",source_rows)
    constant=read_json(ROOT/"models/adaptation_constant_baseline.json")
    constant["test_top1"]=sum(r["actual"]==constant["word"] for r in read_cases("test"))/len(read_cases("test"))
    summary={"selection":choice,"test":tests,"paired":intervals,"secondary_contrasts":contrasts,
        "constant_baseline":constant,
        "promote":intervals["top3"]["ci95"][0]>0,
        "default_candidate":choice["candidate"] if intervals["top3"]["ci95"][0]>0 else "base64",
        "free_text_85_met":chosen["summary"]["top3"]>=.85,
        "synthetic_choices_85_met":chosen["summary"]["synthetic_choice_accuracy"]>=.85,
        "limitations":["External pretrained-model overlap is unknown.","Synthetic options are not a quiz score.",
          "A small fixed-budget adaptation experiment does not establish the best attainable model."]}
    (ROOT/"models/adaptation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evaluate",choices=CANDIDATES)
    parser.add_argument("--split",choices=("development","test"),default="development")
    parser.add_argument("--diagnose",action="store_true")
    parser.add_argument("--compare",action="store_true")
    parser.add_argument("--request",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--candidate",choices=CANDIDATES)
    args=parser.parse_args()
    if args.diagnose: diagnose()
    if args.evaluate: evaluate(args.evaluate,args.split)
    if args.compare: compare()
    if args.request:
        candidate=args.candidate or read_json(ROOT/"models/adaptation_summary.json")["default_candidate"]
        ranker=AdaptedRanker(candidate)
        request=read_json(args.request)
        result=ranker.predict(request["phrase"],request.get("ngram_candidates",[]),request.get("choices",[]))
        result["candidate"]=candidate
        args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
