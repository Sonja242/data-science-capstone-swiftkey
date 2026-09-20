"""Development-only target-insertion diagnosis. Author: Sonja Sahebzad.

The observed word is privileged information here. Nothing in this file is a
production predictor, a new accuracy result, a model fit or a final-test pass.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "target_rank_"
ARMS = ("baseline", "beam16")
MARGIN = 0.10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path, canonical=False):
    p = Path(path)
    if canonical:
        return hashlib.sha256(p.read_text(encoding="utf-8-sig").encode()).hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, obj):
    with (ROOT/"models"/(PREFIX+name+".json")).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def original(case, arm):
    scores = dict(case["baseline"]["scores"])
    if arm != "baseline":
        additions = dict(case["candidates"][arm]["scores"])
        assert not set(scores) & set(additions)
        scores.update(additions)
    return scores


def rank(scores, target):
    if target not in scores:
        return None
    return 1 + sum(score > scores[target] or
                   (score == scores[target] and word < target)
                   for word, score in scores.items() if word != target)


def inspect(scores, target, supplied_score):
    assert len(scores) >= 3 and all(math.isfinite(x) for x in scores.values())
    present = target in scores
    before = rank(scores, target)
    union = dict(scores)
    if not present:
        assert supplied_score is not None and math.isfinite(supplied_score)
        union[target] = supplied_score
    after = rank(union, target)
    other = sorted((w for w in union if w != target), key=lambda w: (-union[w], w))
    threshold = other[2]
    margin = union[target] - union[threshold]
    category = ("already_top3" if present and before <= 3 else
                "covered_below_top3" if present else
                "missing_reaches_top3" if after <= 3 else "missing_still_below_top3")
    return {"target_present": present, "original_rank": before,
            "diagnostic_rank": after, "category": category,
            "target_score": union[target], "third_other_word": threshold,
            "third_other_score": union[threshold], "margin_to_top3": margin,
            "candidate_count": len(scores), "diagnostic_candidate_count": len(union)}


def reasons(arms):
    out = []
    for arm, item in arms.items():
        if abs(item["margin_to_top3"]) <= MARGIN:
            out.append(arm+":near_top3")
        if item["category"] == "missing_reaches_top3":
            out.append(arm+":potential_recovery")
    return out


def inputs():
    # Only development outcomes are opened. The old manifest contains model
    # fingerprints; no final case file or final-result artifact is read.
    from learning_curve_experiment import inference_fingerprint
    old = read(ROOT/"models/word_generation_implementation.json")["fingerprint"]
    assert inference_fingerprint() == old["inference_inputs_sha256"]
    assert digest(ROOT/"python/adaptation_experiment.py", True) == old["scorer_canonical_sha256"]
    assert digest(ROOT/"python/neural_predictor.py") == old["base_code_sha256"]
    artifact = read(ROOT/"models/word_generation_development.json")
    assert artifact["split"] == "development"
    assert artifact["implementation_sha256"] == digest(ROOT/"models/word_generation_implementation.json")
    with (ROOT/"data/adaptation/development.csv").open(encoding="utf-8-sig", newline="") as f:
        cases = list(csv.DictReader(f))
    assert len(cases) == len(artifact["details"]) == 600
    assert len({r["line_hash"] for r in cases}) == 600
    for case, saved in zip(cases, artifact["details"]):
        assert (case["source"],case["line_hash"],case["actual"]) == (saved["source"],saved["line_hash"],saved["actual"])
        for arm in ARMS:
            scores = original(saved,arm)
            ordered = sorted(scores, key=lambda w:(-scores[w],w))[:3]
            expected = saved["baseline"]["words"] if arm == "baseline" else saved["candidates"][arm]["words"]
            assert ordered == expected
    return cases, artifact


def input_hashes():
    return {name:digest(ROOT/name) for name in (
        "data/adaptation/development.csv", "models/word_generation_development.json",
        "models/word_generation_implementation.json")}


def full_score(ranker, phrase, word):
    t = ranker.torch
    prefix = ranker.tokenizer.encode(phrase.strip(),add_special_tokens=False)[-128:]
    seq = ranker._tokens(word)
    ids = t.tensor([prefix+seq],device="cuda")
    out = ranker.model(input_ids=ids,attention_mask=t.ones_like(ids),use_cache=False)
    lp = t.log_softmax(out.logits[0].float(),dim=-1)
    score = sum(lp[len(prefix)-1+i,token] for i,token in enumerate(seq))
    return float(score+t.logsumexp(lp[-1,ranker.boundary],dim=-1))


def run():
    protocol_path = ROOT/"models/target_rank_protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Retain this registered diagnosis; use --verify to inspect existing results.")
    cases, raw = inputs()
    protocol = {"author":"Sonja Sahebzad", "registered_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Exploratory development-only target insertion; no production performance, training or final evaluation",
        "cases":600,"arms":list(ARMS),"input_sha256":input_hashes(),
        "analysis_canonical_sha256":digest(__file__,True),
        "scoring":"Keep all saved candidate scores. Score an absent target once as a singleton with the same operational FP16 scorer. Existing target scores are never replaced.",
        "precision_selection":"All cases where either arm has abs(target score minus third best other score)<=0.10, plus every missing target reaching top3. No subsampling or outcome cap.",
        "near_top3_log_margin":MARGIN,"fp32_batch_size":4,
        "fp32":"Promote the identical FP16 merged weights to FP32, TF32 disabled. Rescore the full baseline/beam16/target union. Restrict that common score map to each original pool plus target. Do not regenerate candidates.",
        "formula_check":"FP32 target and third-other word in every checked arm: cached versus direct full sequence, tolerance1e-4; preserve all observations if exceeded.",
        "interpretation":"Target-aware hypothetical ceiling within the fixed pool only. New competing words could lower rank. Finite precision outside the selected subset is not certified.",
        "no_final_files_opened":True,"production_unchanged":True}
    save("protocol",protocol)
    from adaptation_experiment import AdaptedRanker
    ranker = AdaptedRanker("local256")
    t = ranker.torch
    details = []
    with t.inference_mode():
        for i,(case,saved) in enumerate(zip(cases,raw["details"])):
            target = case["actual"]
            pools = {arm:original(saved,arm) for arm in ARMS}
            assert len(ranker._tokens(target)) == saved["target_tokens"]
            extra = None
            if target not in pools["baseline"]:
                prefix,prefill = ranker._prefill(case["prefix"])
                extra = ranker._score_candidates(prefix,prefill,[target])[target]
                del prefill
            arms = {arm:inspect(pool,target,extra) for arm,pool in pools.items()}
            details.append({"source":case["source"],"line_hash":case["line_hash"],"target":target,
                "target_tokens":saved["target_tokens"],"target_in_training_vocabulary":saved["target_in_vocabulary"],
                "injected_fp16_score":extra,"arms":arms,"fp32_reasons":reasons(arms)})
            if (i+1)%100==0:
                print(f"FP16 target diagnosis {i+1}/600",flush=True)
    save("fp16",{"protocol_sha256":digest(protocol_path),"details":details})
    selected = [i for i,row in enumerate(details) if row["fp32_reasons"]]
    print(f"Checking {len(selected)} selected cases in FP32",flush=True)
    # Identical merged parameter values, with more precise arithmetic, not a new fit.
    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False
    ranker.model.float()
    ranker.batch_size = protocol["fp32_batch_size"]
    t.cuda.empty_cache()
    checks = []
    with t.inference_mode():
        for j,i in enumerate(selected):
            case,saved,row = cases[i],raw["details"][i],details[i]
            target = case["actual"]
            pools = {arm:original(saved,arm) for arm in ARMS}
            words = list(dict.fromkeys(list(pools["beam16"])+[target]))
            prefix,prefill = ranker._prefill(case["prefix"])
            scores = ranker._score_candidates(prefix,prefill,words)
            del prefill
            arms = {arm:inspect({w:scores[w] for w in pool},target,scores[target]) for arm,pool in pools.items()}
            formula_words = list(dict.fromkeys([target]+[x["third_other_word"] for x in arms.values()]))
            direct = {w:full_score(ranker,case["prefix"],w) for w in formula_words}
            errors = {w:abs(scores[w]-direct[w]) for w in direct}
            checks.append({"source":row["source"],"line_hash":row["line_hash"],"target":target,
                "reasons":row["fp32_reasons"],"scores":list(scores.items()),"arms":arms,
                "full_sequence_scores":direct,"absolute_formula_errors":errors,
                "formula_passed":max(errors.values())<=1e-4})
            print(f"FP32 boundary checks {j+1}/{len(selected)}",flush=True)
    save("fp32",{"protocol_sha256":digest(protocol_path),"details":checks,
        "all_formula_checks_passed":all(c["formula_passed"] for c in checks),
        "max_formula_error":max((v for c in checks for v in c["absolute_formula_errors"].values()),default=0),
        "torch":str(t.__version__),"gpu":t.cuda.get_device_name(),
        "same_merged_weights":True,"tf32_disabled":True,"no_candidate_regeneration":True})
    summarize_and_verify(write=True)


def summarize_and_verify(write=False):
    cases, raw = inputs()
    protocol = read(ROOT/"models/target_rank_protocol.json")
    assert protocol["input_sha256"] == input_hashes()
    assert protocol["analysis_canonical_sha256"] == digest(__file__,True)
    a = read(ROOT/"models/target_rank_fp16.json")
    b = read(ROOT/"models/target_rank_fp32.json")
    assert a["protocol_sha256"] == b["protocol_sha256"] == digest(ROOT/"models/target_rank_protocol.json")
    assert len(a["details"]) == len(cases)
    records = []
    selected = []
    checked = {c["line_hash"]:c for c in b["details"]}
    assert len(checked) == len(b["details"])
    for case,saved,row in zip(cases,raw["details"],a["details"]):
        target = case["actual"]
        assert (row["line_hash"],row["source"],row["target"]) == (case["line_hash"],case["source"],target)
        assert row["target_tokens"] == saved["target_tokens"]
        assert row["target_in_training_vocabulary"] == saved["target_in_vocabulary"]
        computed = {arm:inspect(original(saved,arm),target,row["injected_fp16_score"]) for arm in ARMS}
        assert computed == row["arms"] and reasons(computed) == row["fp32_reasons"]
        if row["fp32_reasons"]:
            selected.append(row["line_hash"])
            check = checked[row["line_hash"]]
            assert check["reasons"] == row["fp32_reasons"]
            scores = dict(check["scores"])
            assert len(scores) == len(check["scores"])
            assert set(scores) == set(original(saved,"beam16")) | {target}
            for arm in ARMS:
                assert inspect({w:scores[w] for w in original(saved,arm)},target,scores[target]) == check["arms"][arm]
            for w,v in check["full_sequence_scores"].items():
                assert abs(scores[w]-v) == check["absolute_formula_errors"][w]
            assert check["formula_passed"] == (max(check["absolute_formula_errors"].values())<=1e-4)
        for arm,item in computed.items():
            fp = checked.get(row["line_hash"],{}).get("arms",{}).get(arm)
            records.append({"source":row["source"],"line_hash":row["line_hash"],"target":target,
                "arm":arm,"canonical_tokens":row["target_tokens"],"target_in_training_vocabulary":row["target_in_training_vocabulary"],
                **item,"fp32_checked":fp is not None,
                "fp32_diagnostic_rank":fp["diagnostic_rank"] if fp else None,
                "fp32_margin":fp["margin_to_top3"] if fp else None})
    assert selected == [c["line_hash"] for c in b["details"]]
    categories = ("already_top3","covered_below_top3","missing_reaches_top3","missing_still_below_top3")
    groups = []
    for arm in ARMS:
        for group in ("All","single_token","multiple_tokens","blogs","news","twitter"):
            rows = [r for r in records if r["arm"]==arm and (group=="All" or
                group=="single_token" and r["canonical_tokens"]==1 or
                group=="multiple_tokens" and r["canonical_tokens"]>1 or r["source"]==group)]
            groups.append({"arm":arm,"group":group,"cases":len(rows),
                **{cat:sum(r["category"]==cat for r in rows) for cat in categories},
                "diagnostic_top1":sum(r["diagnostic_rank"]==1 for r in rows),
                "diagnostic_top3":sum(r["diagnostic_rank"]<=3 for r in rows),
                "fp32_checked":sum(r["fp32_checked"] for r in rows),
                "membership_changes_in_checked_subset":sum(r["fp32_checked"] and ((r["diagnostic_rank"]<=3)!=(r["fp32_diagnostic_rank"]<=3)) for r in rows),
                "recoveries_confirmed_fp32":sum(r["category"]=="missing_reaches_top3" and r["fp32_diagnostic_rank"]<=3 for r in rows)})
    outcome = {"author":"Sonja Sahebzad","diagnostic_not_predictive_performance":True,
        "groups":groups,"fp32_cases":len(selected),"fp32_all_formula_checks_passed":b["all_formula_checks_passed"],
        "fp32_max_formula_error":b["max_formula_error"],"input_sha256":{name:digest(ROOT/"models"/name) for name in
          ("target_rank_protocol.json","target_rank_fp16.json","target_rank_fp32.json")}}
    assert b["all_formula_checks_passed"] == all(c["formula_passed"] for c in b["details"])
    assert b["max_formula_error"] == max((v for c in b["details"] for v in c["absolute_formula_errors"].values()),default=0)
    if write:
        save("summary",outcome)
        with (ROOT/"models/target_rank_cases.csv").open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    else:
        assert read(ROOT/"models/target_rank_summary.json") == outcome
        with (ROOT/"models/target_rank_cases.csv").open(encoding="utf-8",newline="") as f:
            exported=list(csv.DictReader(f))
        assert len(exported)==len(records)
        for x,y in zip(exported,records):
            assert x=={k:"" if v is None else str(v) for k,v in y.items()}
    print(json.dumps({"verified":True,"fp32_cases":len(selected),"groups":[x for x in groups if x["group"]=="All"],
        "formula_max_error":b["max_formula_error"],"formula_passed":b["all_formula_checks_passed"]},indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",action="store_true")
    group.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    run() if args.run else summarize_and_verify()
