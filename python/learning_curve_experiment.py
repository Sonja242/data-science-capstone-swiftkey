"""Frozen learning-curve comparisons and local inference. Sonja Sahebzad."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
import numpy as np
from neural_predictor import ROOT, NeuralWordRanker, WORD
from adaptation_experiment import AdaptedRanker
from learning_curve_audit import (audit_prediction, audit_artifact, keys,
    select_on_development, confirmatory_comparison)

SEEDS = (20261410, 20261411, 20261412)
STEPS = (256, 512, 1024)
WIDTHS = (64, 256)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def save_new(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def write_csv(filename, rows):
    with (ROOT / "models" / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol():
    p = read_json(ROOT / "models/learning_curve_protocol.json")
    assert p["development_sha256"] == sha(ROOT / "data/adaptation/development.csv")
    assert p["final_sha256"] == sha(ROOT / "data/learning_curve/final_test.csv")
    assert p["trainer_code_canonical_sha256"] == code_sha(ROOT / "python/learning_curve_training.py")
    assert p["operational_adapter_sha256"] == sha(ROOT / "models/neural/adaptation_local/adapter_model.safetensors")
    return p


@lru_cache(maxsize=1)
def inference_fingerprint():
    base = ROOT / "models/neural/Qwen3-1.7B-Base"
    files = sorted(base.glob("*.json")) + sorted(base.glob("*.safetensors"))
    if (base / "merges.txt").exists():
        files.append(base / "merges.txt")
    files += [ROOT / "data/neural_evaluation/vocabulary.csv",
              ROOT / "models/selected_predictor_v3.rds",
              ROOT / "models/neural/adaptation_local/adapter_config.json",
              ROOT / "models/neural/adaptation_local/adapter_model.safetensors"]
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    config = read_json(ROOT / "models/neural/adaptation_local/adapter_config.json")
    assert config["r"] == 8 and config["lora_alpha"] == 16 and config["lora_dropout"] == .05
    assert set(config["target_modules"]) == {"q_proj", "v_proj"}
    download = read_json(base / "download_manifest.json")
    assert download["revision"] == "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    return hashes


def implementation():
    result = {"code_sha256": code_sha(__file__),
              "audit_sha256": code_sha(ROOT / "python/learning_curve_audit.py"),
              "base_code_sha256": sha(ROOT / "python/neural_predictor.py"),
              "vectorized_code_sha256": code_sha(ROOT / "python/adaptation_experiment.py"),
              "protocol_sha256": sha(ROOT / "models/learning_curve_protocol.json"),
              "inference_inputs_sha256": inference_fingerprint()}
    return result


def frozen():
    protocol()
    expected = read_json(ROOT / "models/learning_curve_implementation.json")
    assert expected["implementation"] == implementation(), "Frozen implementation changed"
    return expected


def cases(split):
    relative = "data/adaptation/development.csv" if split == "development" else "data/learning_curve/final_test.csv"
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checkpoint(seed, step):
    if seed not in SEEDS or step not in STEPS:
        raise ValueError("Unregistered training checkpoint")
    run = read_json(ROOT / "models" / f"learning_curve_training_seed_{seed}.json")
    item = next(x for x in run["checkpoints"] if x["step"] == step)
    folder = ROOT / item["relative_path"]
    assert sha(folder / "training_metadata.json") == item["metadata_sha256"]
    assert sha(folder / "adapter_model.safetensors") == item["adapter_sha256"]
    metadata = read_json(folder / "training_metadata.json")
    assert metadata["completed_step"] == step and metadata["plan"]["seed"] == seed
    assert metadata["fingerprint"]["training_code_canonical_sha256"] == protocol()["trainer_code_canonical_sha256"]
    for name, value in metadata["fingerprint"]["base_file_sha256"].items():
        assert inference_fingerprint()[f"models/neural/Qwen3-1.7B-Base/{name}"] == value
    for name, value in metadata["adapter_files_sha256"].items():
        assert sha(folder / name) == value
    return folder, item, metadata


class CurveRanker(NeuralWordRanker):
    # Reuse the previously verified vectorized mathematical scoring rule.
    _score_candidates = AdaptedRanker._score_candidates

    def __init__(self, seed=None, step=None):
        self.candidate = "learning_curve"
        super().__init__("1.7B", shortlist=256, batch_size=64)
        self.eligible_cpu = self.eligible.cpu().numpy()
        if seed is None:
            folder = ROOT / "models/neural/adaptation_local"
            self.adapter_sha256 = protocol()["operational_adapter_sha256"]
        else:
            folder, item, _ = checkpoint(seed, step)
            self.adapter_sha256 = item["adapter_sha256"]
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, folder, is_trainable=False).merge_and_unload()
        self.model.config.use_cache = True
        self.model.eval()

    def free_sets(self, prefill, ngram_candidates, widths):
        values = prefill.logits[0, -1, self.eligible].float().cpu().numpy()
        # Eligible IDs are ascending; explicit secondary key resolves cutoff ties.
        ordered = np.lexsort((self.eligible_cpu, -values))[:256]
        sets, metadata = {}, {}
        for width in widths:
            chosen = self.eligible_cpu[ordered[:width]]
            raw = [self.decoded[int(i)].strip().lower() for i in chosen]
            words = list(dict.fromkeys(raw + list(ngram_candidates)))
            sets[width] = [word for word in words if WORD.fullmatch(word)]
            cutoff = values[ordered[width - 1]]
            metadata[width] = {"raw_token_cutoff_ties": int(np.sum(values == cutoff)),
                               "unique_neural_words": len(set(raw))}
        if 64 in sets and 256 in sets:
            assert set(sets[64]) <= set(sets[256])
        return sets, metadata

    def predict_widths(self, phrase, ngram_candidates=(), choices=(), widths=WIDTHS):
        widths = tuple(widths)
        assert widths and all(width in WIDTHS for width in widths)
        with self.torch.inference_mode():
            prefix, prefill = self._prefill(phrase)
            sets, metadata = self.free_sets(prefill, ngram_candidates, widths)
            scores = self._score_candidates(prefix, prefill, sets[max(widths)])
            options = list(dict.fromkeys(word.strip().lower() for word in choices))
            if any(not WORD.fullmatch(word) for word in options):
                raise ValueError("Supply single normalized English words as choices")
            # Free batches never depend on supplied options, including their padding.
            missing = [word for word in options if word not in scores]
            if missing:
                scores.update(self._score_candidates(prefix, prefill, missing))
            ranked_options = sorted(options, key=lambda word: (-scores[word], word))
            output = {}
            for width in widths:
                ranking = sorted(sets[width], key=lambda word: (-scores[word], word))
                margin = scores[ranking[2]] - scores[ranking[3]] if len(ranking) > 3 else 0.
                output[width] = {"words": ranking[:3], "shortlist": sets[width],
                    "choices": ranked_options, "full_ranking": ranking,
                    "context_tokens": len(prefix), "top3_margin": margin,
                    **metadata[width], "log_scores": {word: scores[word] for word in ranking}}
            return output


def details_for(ranker, row, output):
    detail = audit_prediction(row, output)
    actual = row["actual"]
    full_rank = output["full_ranking"].index(actual) + 1 if actual in output["full_ranking"] else 0
    detail.update({"target_full_rank": full_rank, "candidate_count": len(output["shortlist"]),
        "target_token_count": len(ranker._tokens(actual)), "context_words": len(row["prefix"].split()),
        "context_tokens": output["context_tokens"], "target_characters": len(actual),
        "top3_margin": output["top3_margin"], "raw_token_cutoff_ties": output["raw_token_cutoff_ties"],
        "unique_neural_words": output["unique_neural_words"]})
    assert bool(full_rank) == detail["shortlist_contains_target"]
    assert detail["rank"] == (full_rank if 0 < full_rank <= 3 else 0)
    return detail


def request_row(ranker, row, widths):
    return ranker.predict_widths(row["prefix"], json.loads(row["ngram_candidates_json"]),
                                json.loads(row["options_json"]), widths)


def verify_result(result, split, expected_identity=None):
    assert result["implementation"] == frozen()["implementation"]
    assert result["split"] == split
    if expected_identity:
        for field, value in expected_identity.items():
            assert result.get(field) == value, f"Artifact identity mismatch: {field}"
    audit_artifact(result, keys(cases(split)))
    if result.get("seed") in SEEDS:
        _, item, _ = checkpoint(result["seed"], result["steps"])
        assert item["adapter_sha256"] == result["adapter_sha256"]
    elif split == "test":
        assert result["name"] == "prior_operational" and result["steps"] == 256 and result["shortlist"] == 256
        assert result["adapter_sha256"] == protocol()["operational_adapter_sha256"]


def evaluate_development(seed, step):
    frozen()
    paths = {width: ROOT / "models" / f"learning_curve_dev_{seed}_{step}_{width}.json" for width in WIDTHS}
    if all(path.exists() for path in paths.values()):
        for width, path in paths.items():
            verify_result(read_json(path), "development", {"seed": seed, "steps": step, "shortlist": width})
        print("Reusing verified development checkpoint", seed, step, flush=True)
        return
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("Partial evaluation exists; preserve and review before retrying")
    ranker = CurveRanker(seed, step)
    ranker.check_scoring()
    rows = cases("development")
    all_details = {width: [] for width in WIDTHS}
    joint_predictions = {}
    start = time.perf_counter()
    ranker.torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(rows):
        predictions = request_row(ranker, row, WIDTHS)
        joint_predictions[row["line_hash"]] = {width: {key: predictions[width][key]
            for key in ("words", "shortlist", "choices")} for width in WIDTHS}
        for width in WIDTHS:
            all_details[width].append(details_for(ranker, row, predictions[width]))
        if (i + 1) % 100 == 0:
            print(f"Development seed{seed} step{step}: {i+1}/{len(rows)}; both widths", flush=True)
    joint_ms = 1000 * (time.perf_counter() - start) / len(rows)
    timing_rows = [row for source in ("blogs", "news", "twitter") for row in
                   [x for x in rows if x["source"] == source][:20]]
    timings = {width: [] for width in WIDTHS}
    for i, row in enumerate(timing_rows):
        # Alternate order within pairs to reduce warm-cache/order imbalance.
        for width in WIDTHS if i % 2 == 0 else WIDTHS[::-1]:
            ranker.torch.cuda.synchronize()
            started = time.perf_counter()
            direct = request_row(ranker, row, (width,))[width]
            ranker.torch.cuda.synchronize()
            timings[width].append(1000 * (time.perf_counter() - started))
            for field in ("words", "shortlist", "choices"):
                assert direct[field] == joint_predictions[row["line_hash"]][width][field], \
                    f"Shared/direct output differs: {seed}/{step}/{width}/{field}; review before selecting"
    for width in WIDTHS:
        result = {"seed": seed, "steps": step, "shortlist": width, "split": "development",
            "details": all_details[width], "adapter_sha256": ranker.adapter_sha256,
            "implementation": implementation(), "joint_scoring_ms_per_case": joint_ms,
            "timing_case_hashes": [row["line_hash"] for row in timing_rows],
            "direct_shared_check": "Free words, membership and choices identical on all60 fixed timing cases"}
        result["summary"] = {**audit_artifact(result), "milliseconds": float(np.mean(timings[width])),
            "timing_cases": len(timing_rows), "latency_p95_ms": float(np.quantile(timings[width], .95)),
            "peak_gpu_mib": ranker.torch.cuda.max_memory_allocated() / 1024**2}
        save_new(paths[width], result)
        print(seed, step, width, json.dumps(result["summary"]), flush=True)


def select_development():
    results = []
    for seed in SEEDS:
        for step in STEPS:
            for width in WIDTHS:
                result = read_json(ROOT / "models" / f"learning_curve_dev_{seed}_{step}_{width}.json")
                verify_result(result, "development", {"seed": seed, "steps": step, "shortlist": width})
                results.append(result)
    selection = select_on_development(results, checkpoints=STEPS, shortlists=WIDTHS, seeds=SEEDS)
    selection["implementation"] = implementation()
    selection["artifact_sha256"] = {f"{r['seed']}_{r['steps']}_{r['shortlist']}":
        sha(ROOT / "models" / f"learning_curve_dev_{r['seed']}_{r['steps']}_{r['shortlist']}.json") for r in results}
    path = ROOT / "models/learning_curve_selection.json"
    if path.exists():
        assert read_json(path) == selection
    else:
        save_new(path, selection)
    write_csv("learning_curve_development.csv", [{"seed": r["seed"], "steps": r["steps"],
        "shortlist": r["shortlist"], **r["summary"]} for r in results])
    groups = []
    for group in selection["development_groups"]:
        selected = [r for r in results if r["steps"] == group["steps"] and r["shortlist"] == group["shortlist"]]
        groups.append({**group, "synthetic_choice_accuracy": float(np.mean([r["summary"]["synthetic_choice_accuracy"] for r in selected])),
            "shortlist_recall": float(np.mean([r["summary"]["shortlist_recall"] for r in selected])),
            "top3_min": min(r["summary"]["top3"] for r in selected), "top3_max": max(r["summary"]["top3"] for r in selected)})
    write_csv("learning_curve_development_groups.csv", groups)
    training = []
    for seed in SEEDS:
        for step in STEPS:
            _, _, metadata = checkpoint(seed, step)
            training.append({"seed": seed, "steps": step, "input_tokens": metadata["input_tokens"],
                "training_seconds": metadata["runtime"]["training_seconds_excluding_checkpoint_io"],
                "peak_gpu_mib": metadata["runtime"]["peak_gpu_allocated_mib"]})
    write_csv("learning_curve_training_cost.csv", training)
    print("Selection fixed BEFORE opening final examples:", json.dumps(selection["selected"]),
          "representative seed", selection["representative_seed"], flush=True)
    return selection


def verify_selection():
    """Bind the frozen choice to the complete pre-test development grid."""
    selection = read_json(ROOT / "models/learning_curve_selection.json")
    assert selection["implementation"] == implementation()
    results = []
    for seed in SEEDS:
        for step in STEPS:
            for width in WIDTHS:
                name = f"{seed}_{step}_{width}"
                path = ROOT / "models" / f"learning_curve_dev_{name}.json"
                assert sha(path) == selection["artifact_sha256"][name]
                result = read_json(path)
                verify_result(result, "development", {"seed": seed, "steps": step, "shortlist": width})
                results.append(result)
    recomputed = select_on_development(results, checkpoints=STEPS, shortlists=WIDTHS, seeds=SEEDS)
    for key, value in recomputed.items():
        assert selection[key] == value
    return selection


def evaluate_final(seed=None):
    frozen()
    selection = verify_selection()
    name = "prior_operational" if seed is None else f"seed_{seed}"
    step = 256 if seed is None else selection["selected"]["steps"]
    width = 256 if seed is None else selection["selected"]["shortlist"]
    identity = {"name": name, "seed": seed or 20261310, "steps": step, "shortlist": width}
    path = ROOT / "models" / f"learning_curve_final_{name}.json"
    if path.exists():
        verify_result(read_json(path), "test", identity)
        print("Reusing verified final result", name, flush=True)
        return
    if seed is None:
        ranker = AdaptedRanker("local256")
        width, step = 256, 256
        adapter_hash = ranker.adapter_hash
    else:
        step, width = selection["selected"]["steps"], selection["selected"]["shortlist"]
        ranker = CurveRanker(seed, step)
        adapter_hash = ranker.adapter_sha256
    ranker.check_scoring()
    rows = cases("test")
    details = []
    ranker.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for i, row in enumerate(rows):
        if seed is None:
            prediction = ranker.predict(row["prefix"], json.loads(row["ngram_candidates_json"]), json.loads(row["options_json"]))
            details.append(audit_prediction(row, prediction))
        else:
            prediction = request_row(ranker, row, (width,))[width]
            details.append(details_for(ranker, row, prediction))
        if (i + 1) % 100 == 0:
            print(f"Final {name}: {i+1}/{len(rows)}", flush=True)
    result = {"name": name, "seed": seed or 20261310, "steps": step, "shortlist": width,
        "split": "test", "details": details, "implementation": implementation(), "adapter_sha256": adapter_hash}
    result["summary"] = {**audit_artifact(result), "milliseconds": 1000 * (time.perf_counter() - started) / len(rows),
                         "peak_gpu_mib": ranker.torch.cuda.max_memory_allocated() / 1024**2}
    save_new(path, result)
    print(name, json.dumps(result["summary"]), flush=True)


def finish():
    selection = select_development()
    # These are the only final evaluations allowed by the frozen design.
    subprocess.run([sys.executable, __file__, "--final", "operational"], cwd=ROOT, check=True)
    for seed in SEEDS:
        subprocess.run([sys.executable, __file__, "--final", str(seed)], cwd=ROOT, check=True)
    baseline = read_json(ROOT / "models/learning_curve_final_prior_operational.json")
    replicas = [read_json(ROOT / "models" / f"learning_curve_final_seed_{seed}.json") for seed in SEEDS]
    verify_result(baseline, "test", {"name": "prior_operational", "seed": 20261310, "steps": 256, "shortlist": 256})
    for seed, result in zip(SEEDS, replicas):
        verify_result(result, "test", {"name": f"seed_{seed}", "seed": seed,
            "steps": selection["selected"]["steps"], "shortlist": selection["selected"]["shortlist"]})
    representative = next(r for r in replicas if r["seed"] == selection["representative_seed"])
    comparison = confirmatory_comparison(baseline, representative, replicas)
    promote = comparison["primary_top3"]["promote"]
    summary = {"selection": selection, "comparison": comparison, "promote": promote,
        "default_spec": {"kind": "curve", "seed": representative["seed"], "steps": representative["steps"],
                         "shortlist": representative["shortlist"]} if promote else {"kind": "operational", "seed": 20261310, "steps": 256, "shortlist": 256},
        "free_text_85_met": representative["summary"]["top3"] >= .85,
        "synthetic_choice_85_met": representative["summary"]["synthetic_choice_accuracy"] >= .85,
        "implementation": implementation()}
    path = ROOT / "models/learning_curve_summary.json"
    if path.exists():
        assert read_json(path) == summary
    else:
        save_new(path, summary)
    write_csv("learning_curve_test.csv", [{"name": r["name"], "seed": r["seed"], "steps": r["steps"],
        "shortlist": r["shortlist"], **r["summary"]} for r in [baseline, *replicas]])
    source_rows = []
    for r in [baseline, *replicas]:
        for source in ("blogs", "news", "twitter"):
            subset = [d for d in r["details"] if d["source"] == source]
            source_rows.append({"name": r["name"], "source": source, **audit_artifact({"details": subset})})
    write_csv("learning_curve_sources.csv", source_rows)
    print(json.dumps(summary, indent=2), flush=True)


def check_scoring():
    protocol()
    ranker = CurveRanker()
    ranker.check_scoring()
    checks = []
    phrases = ["we went to the restaurant for", "she said that she would rather",
               "the team said " * 60 + "we are talking about"]
    options = ["dinner", "lunch", "uncharacteristically", "can't"]
    for phrase in phrases:
        joint = ranker.predict_widths(phrase, ["dinner", "the", "there", "can't"])
        with_options = ranker.predict_widths(phrase, ["dinner", "the", "there", "can't"], options)
        direct = ranker.predict_widths(phrase, ["dinner", "the", "there", "can't"], options, widths=(64,))[64]
        assert set(joint[64]["shortlist"]) <= set(joint[256]["shortlist"])
        errors = [abs(joint[64]["log_scores"][w] - direct["log_scores"][w]) for w in direct["shortlist"]]
        # FP16 batch-shape rounding: neutral diagnostic max0.000774, same rankings.
        # Frozen before development; exact output checks remain mandatory.
        assert max(errors) < .001
        assert joint[64]["words"] == direct["words"]
        assert direct["choices"] == with_options[64]["choices"]
        assert joint[64]["context_tokens"] == min(128, len(ranker.tokenizer.encode(phrase.strip(), add_special_tokens=False)))
        for width in WIDTHS:
            assert joint[width]["shortlist"] == with_options[width]["shortlist"]
            assert joint[width]["words"] == with_options[width]["words"]
            assert joint[width]["log_scores"] == with_options[width]["log_scores"]
        checks.append({"context_tokens": joint[64]["context_tokens"], "max_shared_direct_log_score_error": max(errors),
            "width64_candidates": len(joint[64]["shortlist"]), "width256_candidates": len(joint[256]["shortlist"]),
            "nesting_passed": True, "option_invariance_passed": True})
    result = {"passed": True, "absolute_log_score_tolerance": .001,
              "exact_prediction_agreement_required": True, "checks": checks, "implementation": implementation(),
              "recorded_utc": datetime.now(timezone.utc).isoformat()}
    save_new(ROOT / "models/learning_curve_scoring_checks.json", result)
    save_new(ROOT / "models/learning_curve_implementation.json", result)
    print("PASS: nested candidate sets, shared/direct scoring, option invariance, full/cached scoring and long-context truncation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-scoring", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, choices=STEPS)
    parser.add_argument("--evaluate-development", action="store_true")
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--final", choices=("operational", *(str(seed) for seed in SEEDS)))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.evaluate_development and (args.seed is None or args.steps is None):
        parser.error("--evaluate-development requires --seed and --steps")
    if args.request and args.output is None:
        parser.error("--request requires --output")
    if args.check_scoring:
        check_scoring()
    if args.evaluate_development:
        evaluate_development(args.seed, args.steps)
    if args.finish:
        finish()
    if args.final:
        evaluate_final(None if args.final == "operational" else int(args.final))
    if args.request:
        frozen()
        spec = read_json(ROOT / "models/learning_curve_summary.json")["default_spec"]
        request = read_json(args.request)
        if spec["kind"] == "curve":
            ranker = CurveRanker(spec["seed"], spec["steps"])
            result = ranker.predict_widths(request["phrase"], request.get("ngram_candidates", []),
                                          request.get("choices", []), (spec["shortlist"],))[spec["shortlist"]]
            result.pop("full_ranking", None)
        else:
            ranker = AdaptedRanker("local256")
            result = ranker.predict(request["phrase"], request.get("ngram_candidates", []), request.get("choices", []))
        result["selected_spec"] = spec
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
