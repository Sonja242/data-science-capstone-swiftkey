"""Controlled complete-word proposal experiment. Author: Sonja Sahebzad.

Training-vocabulary trie proposals only. Archived model and scoring code stays
unchanged. All original free candidates retain their exact original scores.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from neural_predictor import ROOT, WORD
from adaptation_experiment import AdaptedRanker
from learning_curve_experiment import sha, code_sha, read_json, save_new, inference_fingerprint

BEAMS = (16, 64)
MAX_TOKENS = 6
CAP = 128
P = ROOT / "models/word_generation_protocol.json"
F = ROOT / "models/word_generation_implementation.json"


class WordTrie:
    """Canonical complete words, including terminal nodes with children."""
    def __init__(self, entries, max_tokens=MAX_TOKENS):
        self.children = [{}]
        self.terminal = [None]
        self.paths = [()]
        self.statistics = Counter()
        for word, sequence in entries:
            self.statistics["vocabulary_words"] += 1
            if not WORD.fullmatch(word):
                self.statistics["filtered_words"] += 1
                continue
            if len(sequence) > max_tokens:
                self.statistics["over_depth_words"] += 1
                continue
            if len(sequence) < 2:
                self.statistics["single_token_words"] += 1
                continue
            self.statistics["eligible_multiple_token_words"] += 1
            node = 0
            for token in sequence:
                if token not in self.children[node]:
                    index = len(self.children)
                    self.children[node][token] = index
                    self.children.append({})
                    self.terminal.append(None)
                    self.paths.append(self.paths[node] + (token,))
                node = self.children[node][token]
            if self.terminal[node] not in (None, word):
                raise ValueError("Different canonical words share a token path")
            self.terminal[node] = word
        self.statistics["nodes"] = len(self.children)
        self.statistics["terminal_with_children"] = sum(
            bool(word and self.children[i]) for i, word in enumerate(self.terminal))

    def expand(self, frontier, edge_values, depth, beam, completed):
        """All vocabulary edges are scored before a deterministic beam cutoff."""
        continuing = []
        for (node, logp), values in zip(frontier, edge_values):
            for (token, child), value in zip(self.children[node].items(), values):
                joint = logp + float(value)
                word = self.terminal[child]
                if word is not None:
                    completed[word] = joint
                if self.children[child] and depth < MAX_TOKENS:
                    continuing.append((child, joint))
        continuing.sort(key=lambda item: (-item[1], self.paths[item[0]]))
        return continuing[:beam]


class CompleteWordRanker(AdaptedRanker):
    def __init__(self):
        super().__init__("local256")
        with (ROOT / "data/neural_evaluation/vocabulary.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.vocabulary = {row["word"] for row in csv.DictReader(stream) if WORD.fullmatch(row["word"])}
        words = sorted(self.vocabulary)
        # Tokenize once, from the same training-only vocabulary as the old model.
        encodings = self.tokenizer([" " + w for w in words], add_special_tokens=False)["input_ids"]
        self.trie = WordTrie(zip(words, encodings))

    def baseline_set(self, prefill, ngram_candidates):
        top = self.eligible[self.torch.topk(prefill.logits[0, -1, self.eligible], 256).indices].tolist()
        return [w for w in dict.fromkeys([self.decoded[i].strip().lower() for i in top]
                                         + list(ngram_candidates)) if WORD.fullmatch(w)]

    def proposals(self, prefix, prefill, beam):
        """Bounded vocabulary-trie beam search, no targets/options accepted."""
        if beam not in BEAMS:
            raise ValueError("Unregistered beam width")
        torch = self.torch
        frontier = [(0, 0.)]
        completed = {}
        expanded = 0
        depth = 0
        for depth in range(1, MAX_TOKENS + 1):
            if not frontier:
                break
            if depth == 1:
                logits = prefill.logits[:, -1, :].float()
            else:
                sequences = [self.trie.paths[node] for node, _ in frontier]
                ids = torch.tensor(sequences, device="cuda", dtype=torch.long)
                cache = copy.deepcopy(prefill.past_key_values)
                cache.batch_repeat_interleave(len(sequences))
                attention = torch.ones((len(sequences), len(prefix) + depth - 1), device="cuda", dtype=torch.long)
                output = self.model(input_ids=ids, attention_mask=attention,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = output.logits[:, -1, :].float()
            counts = [len(self.trie.children[node]) for node, _ in frontier]
            token_ids = [token for node, _ in frontier for token in self.trie.children[node]]
            rows = np.repeat(np.arange(len(frontier)), counts)
            normalizers = torch.logsumexp(logits, dim=-1)
            values = (logits[torch.as_tensor(rows, device="cuda"),
                             torch.tensor(token_ids, device="cuda")]
                      - normalizers[torch.as_tensor(rows, device="cuda")]).cpu().numpy()
            edge_values = np.split(values, np.cumsum(counts)[:-1])
            expanded += len(frontier)
            frontier = self.trie.expand(frontier, edge_values, depth, beam, completed)
            del logits, normalizers
            if depth > 1:
                del output, cache
        ranked = sorted(completed, key=lambda word: (-completed[word], word))[:CAP]
        return ranked, {"beam": beam, "max_word_tokens": MAX_TOKENS,
            "proposal_cap": CAP, "completed_before_cap": len(completed),
            "expanded_nodes": expanded, "returned_proposals": len(ranked)}

    def predict_variants(self, phrase, ngram_candidates=(), choices=(), beams=BEAMS):
        torch = self.torch
        with torch.inference_mode():
            prefix, prefill = self._prefill(phrase)
            original = self.baseline_set(prefill, ngram_candidates)
            scores = self._score_candidates(prefix, prefill, original)
            ranking = sorted(original, key=lambda w: (-scores[w], w))
            result = {"baseline": {"scores": list(scores.items()), "words": ranking[:3]},
                      "candidates": {}}
            for beam in beams:
                proposed, metadata = self.proposals(prefix, prefill, beam)
                additions = [w for w in proposed if w not in scores]
                extra = self._score_candidates(prefix, prefill, additions) if additions else {}
                union = dict(scores, **extra)
                ranked = sorted(union, key=lambda w: (-union[w], w))
                # The baseline dictionary is never rescored or mutated.
                assert all(union[w] == score for w, score in scores.items())
                result["candidates"][f"beam{beam}"] = {
                    "scores": list(extra.items()), "words": ranked[:3],
                    "proposal_words": proposed, "search": metadata}
            options = list(dict.fromkeys(w.strip().lower() for w in choices))
            if any(not WORD.fullmatch(w) for w in options):
                raise ValueError("Options must be single normalized English words")
            # Separate, identical scoring task for both arms; never feeds proposals.
            option_scores = self._score_candidates(prefix, prefill, options) if options else {}
            result["options"] = {"scores": list(option_scores.items()),
                                 "words": sorted(options, key=lambda w: (-option_scores[w], w))}
            return result


def fingerprint():
    return {"implementation_canonical_sha256": code_sha(__file__),
        "scorer_canonical_sha256": code_sha(ROOT / "python/adaptation_experiment.py"),
        "base_code_sha256": sha(ROOT / "python/neural_predictor.py"),
        "inference_inputs_sha256": inference_fingerprint(),
        "protocol_sha256": sha(P)}


def frozen():
    p = read_json(P)
    assert p["development_sha256"] == sha(ROOT / "data/adaptation/development.csv")
    assert p["final_sha256"] == sha(ROOT / "data/word_generation/final_test.csv")
    assert read_json(F)["fingerprint"] == fingerprint(), "Frozen implementation or inputs changed"
    return p


def neutral_checks(ranker):
    phrases = ["we are talking about", "the scientific explanation is", "i will see you"]
    checks = []
    torch = ranker.torch
    for phrase in phrases:
        reference = AdaptedRanker.predict(ranker, phrase, ["uncharacteristically", "science", "can't"])
        trial = ranker.predict_variants(phrase, ["uncharacteristically", "science", "can't"], ["art", "the", "can't", "science"])
        assert reference["words"] == trial["baseline"]["words"]
        assert reference["shortlist"] == [w for w, _ in trial["baseline"]["scores"]]
        no_options = ranker.predict_variants(phrase, ["uncharacteristically", "science", "can't"])
        assert trial["baseline"] == no_options["baseline"]
        assert trial["candidates"] == no_options["candidates"]
        with torch.inference_mode():
            prefix, prefill = ranker._prefill(phrase)
            words = ["the", "science", "uncharacteristically", "can't"]
            cached = ranker._score_candidates(prefix, prefill, words)
            errors = []
            for word in words:
                seq = ranker._tokens(word)
                ids = torch.tensor([prefix + seq], device="cuda")
                out = ranker.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                logp = torch.log_softmax(out.logits[0].float(), dim=-1)
                direct = sum(logp[len(prefix)-1+k, token] for k, token in enumerate(seq))
                direct += torch.logsumexp(logp[-1, ranker.boundary], dim=-1)
                error = abs(float(direct) - cached[word])
                # Neutral FP32 diagnostic verifies the formula to <=1e-4.
                # FP16 batching may change logs by up to 0.02 (about2% ratio).
                assert error <= .02, (word, error)
                errors.append(error)
        checks.append({"phrase": phrase, "operational_free_equivalence": True,
                       "options_free_invariance": True, "max_absolute_log_score_error": max(errors)})
    return checks


def freeze():
    if P.exists() or F.exists():
        raise FileExistsError("Preserve the existing study registration")
    manifest = read_json(ROOT / "models/word_generation_holdout_manifest.json")
    p = {"author": "Sonja Sahebzad", "recorded_utc": datetime.now(timezone.utc).isoformat(),
         "purpose": "Isolate complete-word candidate generation; no additional training",
         "model": "Unchanged operational local256 adapter on pinned Qwen3-1.7B-Base",
         "context_tokens": 128, "baseline_token_cutoff": 256, "ngram_candidates": 20,
         "beams": list(BEAMS), "maximum_word_tokens": MAX_TOKENS, "terminal_proposal_cap": CAP,
         "word_rule": "lowercase ASCII letters with internal apostrophes; training vocabulary only",
         "search": "Canonical-token trie. Beam by cumulative token log-probability; terminate only at vocabulary word nodes. Terminal nodes may continue. Rank at most128 completed words by token path score; rescore additions with unchanged canonical likelihood plus boundary mass. Preserve old free set and every old score.",
         "score_layout": "Original free batches first; new words in separate batches; all options separately after free predictions, identical for both arms. Baseline control equals operational free inference without options.",
         "development_cases": 600, "final_cases": 900,
         "selection": "Among beam16/beam64: integer top3 successes, then top1 successes, then mean latency on first20 development cases per source; then smaller beam",
         "primary": "Paired final exact free top3; promote only if stratified paired bootstrap95% lower bound>0 and exact two-sided McNemar p<0.05",
         "bootstrap_repetitions": 10000, "bootstrap_seed": 20261590,
         "secondary": "Top1, coverage, gains/losses, gains previously uncovered, losses, canonical single/multiple-token groups, source groups, synthetic four-choice accuracy and speed are descriptive; no posthoc speed promotion rule",
         "timing": "CUDA synchronized per-case end-to-end GPU inference excluding model/trie loading and R shortlist construction; alternating baseline/configuration order, first20 dev per source then all900 final. Measured predictions must equal joint evaluation outputs.",
         "final_only_after_selection": True,
         "numerics": "Shared original scores and option-free invariance exact. Separate FP16 cached/full checks use absolute log-score tolerance0.02 after a preserved neutral diagnostic; identical merged weights inFP32 agree within1e-4. No model accuracy outcomes used to set this bound.",
         "holdout_manifest_sha256": sha(ROOT / "models/word_generation_holdout_manifest.json"),
         "development_sha256": manifest["development_csv_sha256"],
         "final_sha256": manifest["final_csv_sha256"]}
    save_new(P, p)
    diagnostic = read_json(ROOT / "models/word_generation_neutral_precision_diagnostic.json")
    assert all(row["rank_identical"] for row in diagnostic["results"])
    assert all(row["max_absolute_error"] <= .0001 for row in diagnostic["results"] if row["precision"] == "float32")
    ranker = CompleteWordRanker()
    checks = neutral_checks(ranker)
    save_new(F, {"fingerprint": fingerprint(), "neutral_checks": checks,
                 "precision_diagnostic_sha256": sha(ROOT / "models/word_generation_neutral_precision_diagnostic.json"),
                 "trie_statistics": dict(ranker.trie.statistics),
                 "torch": torch_version(ranker), "gpu": ranker.torch.cuda.get_device_name()})
    print(json.dumps({"frozen": True, "checks": checks, "trie": dict(ranker.trie.statistics)}), flush=True)


def torch_version(ranker):
    return str(ranker.torch.__version__)


def read_cases(split):
    file = "data/adaptation/development.csv" if split == "development" else "data/word_generation/final_test.csv"
    with (ROOT / file).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(details, candidate):
    rows = []
    for d in details:
        baseline = dict(d["baseline"]["scores"])
        item = d["baseline"] if candidate == "baseline" else d["candidates"][candidate]
        union = baseline if candidate == "baseline" else dict(baseline, **dict(item["scores"]))
        actual = d["actual"]
        words = item["words"]
        assert words == sorted(union, key=lambda w: (-union[w], w))[:3]
        rows.append({"source": d["source"], "line_hash": d["line_hash"],
            "rank": words.index(actual)+1 if actual in words else 0,
            "shortlist_contains_target": actual in union,
            "choice_correct": d["options"]["words"][0] == actual})
    from learning_curve_audit import audit_details
    return audit_details(rows), rows


def evaluate(split):
    frozen()
    path = ROOT / "models" / f"word_generation_{split}.json"
    if path.exists():
        raise FileExistsError("Do not overwrite evaluated outcomes")
    if split == "final":
        selected = read_json(ROOT / "models/word_generation_selection.json")
        assert selected["development_artifact_sha256"] == sha(ROOT / "models/word_generation_development.json")
        assert selected["implementation_sha256"] == sha(F)
        beams = (int(selected["candidate"].removeprefix("beam")),)
    else:
        beams = BEAMS
    cases = read_cases(split)
    ranker = CompleteWordRanker()
    details = []
    for i, case in enumerate(cases):
        result = ranker.predict_variants(case["prefix"], json.loads(case["ngram_candidates_json"]), json.loads(case["options_json"]), beams)
        details.append({"source": case["source"], "line_hash": case["line_hash"],
                        "actual": case["actual"], "target_tokens": len(ranker._tokens(case["actual"])),
                        "target_in_vocabulary": case["actual"] in ranker.vocabulary, **result})
        if (i+1) % 50 == 0:
            print(f"{split} predictions {i+1}/{len(cases)}", flush=True)
    # Separate timing executes each condition fully; shared joint passes above
    # never masquerade as standalone latency measurements.
    seen = Counter()
    timing_indices = []
    for i, case in enumerate(cases):
        if split == "final" or seen[case["source"]] < 20:
            timing_indices.append(i)
            seen[case["source"]] += 1
    timings = {"baseline": [], **{f"beam{b}": [] for b in beams}}
    names = list(timings)
    for j, i in enumerate(timing_indices):
        case = cases[i]
        order = names[j % len(names):] + names[:j % len(names)]
        for name in order:
            ranker.torch.cuda.synchronize()
            start = time.perf_counter()
            trial = ranker.predict_variants(case["prefix"], json.loads(case["ngram_candidates_json"]),
                       json.loads(case["options_json"]), () if name == "baseline" else (int(name[4:]),))
            ranker.torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)*1000
            assert trial["baseline"] == details[i]["baseline"]
            assert trial["options"] == details[i]["options"]
            if name != "baseline":
                assert trial["candidates"][name] == details[i]["candidates"][name]
            timings[name].append({"source": case["source"], "line_hash": case["line_hash"], "milliseconds": elapsed})
        if (j+1) % 100 == 0:
            print(f"{split} independent timings {j+1}/{len(timing_indices)}", flush=True)
    summaries = {name: {**summarize(details, name)[0],
        "milliseconds": float(np.mean([x["milliseconds"] for x in times])),
        "timing_cases": len(times)} for name, times in timings.items()}
    artifact = {"split": split, "author": "Sonja Sahebzad", "implementation_sha256": sha(F),
                "protocol_sha256": sha(P), "summaries": summaries, "timings": timings,
                "details": details, "peak_gpu_mib": ranker.torch.cuda.max_memory_allocated()/1024**2}
    if split == "final":
        artifact["selection_sha256"] = sha(ROOT / "models/word_generation_selection.json")
    # Compact JSON keeps published case-level numerical evidence manageable.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(artifact, stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps(summaries, indent=2), flush=True)


def select():
    frozen()
    development = read_json(ROOT / "models/word_generation_development.json")
    assert development["split"] == "development" and development["implementation_sha256"] == sha(F)
    metrics = development["summaries"]
    candidates = [f"beam{b}" for b in BEAMS]
    winner = min(candidates, key=lambda name: (
        -sum(d["actual"] in d["candidates"][name]["words"] for d in development["details"]),
        -sum(d["actual"] == d["candidates"][name]["words"][0] for d in development["details"]),
        metrics[name]["milliseconds"], candidates.index(name)))
    result = {"candidate": winner, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "development_artifact_sha256": sha(ROOT / "models/word_generation_development.json"),
        "implementation_sha256": sha(F), "protocol_sha256": sha(P),
        "final_not_used": True, "development_summaries": metrics}
    save_new(ROOT / "models/word_generation_selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate", choices=("development", "final"))
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.evaluate:
        evaluate(args.evaluate)
    elif args.select:
        select()
    elif args.request:
        frozen()
        selection = read_json(ROOT / "models/word_generation_selection.json")
        beam = int(selection["candidate"][4:])
        request = read_json(args.request)
        result = CompleteWordRanker().predict_variants(request["phrase"], request.get("ngram_candidates", []), request.get("choices", []), (beam,))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
