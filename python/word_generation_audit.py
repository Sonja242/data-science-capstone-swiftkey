"""CPU validation for bounded multi-token word generation. Sonja Sahebzad.

Archived learning-curve code is reused read-only. This module never loads models,
runs a GPU, or discovers final data automatically. Final inputs must be explicit.
"""
from collections import Counter
import math
import numpy as np
from learning_curve_audit import (
    SOURCES, WORD, keys, audit_details, audit_partitions,
    paired_bootstrap, exact_paired_p,
)

CANDIDATES = ("beam16", "beam64")
PRIOR_FINAL_COUNTS = {"v3": 3000, "neural": 900, "adaptation": 900, "learning_curve": 900}


def audit_reserved_hashes(development, reserved, prior_finals, training_hashes,
                          prior_development_hashes=(), expected_prior_counts=PRIOR_FINAL_COUNTS,
                          development_per_source=200, final_per_source=300):
    """Accept hash/source manifests only; no reserved phrases or targets needed."""
    if set(prior_finals) != set(expected_prior_counts):
        raise ValueError("Every previously evaluated final set must be explicitly excluded")
    previous = set()
    for name, rows in prior_finals.items():
        current = {digest for _, digest in keys(rows)}
        if len(current) != expected_prior_counts[name]:
            raise ValueError(f"Incorrect prior final count: {name}")
        if previous & current:
            raise ValueError("Prior final manifests overlap unexpectedly")
        previous.update(current)
    result = audit_partitions(development, reserved,
        previous | set(prior_development_hashes), training_hashes,
        development_per_source, final_per_source)
    result["excluded_prior_final_unique_lines"] = len(previous)
    result["excluded_prior_final_counts"] = dict(expected_prior_counts)
    return result


def score_map(pairs, label, allow_empty=False):
    if not isinstance(pairs, list) or (not pairs and not allow_empty):
        raise ValueError(f"{label}: expected a list of word/score pairs")
    values = {}
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{label}: malformed score pair")
        word, score = pair
        if not isinstance(word, str) or not WORD.fullmatch(word):
            raise ValueError(f"{label}: word is not normalized English")
        if word in values:
            raise ValueError(f"{label}: duplicate word score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{label}: non-finite or nonnumeric score")
        # Canonical whole-word log likelihood plus boundary log mass cannot be positive.
        if score > 1e-6:
            raise ValueError(f"{label}: positive log likelihood")
        values[word] = float(score)
    return values


def ranking(scores):
    return sorted(scores, key=lambda word: (-scores[word], word))


def derive_artifacts(raw, candidates=CANDIDATES, expected_keys=None,
                     expected_split=None, terminal_cap=128, score_tolerance=.02):
    """Recompute every outcome from the canonical baseline and added-word scores.

    Raw details contain source, line_hash, actual, target_tokens,
    target_in_vocabulary, baseline, options and candidates. The same baseline
    scores are reused in every union; options never become free members.
    score_tolerance is the frozen neutral-reference value for corpus diagnostics,
    not a global corpus rejection threshold. Neutral tests remain external gates.
    """
    if not math.isfinite(score_tolerance) or score_tolerance < 0:
        raise ValueError("Invalid neutral score tolerance")
    if raw.get("split") not in ("development", "final"):
        raise ValueError("Artifact split must be development or final")
    if expected_split is not None and raw["split"] != expected_split:
        raise ValueError("Unexpected split")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Require unique declared candidate IDs")
    records = raw["details"]
    ordered = keys(records)
    if expected_keys is not None and ordered != list(expected_keys):
        raise ValueError("Ordered case identities differ")
    details = {name: [] for name in ("baseline", *candidates)}
    numerical_comparisons = []
    for row in records:
        target = row["actual"]
        if not isinstance(target, str) or not WORD.fullmatch(target):
            raise ValueError("Target is not a normalized word")
        if type(row["target_tokens"]) is not int or row["target_tokens"] < 1:
            raise ValueError("target_tokens must count canonical tokenizer tokens")
        if type(row["target_in_vocabulary"]) is not bool:
            raise ValueError("Vocabulary membership must be boolean")
        base = score_map(row["baseline"]["scores"], "baseline")
        base_order = ranking(base)
        if row["baseline"]["words"] != base_order[:3]:
            raise ValueError("Baseline words differ from canonical score ranking")
        option_scores = score_map(row["options"]["scores"], "options")
        if len(option_scores) != 4 or target not in option_scores:
            raise ValueError("Four unique options must include the observed target")
        option_order = ranking(option_scores)
        if row["options"]["words"] != option_order:
            raise ValueError("Option ranking differs from its independent scores")
        if set(row["candidates"]) != set(candidates):
            raise ValueError("Candidate configuration grid differs from the declared set")
        common = {"source": row["source"], "line_hash": row["line_hash"],
                  "target_tokens": row["target_tokens"],
                  "target_in_vocabulary": row["target_in_vocabulary"],
                  "choice_correct": option_order[0] == target}

        def detail(scores, order):
            position = order.index(target) + 1 if target in scores else 0
            margin = scores[order[2]] - scores[order[3]] if len(order) > 3 else None
            return {**common, "rank": position if 0 < position <= 3 else 0,
                    "top3_boundary_margin": margin,
                    "numerically_close_top3_boundary": margin is not None and margin <= 2 * score_tolerance,
                    "target_full_rank": position,
                    "shortlist_contains_target": target in scores,
                    "candidate_count": len(scores)}

        details["baseline"].append(detail(base, base_order))
        seen_scores = {word: [("baseline", value)] for word, value in base.items()}

        def record_comparison(word, left_label, left_score, right_label, right_score, kind):
            difference = abs(right_score - left_score)
            numerical_comparisons.append({"source": row["source"], "line_hash": row["line_hash"],
                "word": word, "kind": kind, "left": left_label, "right": right_label,
                "left_log_score": left_score, "right_log_score": right_score,
                "absolute_difference": difference, "outside_neutral_reference": difference > score_tolerance})
        for candidate in candidates:
            proposal = row["candidates"][candidate]
            added = score_map(proposal["scores"], candidate, allow_empty=True)
            if len(added) > terminal_cap:
                raise ValueError("More scored additions than the frozen terminal cap")
            if set(added) & set(base):
                raise ValueError("Added-only scores must not overwrite baseline scores")
            for word, value in added.items():
                for label, previous in seen_scores.get(word, []):
                    record_comparison(word, label, previous, candidate, value, "cross_beam")
                seen_scores.setdefault(word, []).append((candidate, value))
            union = {**base, **added}
            order = ranking(union)
            if proposal["words"] != order[:3]:
                raise ValueError("Expanded words differ from canonical union ranking")
            expanded = detail(union, order)
            original = details["baseline"][-1]
            if original["target_full_rank"] and expanded["target_full_rank"] < original["target_full_rank"]:
                raise ValueError("Covered target improved despite unchanged scores and a superset")
            details[candidate].append(expanded)
        for word, value in option_scores.items():
            for label, previous in seen_scores.get(word, []):
                record_comparison(word, label, previous, "options", value, "option_vs_free")
    outside = [row for row in numerical_comparisons if row["outside_neutral_reference"]]
    maximum = max(numerical_comparisons, key=lambda row: row["absolute_difference"], default=None)
    diagnostics = {"scope": "Independent corpus batches; descriptive comparison with the fixed neutral reference, not a corpus-wide correctness gate",
        "neutral_log_score_reference": score_tolerance,
        "status": "outside_neutral_reference_observed" if outside else
                  "within_neutral_reference_for_compared_scores" if numerical_comparisons else "no_common_word_comparisons",
        "comparison_count": len(numerical_comparisons), "outside_reference_count": len(outside),
        "affected_case_count": len({row["line_hash"] for row in outside}),
        "max_absolute_difference": maximum["absolute_difference"] if maximum else None,
        "maximum_comparison": maximum, "comparisons": numerical_comparisons,
        "decision_effect": "None: stored canonical scores, rankings, selection and promotion are unchanged. Preserve and investigate unusual discrepancies before final interpretation."}
    return {name: {"candidate": name, "split": raw["split"], "details": values,
                   "summary": audit_details(values, ordered), "hard_invariants_passed": True,
                   "numerical_diagnostics": diagnostics} for name, values in details.items()}


def select_generation(raw, milliseconds, candidate_order=CANDIDATES, score_tolerance=.02, expected_cases=600):
    """Exact-count selection from all development configurations, never final."""
    if len(raw["details"]) != expected_cases:
        raise ValueError("Development case count differs from protocol")
    artifacts = derive_artifacts(raw, candidate_order, expected_split="development", score_tolerance=score_tolerance)
    if set(milliseconds) != set(candidate_order):
        raise ValueError("Supply independent latency for every declared candidate")
    rows = []
    for name in candidate_order:
        elapsed = milliseconds[name]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid independently measured latency")
        artifact = artifacts[name]
        rows.append({"candidate": name, **artifact["summary"],
                     "top3_correct": sum(r["rank"] > 0 for r in artifact["details"]),
                     "top1_correct": sum(r["rank"] == 1 for r in artifact["details"]),
                     "milliseconds": float(elapsed)})
    chosen = min(rows, key=lambda row: (-row["top3_correct"], -row["top1_correct"],
                   row["milliseconds"], candidate_order.index(row["candidate"])))
    return {"candidate": chosen["candidate"], "selected": chosen,
            "development": rows, "baseline": artifacts["baseline"]["summary"],
            "numerical_diagnostics": artifacts["baseline"]["numerical_diagnostics"],
            "rule": "Integer top-3 successes, integer top-1 successes, independent latency, fixed candidate order"}


def paired_counts(baseline, expanded):
    if keys(baseline) != keys(expanded):
        raise ValueError("Paired identities differ")
    audit_details(baseline)
    audit_details(expanded)
    gains = sum(a["rank"] == 0 and b["rank"] > 0 for a, b in zip(baseline, expanded))
    losses = sum(a["rank"] > 0 and b["rank"] == 0 for a, b in zip(baseline, expanded))
    covered_gains = sum(a["rank"] == 0 and b["rank"] > 0 and a["shortlist_contains_target"]
                        for a, b in zip(baseline, expanded))
    lost_coverage = sum(a["shortlist_contains_target"] and not b["shortlist_contains_target"]
                       for a, b in zip(baseline, expanded))
    if covered_gains or lost_coverage:
        raise ValueError("Results violate the fixed-score candidate-superset design")
    n = len(baseline)
    old_correct = sum(r["rank"] > 0 for r in baseline)
    new_correct = sum(r["rank"] > 0 for r in expanded)
    if new_correct - old_correct != gains - losses:
        raise ValueError("Paired transition counts do not reconcile")
    return {"cases": n, "baseline_top3_correct": old_correct,
            "expanded_top3_correct": new_correct, "gains": gains, "losses": losses,
            "both_correct": sum(a["rank"] > 0 and b["rank"] > 0 for a, b in zip(baseline, expanded)),
            "both_incorrect": sum(a["rank"] == 0 and b["rank"] == 0 for a, b in zip(baseline, expanded)),
            "difference": (new_correct - old_correct) / n,
            "baseline_coverage": sum(r["shortlist_contains_target"] for r in baseline),
            "expanded_coverage": sum(r["shortlist_contains_target"] for r in expanded),
            "gains_from_previously_missing_target": gains,
            "unexpected_covered_target_gains": covered_gains,
            "lost_coverage": lost_coverage,
            "baseline_close_boundary_cases": sum(r.get("numerically_close_top3_boundary", False) for r in baseline),
            "expanded_close_boundary_cases": sum(r.get("numerically_close_top3_boundary", False) for r in expanded),
            "gains_at_close_expanded_boundary": sum(a["rank"] == 0 and b["rank"] > 0 and
                b.get("numerically_close_top3_boundary", False) for a, b in zip(baseline, expanded)),
            "losses_at_close_expanded_boundary": sum(a["rank"] > 0 and b["rank"] == 0 and
                b.get("numerically_close_top3_boundary", False) for a, b in zip(baseline, expanded))}


def primary_comparison(raw, selected_candidate, expected_keys=None, bootstrap_seed=20261590, score_tolerance=.02, expected_cases=900):
    """One frozen primary contrast; token/source groups are descriptive only."""
    if len(raw["details"]) != expected_cases:
        raise ValueError("Final case count differs from protocol")
    artifacts = derive_artifacts(raw, (selected_candidate,), expected_keys, expected_split="final", score_tolerance=score_tolerance)
    baseline = artifacts["baseline"]["details"]
    expanded = artifacts[selected_candidate]["details"]
    source_counts = Counter(r["source"] for r in baseline)
    if set(source_counts) != set(SOURCES) or len(set(source_counts.values())) != 1:
        raise ValueError("Primary requires the source-balanced protocol")
    paired = paired_counts(baseline, expanded)
    a = np.array([r["rank"] > 0 for r in baseline], dtype=int)
    b = np.array([r["rank"] > 0 for r in expanded], dtype=int)
    interval = paired_bootstrap(b - a, [r["source"] for r in baseline], seed=bootstrap_seed)
    interval["exact_paired_p"] = exact_paired_p(a, b)
    interval["promote"] = interval["interval"][0] > 0 and interval["exact_paired_p"] < .05
    groups = []
    for label, predicate in (("single_token", lambda row: row["target_tokens"] == 1),
                             ("multi_token", lambda row: row["target_tokens"] > 1),
                             *((s, lambda row, source=s: row["source"] == source) for s in SOURCES)):
        indices = [i for i, row in enumerate(baseline) if predicate(row)]
        groups.append({"group": label, **(paired_counts([baseline[i] for i in indices],
                       [expanded[i] for i in indices]) if indices else {"cases": 0})})
    if any(x["choice_correct"] != y["choice_correct"] for x, y in zip(baseline, expanded)):
        raise ValueError("Fixed independently scored options changed across generation arms")
    return {"selected_candidate": selected_candidate, "primary_top3": {**interval, **paired},
            "baseline_metrics": artifacts["baseline"]["summary"],
            "expanded_metrics": artifacts[selected_candidate]["summary"],
            "descriptive_groups": groups,
            "numerical_diagnostics": artifacts["baseline"]["numerical_diagnostics"],
            "close_boundary_note": f"Descriptive only: third/fourth log-score margin <= {2 * score_tolerance:g}; no change to scoring, selection or promotion. Neutral numerical checks do not guarantee an error bound for every corpus case.",
            "choice_interpretation": "Identical fixed-model option scores in both arms; not evidence that generation improves multiple-choice accuracy.",
            "limitations": "One normalized observed next word per line. Token groups are descriptive; repeated development use, related documents and unknown pretrained overlap remain possible."}