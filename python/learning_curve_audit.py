"""Independent CPU checks for the preregistered learning-curve study.

Author: Sonja Sahebzad. This module never loads model weights or launches a GPU.
New held-out sentences are not needed: partition auditing accepts hash manifests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

SOURCES = ("blogs", "news", "twitter")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*\Z")
SEEDS = (20261410, 20261411, 20261412)


def keys(rows):
    result = [(row["source"], row["line_hash"]) for row in rows]
    if not result:
        raise ValueError("Empty case set")
    if len({key[1] for key in result}) != len(result):
        raise ValueError("Repeated normalized line: line is the resampling unit")
    if any(source not in SOURCES for source, _ in result):
        raise ValueError("Unrecognized source")
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for _, digest in result):
        raise ValueError("Invalid normalized-line SHA256")
    return result


def audit_partitions(development, final_manifest, prior_case_hashes,
                     training_hashes, development_per_source=200,
                     final_per_source=300):
    """Audit sanitized metadata, without needing prefixes, targets or outcomes."""
    dev_keys, test_keys = keys(development), keys(final_manifest)
    dev = {key[1] for key in dev_keys}
    test = {key[1] for key in test_keys}
    if Counter(key[0] for key in dev_keys) != Counter({s: development_per_source for s in SOURCES}):
        raise ValueError("Development source counts differ from protocol")
    if Counter(key[0] for key in test_keys) != Counter({s: final_per_source for s in SOURCES}):
        raise ValueError("Final source counts differ from protocol")
    if test & (dev | set(prior_case_hashes)):
        raise ValueError("Final cases overlap current development or previous evaluations")
    if (dev | test) & set(training_hashes):
        raise ValueError("Normalized training lines overlap held-out lines")
    return {"passed": True, "development_cases": len(dev), "final_cases": len(test),
            "line_overlap": 0,
            "limits": "Exact normalized-line separation does not exclude near duplicates or pretraining overlap."}


def audit_details(details, expected_keys=None):
    actual_keys = keys(details)
    if expected_keys is not None and actual_keys != list(expected_keys):
        raise ValueError("Ordered case identities differ; do not zip misaligned results")
    for row in details:
        if type(row.get("rank")) is not int or row["rank"] not in (0, 1, 2, 3):
            raise ValueError("rank must be 0 for absent from top-3, or 1, 2, 3")
        if type(row.get("choice_correct")) is not bool:
            raise ValueError("choice_correct must be a boolean")
        if type(row.get("shortlist_contains_target")) is not bool:
            raise ValueError("shortlist_contains_target must be a boolean")
        if row["rank"] and not row["shortlist_contains_target"]:
            raise ValueError("Top-3 target cannot be absent from the shortlist")
    n = len(details)
    return {"cases": n, "top1": sum(r["rank"] == 1 for r in details) / n,
            "top3": sum(r["rank"] > 0 for r in details) / n,
            "mrr_at3": sum(1 / r["rank"] if r["rank"] else 0 for r in details) / n,
            "synthetic_choice_accuracy": sum(r["choice_correct"] for r in details) / n,
            "shortlist_recall": sum(r["shortlist_contains_target"] for r in details) / n}


def audit_prediction(case, output, without_choices=None):
    """Independently reconstruct the scoring detail from an optional raw output."""
    target = case["actual"]
    options = case.get("options")
    if options is None:
        options = json.loads(case["options_json"])
    if len(options) != 4 or len(set(options)) != 4 or target not in options:
        raise ValueError("Four distinct synthetic options must include the target exactly once")
    if any(not WORD.fullmatch(word) for word in [target, *options]):
        raise ValueError("Targets/options must use the documented normalized word definition")
    words, shortlist, choices = output["words"], output["shortlist"], output["choices"]
    if not 1 <= len(words) <= 3 or len(set(words)) != len(words):
        raise ValueError("Free output must contain one to three distinct words")
    if len(set(shortlist)) != len(shortlist) or not set(words) <= set(shortlist):
        raise ValueError("Free predictions must be unique members of the independent shortlist")
    if len(choices) != 4 or set(choices) != set(options):
        raise ValueError("Choice output must be a permutation of the supplied options")
    if any(not WORD.fullmatch(word) for word in [*words, *shortlist]):
        raise ValueError("Free predictions contain a non-normalized word")
    if without_choices is not None:
        if words != without_choices["words"] or shortlist != without_choices["shortlist"]:
            raise ValueError("Supplying options altered free predictions or the free shortlist")
    row = {"source": case["source"], "line_hash": case["line_hash"],
           "rank": words.index(target) + 1 if target in words else 0,
           "choice_correct": choices[0] == target,
           "shortlist_contains_target": target in shortlist}
    audit_details([row])
    return row


def audit_artifact(artifact, expected_keys=None):
    computed = audit_details(artifact["details"], expected_keys)
    for name, value in computed.items():
        if name in artifact.get("summary", {}) and not math.isclose(
                value, artifact["summary"][name], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Stored metric does not match details: {name}")
    return computed


def select_on_development(results, checkpoints=(256, 512, 1024),
                          shortlists=(64,), seeds=SEEDS):
    """Select groups by mean top-3, top-1, speed; representative by proximity.

    Results have seed, steps, shortlist, split, summary and details keys.
    All candidates must cover exactly the same development cases. No test input
    is accepted here. Fixed checkpoint/shortlist/seed order resolves exact ties.
    """
    expected = None
    groups = defaultdict(dict)
    for result in results:
        if result.get("split") != "development":
            raise ValueError("Selection accepts development results only")
        group = (result["steps"], result["shortlist"])
        if group[0] not in checkpoints or group[1] not in shortlists or result["seed"] not in seeds:
            raise ValueError("Unexpected candidate not listed in the frozen protocol")
        if result["seed"] in groups[group]:
            raise ValueError("Duplicate seed result")
        current = keys(result["details"])
        if expected is None:
            expected = current
        computed = audit_artifact(result, expected)
        latency = result["summary"]["milliseconds"]
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid latency")
        groups[group][result["seed"]] = {**computed, "milliseconds": latency}
    expected_groups = {(step, width) for step in checkpoints for width in shortlists}
    if set(groups) != expected_groups or any(set(group) != set(seeds) for group in groups.values()):
        raise ValueError("Incomplete checkpoint/shortlist/seed grid")
    averages = []
    for step in checkpoints:
        for width in shortlists:
            rows = groups[(step, width)]
            averages.append({"steps": step, "shortlist": width, **{
                metric: sum(rows[seed][metric] for seed in seeds) / len(seeds)
                for metric in ("top3", "top1", "milliseconds")}})
    winner = min(averages, key=lambda x: (-x["top3"], -x["top1"], x["milliseconds"],
                 checkpoints.index(x["steps"]), shortlists.index(x["shortlist"])))
    group = groups[(winner["steps"], winner["shortlist"])]
    representative = min(seeds, key=lambda seed: (
        abs(group[seed]["top3"] - winner["top3"]),
        abs(group[seed]["top1"] - winner["top1"]), seeds.index(seed)))
    return {"selected": winner, "representative_seed": representative,
            "development_groups": averages,
            "rule": "Mean top-3, mean top-1, mean latency; representative nearest mean top-3, then mean top-1, then fixed seed order"}



def select_development_summary(rows, checkpoints=(256, 512, 1024),
                               shortlists=(64, 256), seeds=SEEDS):
    """Select from validated flat development summaries.

    Caller must run artifact/alignment checks first: summary rows alone cannot
    prove case identity. Cases must be equal across the complete frozen grid.
    """
    groups = defaultdict(dict)
    cases = set()
    for row in rows:
        if row.get("split", "development") != "development":
            raise ValueError("Selection accepts development results only")
        group = (row["steps"], row["shortlist"])
        seed = row["seed"]
        if group[0] not in checkpoints or group[1] not in shortlists or seed not in seeds:
            raise ValueError("Unexpected candidate not listed in the frozen protocol")
        if seed in groups[group]:
            raise ValueError("Duplicate seed result")
        if type(row["cases"]) is not int or row["cases"] < 1:
            raise ValueError("Invalid case count")
        cases.add(row["cases"])
        if not (0 <= row["top1"] <= row["top3"] <= 1):
            raise ValueError("Invalid top-1/top-3 metrics")
        if not math.isfinite(row["milliseconds"]) or row["milliseconds"] <= 0:
            raise ValueError("Invalid latency")
        groups[group][seed] = row
    expected_groups = {(step, width) for step in checkpoints for width in shortlists}
    if len(cases) != 1 or set(groups) != expected_groups or any(set(group) != set(seeds) for group in groups.values()):
        raise ValueError("Incomplete checkpoint/shortlist/seed grid or unequal case counts")
    averages = []
    for step in checkpoints:
        for width in shortlists:
            group = groups[(step, width)]
            averages.append({"steps": step, "shortlist": width, **{
                metric: sum(group[seed][metric] for seed in seeds) / len(seeds)
                for metric in ("top3", "top1", "milliseconds")}})
    winner = min(averages, key=lambda x: (-x["top3"], -x["top1"], x["milliseconds"],
                 checkpoints.index(x["steps"]), shortlists.index(x["shortlist"])))
    group = groups[(winner["steps"], winner["shortlist"])]
    representative = min(seeds, key=lambda seed: (
        abs(group[seed]["top3"] - winner["top3"]),
        abs(group[seed]["top1"] - winner["top1"]), seeds.index(seed)))
    return {"selected": winner, "representative_seed": representative,
            "development_groups": averages,
            "rule": "Mean top-3, mean top-1, mean latency; representative nearest mean top-3, then mean top-1, then fixed seed order"}

def paired_bootstrap(differences, sources, repetitions=10000, seed=20261490, confidence=.95):
    """Line bootstrap; equal source weights; model seeds stay fixed.

    One numeric difference per unique line must already be supplied. In
    particular, average seed outcomes per line BEFORE calling this function.
    Never flatten seed x line observations into independent replicates.
    """
    values = np.asarray(differences, dtype=float)
    labels = np.asarray(sources)
    if values.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("Require one finite difference per line")
    if set(labels) != set(SOURCES) or any(sum(labels == s) < 2 for s in SOURCES):
        raise ValueError("All three source strata require at least two lines")
    if not 0 < confidence < 1 or repetitions < 1000:
        raise ValueError("Invalid confidence level or too few bootstrap replicates")
    rng = np.random.default_rng(seed)
    boot = np.zeros(repetitions)
    means = []
    for source in SOURCES:
        group = values[labels == source]
        means.append(float(group.mean()))
        # Generate bounded-size batches even for larger audit datasets.
        for start in range(0, repetitions, 500):
            stop = min(repetitions, start + 500)
            boot[start:stop] += rng.choice(group, size=(stop - start, len(group)), replace=True).mean(axis=1) / 3
    tail = (1 - confidence) / 2
    return {"difference": sum(means) / 3, "confidence": confidence,
            "interval": np.quantile(boot, [tail, 1 - tail]).tolist(),
            "replicates": repetitions, "bootstrap_seed": seed,
            "estimand": "Equal-source mean difference across unique normalized lines; fixed model seeds"}


def exact_paired_p(baseline, challenger):
    """Two-sided exact McNemar test for one paired binary endpoint."""
    if len(baseline) != len(challenger):
        raise ValueError('Paired binary outcomes differ in length')
    if any(value not in (0, 1, False, True) for value in [*baseline, *challenger]):
        raise ValueError('Paired outcomes must be binary')
    wins = sum(bool(b) and not bool(a) for a, b in zip(baseline, challenger))
    losses = sum(bool(a) and not bool(b) for a, b in zip(baseline, challenger))
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (1 << discordant)
    return min(1.0, 2 * tail)


def confirmatory_comparison(baseline, representative, replicas=(), bootstrap_seed=20261490):
    """One confirmatory deployment contrast; replica mean remains secondary.

    The caller must freeze representative identity on development first.
    New final outcomes are intentionally not read automatically by this module.
    """
    ordered = keys(baseline["details"])
    audit_artifact(baseline, ordered)
    audit_artifact(representative, ordered)
    counts = Counter(source for source, _ in ordered)
    if len(set(counts.values())) != 1:
        raise ValueError("Exact pooled paired gate assumes the balanced-source protocol")
    a = np.array([row["rank"] > 0 for row in baseline["details"]], dtype=float)
    b = np.array([row["rank"] > 0 for row in representative["details"]], dtype=float)
    sources = [source for source, _ in ordered]
    primary = paired_bootstrap(b - a, sources, seed=bootstrap_seed)
    primary["exact_paired_p"] = exact_paired_p(a, b)
    primary["promote"] = primary["interval"][0] > 0 and primary["exact_paired_p"] < .05
    output = {"primary_top3": primary,
              "rule": "One development-selected representative versus prior operational model; positive 95% lower bound and exact paired p below 0.05"}
    if replicas:
        seed_ids = [replica["seed"] for replica in replicas]
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError("Duplicate seed in replica analysis")
        for replica in replicas:
            audit_artifact(replica, ordered)
        matrix = np.array([[row["rank"] > 0 for row in replica["details"]] for replica in replicas], dtype=float)
        output["secondary_seed_mean_top3"] = paired_bootstrap(matrix.mean(axis=0) - a, sources,
                                                                seed=bootstrap_seed + 1)
        output["secondary_seed_mean_top3"]["limitation"] = "Descriptive conditional interval for the evaluated seeds; not uncertainty over future random seeds or an ensemble score"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", default=[],
                        help="Explicit artifact to verify; no files are found automatically")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.artifact:
        artifact = json.loads(path.read_text(encoding="utf-8-sig"))
        results.append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "metrics": audit_artifact(artifact)})
    payload = {"passed": True, "audited_artifacts": results,
               "note": "Metrics and internal consistency only; selection and disjointness need explicit corresponding manifests."}
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()