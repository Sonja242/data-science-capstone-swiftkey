"""CPU-only audit of archived DEVELOPMENT predictions. Author: Sonja Sahebzad.

No model weights, GPU, training or final-test files are used. A tokenizer is
loaded only as a CPU text processor. The audit never rewrites its source data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

CANDIDATES = ("base64", "base256", "local256", "augmented256")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*\Z")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def category(detail):
    if detail["rank"] > 0:
        return "top3_correct"
    return "present_outside_top3" if detail["shortlist_contains_target"] else "target_absent"


def metrics(details):
    n = len(details)
    counts = Counter(category(d) for d in details)
    covered = sum(d["shortlist_contains_target"] for d in details)
    return {"cases": n, "top1_count": sum(d["rank"] == 1 for d in details),
            "top3_count": counts["top3_correct"], "covered_count": covered,
            "present_outside_top3_count": counts["present_outside_top3"],
            "target_absent_count": counts["target_absent"],
            "top3": counts["top3_correct"] / n if n else None,
            "coverage": covered / n if n else None,
            "top3_given_covered": counts["top3_correct"] / covered if covered else None,
            "synthetic_choice_correct": sum(d["choice_correct"] for d in details)}


def validate_artifact(artifact, rows, dataset_sha):
    assert artifact["split"] == "development", "Only development artifacts are allowed"
    assert artifact["data_sha256"] == dataset_sha, "Dataset fingerprint mismatch"
    details = artifact["details"]
    assert len(details) == len(rows)
    for row, detail in zip(rows, details):
        assert (row["source"], row["line_hash"]) == (detail["source"], detail["line_hash"])
        assert detail["rank"] in (0, 1, 2, 3)
        assert not detail["rank"] or detail["shortlist_contains_target"]
        assert detail["context_words"] == len(row["prefix"].split())
        assert detail["target_characters"] == len(row["actual"])
    observed = metrics(details)
    assert observed["cases"] == artifact["summary"]["cases"]
    assert abs(observed["top3"] - artifact["summary"]["top3"]) < 1e-12
    assert abs(observed["coverage"] - artifact["summary"]["shortlist_recall"]) < 1e-12
    return observed


def paired_transition(left, right):
    assert len(left) == len(right)
    transitions = Counter((category(a), category(b)) for a, b in zip(left, right))
    return {"newly_covered": sum(not a["shortlist_contains_target"] and b["shortlist_contains_target"]
                                 for a, b in zip(left, right)),
            "lost_coverage": sum(a["shortlist_contains_target"] and not b["shortlist_contains_target"]
                                for a, b in zip(left, right)),
            "new_top3": sum(a["rank"] == 0 and b["rank"] > 0 for a, b in zip(left, right)),
            "lost_top3": sum(a["rank"] > 0 and b["rank"] == 0 for a, b in zip(left, right)),
            "same_recorded_rank": sum(a["rank"] == b["rank"] for a, b in zip(left, right)),
            "transitions": [{"from": a, "to": b, "cases": n} for (a, b), n in sorted(transitions.items())]}


def tokenize_audit(root, rows, local_details):
    # tokenizers is a CPU Rust tokenizer library, not a neural inference engine.
    from tokenizers import Tokenizer
    path = root / "models/neural/Qwen3-1.7B-Base/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    vocabulary = {row["word"] for row in read_csv(root / "data/neural_evaluation/vocabulary.csv")}
    token_count = tokenizer.get_vocab_size(with_added_tokens=True)
    decoded = tokenizer.decode_batch([[i] for i in range(token_count)], skip_special_tokens=False)
    eligible = [i for i, token in enumerate(decoded)
                if token.startswith(" ") and WORD.fullmatch(token.strip().lower())
                and token.strip().lower() in vocabulary]
    mapped = Counter(decoded[i].strip().lower() for i in eligible)
    canonical = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    target_lengths = [len(canonical(" " + r["actual"])) for r in rows]
    representable = [r["actual"] in mapped for r in rows]
    ngram_contains = [r["actual"] in json.loads(r["ngram_candidates_json"]) for r in rows]
    wrong_segmentations = 0
    comparisons = 0
    for row in rows:
        prefix = canonical(row["prefix"])
        for word in {row["actual"], *json.loads(row["ngram_candidates_json"])}:
            comparisons += 1
            wrong_segmentations += canonical(row["prefix"] + " " + word) != prefix + canonical(" " + word)
    initial_byte_replacement = sum(token.startswith("\ufffd") for token in decoded)
    breakdown = []
    for label, mask in (("one canonical token", [n == 1 for n in target_lengths]),
                        ("multiple canonical tokens", [n > 1 for n in target_lengths]),
                        ("not proposed by any eligible single token", [not x for x in representable]),
                        ("target in R top20", ngram_contains),
                        ("target not in R top20", [not x for x in ngram_contains])):
        breakdown.append({"group": label, **metrics([d for d, keep in zip(local_details, mask) if keep])})
    # A target not reachable by either generator cannot appear in a free shortlist.
    impossible = [not neural and not ng for neural, ng in zip(representable, ngram_contains)]
    assert not any(d["shortlist_contains_target"] and bad for d, bad in zip(local_details, impossible))
    return {"tokenizer_sha256": sha(path), "token_count": token_count,
            "eligible_token_ids": len(eligible), "unique_lowercase_eligible_words": len(mapped),
            "eligible_ids_with_case_fold_change": sum(decoded[i].strip() != decoded[i].strip().lower() for i in eligible),
            "eligible_ids_not_equal_to_single_canonical_id": sum(canonical(" " + decoded[i].strip().lower()) != [i] for i in eligible),
            "words_represented_by_multiple_eligible_ids": sum(n > 1 for n in mapped.values()),
            "target_words_outside_training_vocabulary": sum(r["actual"] not in vocabulary for r in rows),
            "unreachable_by_either_candidate_generator": sum(impossible),
            "prefixes_over_128_tokens": sum(len(canonical(r["prefix"])) > 128 for r in rows),
            "prefix_target_concatenation_checks": comparisons,
            "prefix_target_concatenation_mismatches": wrong_segmentations,
            "individually_decoded_tokens_starting_replacement_character": initial_byte_replacement,
            "subgroups": breakdown,
            "limitations": ["Canonical path scores do not sum all tokenizations or capitalization variants.",
                            "A replacement character from isolated byte-token decoding is not proof of a wrong boundary probability.",
                            "Full shortlists and per-word scores were not retained in the archived artifacts."]}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.out_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError("Audit output must be outside the live project")
    source = root / "data/adaptation/development.csv"
    rows = read_csv(source)
    dataset_sha = sha(source)
    artifacts = {}
    inputs = {str(source.relative_to(root)): dataset_sha}
    base_code_sha = sha(root / "python/neural_predictor.py")
    comparison_code_sha = code_sha(root / "python/adaptation_experiment.py")
    summary = []
    groups = []
    for candidate in CANDIDATES:
        path = root / "models" / f"adaptation_{candidate}_development.json"
        artifact = read_json(path)
        assert artifact["candidate"] == candidate
        assert artifact["base_code_sha256"] == base_code_sha, "Archived inference source changed"
        assert artifact["code_sha256"] == comparison_code_sha, "Comparison source changed"
        artifacts[candidate] = artifact
        inputs[str(path.relative_to(root))] = sha(path)
        summary.append({"candidate": candidate, **validate_artifact(artifact, rows, dataset_sha)})
        filters = [("source", source_name, [r["source"] == source_name for r in rows])
                   for source_name in sorted({r["source"] for r in rows})]
        filters += [("context words", label, [lo <= len(r["prefix"].split()) <= hi for r in rows])
                    for label, lo, hi in (("1-3", 1, 3), ("4-9", 4, 9), ("10+", 10, 10**9))]
        filters += [("target characters", label, [lo <= len(r["actual"]) <= hi for r in rows])
                    for label, lo, hi in (("1-3", 1, 3), ("4-6", 4, 6), ("7+", 7, 10**9))]
        for dimension, label, mask in filters:
            groups.append({"candidate": candidate, "dimension": dimension, "group": label,
                           **metrics([d for d, keep in zip(artifact["details"], mask) if keep])})
    transitions = {f"{a}_to_{b}": paired_transition(artifacts[a]["details"], artifacts[b]["details"])
                   for a, b in (("base64", "base256"), ("base256", "local256"), ("local256", "augmented256"))}
    tokenization = tokenize_audit(root, rows, artifacts["local256"]["details"])
    inputs["python/neural_predictor.py"] = sha(root / "python/neural_predictor.py")
    result = {"author": "Sonja Sahebzad", "split": "development", "summary": summary,
              "paired_transitions": transitions, "tokenization": tokenization,
              "input_sha256": inputs, "audit_code_sha256_canonical_utf8": code_sha(__file__),
              "data_access": "Only named development artifacts, vocabulary, tokenizer and inference source were read. No final test was opened.",
              "rank_warning": "rank=0 censors every rank outside the top three, including absent targets."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "development_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out / "development_summary.csv", summary)
    write_csv(out / "development_subgroups.csv", groups)
    write_csv(out / "tokenization_subgroups.csv", tokenization["subgroups"])
    print(json.dumps({"summary": summary, "paired_transitions": transitions, "tokenization": tokenization}, indent=2))


if __name__ == "__main__":
    main()
