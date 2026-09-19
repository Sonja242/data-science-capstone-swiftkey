"""Select on development data, then evaluate one frozen candidate. Sonja Sahebzad."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_cases(split):
    with (ROOT / "data/neural_evaluation" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify_artifact(artifact, split):
    code = ROOT / "python/neural_predictor.py"
    data = ROOT / "data/neural_evaluation" / f"{split}.csv"
    assert artifact["code_sha256"] == hashlib.sha256(code.read_bytes()).hexdigest()
    assert artifact["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    cases = read_cases(split)
    assert len(cases) == artifact["summary"]["cases"]
    assert [(x["source"], x["line_hash"]) for x in cases] == [
        (x["source"], x["line_hash"]) for x in artifact["details"]]


def write_csv(name, rows):
    with (ROOT / "models" / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, details, neural):
    if neural:
        rank = np.array([int(x["rank"]) for x in details])
        correct = np.array([x["choice_correct"] for x in details])
    else:
        rank = np.array([int(x["ngram_rank"]) for x in rows])
        correct = np.array([x["ngram_choice"] == x["actual"] for x in rows])
    return {"cases": len(rows), "top1": float(np.mean(rank == 1)),
            "top3": float(np.mean(rank > 0)), "synthetic_choice_accuracy": float(np.mean(correct)),
            "mrr_at3": float(np.mean(np.divide(1., rank, out=np.zeros(len(rank)), where=rank>0)))}


def main():
    candidates = []
    for size in ("0.6B", "1.7B"):
        artifact = read_json(ROOT / "models" / f"neural_{size}_development.json")
        verify_artifact(artifact, "development")
        candidates.append((size, artifact))
    # Predeclared primary criterion: top-three accuracy, then four-choice accuracy,
    # then speed. Test results never enter this decision.
    size, selected = max(candidates, key=lambda x: (
        x[1]["summary"]["top3"], x[1]["summary"]["synthetic_choice_accuracy"],
        -x[1]["summary"]["milliseconds"]))
    selection = {"size": size, "model": selected["summary"]["model"],
        "criterion": "Development top-3; ties: synthetic four-choice accuracy, then speed.",
        "revision": selected["download"]["revision"], "code_sha256": selected["code_sha256"],
        "development_sha256": selected["data_sha256"]}
    path = ROOT / "models/neural_selection.json"
    if path.exists():
        assert read_json(path) == selection, "Selection changed; do not reuse this final test."
    else:
        path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("Selection frozen before final test:", size, flush=True)
    test_path = ROOT / "models" / f"neural_{size}_test.json"
    if not test_path.exists():
        subprocess.run([sys.executable, str(ROOT / "python/neural_predictor.py"),
                        "--size", size, "--evaluate", "test"], check=True, cwd=ROOT)
    test = read_json(test_path)
    verify_artifact(test, "test")
    assert test["download"] == selected["download"]
    rows = read_cases("test")
    baseline = metrics(rows, None, False)
    new = metrics(rows, test["details"], True)
    comparison = [{"model": "Local Kneser-Ney v3", **baseline},
                  {"model": f"Qwen3-{size}-Base + local shortlist", **new}]
    write_csv("neural_comparison_test.csv", comparison)
    development = [{"model": "Local Kneser-Ney v3", **metrics(read_cases("development"), None, False),
                    "milliseconds": ""}]
    for candidate_size, result in candidates:
        development.append({"model": f"Qwen3-{candidate_size}-Base + local shortlist",
            **metrics(read_cases("development"), result["details"], True),
            "milliseconds": result["summary"]["milliseconds"]})
    write_csv("neural_comparison_development.csv", development)
    per_source = []
    for source in sorted({r["source"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["source"] == source]
        source_rows = [rows[i] for i in ids]
        source_details = [test["details"][i] for i in ids]
        per_source.extend([
            {"source": source, "model": "Local Kneser-Ney v3", **metrics(source_rows, None, False)},
            {"source": source, "model": f"Qwen3-{size}-Base + local shortlist",
             **metrics(source_rows, source_details, True)}])
    write_csv("neural_comparison_sources.csv", per_source)
    paired = np.array([int(d["rank"]>0)-int(int(r["ngram_rank"])>0)
                       for r,d in zip(rows,test["details"])])
    rng = np.random.default_rng(20261221)
    # Preserve the equal-source design when resampling paired outcomes.
    samples = np.zeros(10000)
    for source in sorted({r["source"] for r in rows}):
        group = paired[[i for i,r in enumerate(rows) if r["source"] == source]]
        samples += rng.choice(group, (10000,len(group)), replace=True).mean(axis=1)/3
    summary = {"selection": selection, "comparison": comparison,
        "paired_top3_difference": float(paired.mean()),
        "paired_top3_95ci": np.quantile(samples,[.025,.975]).tolist(),
        "test_metadata": {k:v for k,v in test.items() if k != "details"},
        "target_minimum": .85, "target_stretch": .998,
        "target_met_free_text_top3": new["top3"] >= .85,
        "target_met_synthetic_four_choice": new["synthetic_choice_accuracy"] >= .85,
        "evaluation_scope": "Equal-source local holdout; external pretraining overlap unknown. Synthetic answer options."}
    (ROOT / "models/neural_comparison_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
