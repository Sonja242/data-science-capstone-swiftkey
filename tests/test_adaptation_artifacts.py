"""Cross-artifact leakage and provenance checks; no GPU or model fitting."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = ROOT / "data/adaptation"
    development = read_csv(data / "development.csv")
    final = read_csv(data / "test.csv")
    dev_hashes = {r["line_hash"] for r in development}
    test_hashes = {r["line_hash"] for r in final}
    assert len(dev_hashes) == len(development) == 600
    assert len(test_hashes) == len(final) == 900
    assert not dev_hashes & test_hashes
    previous_hashes = set()
    for split in ("development", "test"):
        previous_hashes.update(r["line_hash"] for r in read_csv(
            ROOT / "data/neural_evaluation" / f"{split}.csv"))
    assert not (dev_hashes | test_hashes) & previous_hashes
    train_counts = {}
    for filename in ("local_training.txt", "taskmaster_training.txt"):
        lines = (data / filename).read_text(encoding="utf-8").splitlines()
        hashes = {hashlib.sha256(line.encode()).hexdigest() for line in lines}
        assert not hashes & (dev_hashes | test_hashes)
        train_counts[filename] = len(lines)
    for recipe in ("local", "augmented"):
        manifest = read_json(ROOT / "models" / f"adaptation_training_{recipe}.json")
        assert manifest["training_code_sha256"] == sha(ROOT / "python/train_adaptation.py")
        assert manifest["local_data_sha256"] == sha(data / "local_training.txt")
        assert manifest["external_data_sha256"] == sha(data / "taskmaster_training.txt")
        assert manifest["adapter_sha256"] == sha(
            ROOT / "models/neural" / f"adaptation_{recipe}" / "adapter_model.safetensors")
        assert manifest["steps"] == len(manifest["history"]) == 256
        assert manifest["input_tokens"] == 262144
    protocol = read_json(ROOT / "models/adaptation_protocol.json")
    code = (ROOT / "python/adaptation_experiment.py").read_text(encoding="utf-8-sig")
    assert protocol["comparison_code_sha256"] == hashlib.sha256(code.encode()).hexdigest()
    assert protocol["test_sha256"] == sha(data / "test.csv")
    equivalent = {}
    archive = ROOT / "backups/adaptation-scoring-20260919-232233"
    for candidate in ("base64", "base256"):
        name = f"adaptation_{candidate}_development.json"
        if (archive / name).exists() and (ROOT / "models" / name).exists():
            before = read_json(archive / name)
            after = read_json(ROOT / "models" / name)
            assert before["details"] == after["details"]
            equivalent[candidate] = len(after["details"])
    result = {
        "passed": True,
        "development_cases": len(development),
        "test_cases": len(final),
        "training_lines": train_counts,
        "new_holdout_overlaps_previous_neural_cases": 0,
        "training_exact_line_overlaps_new_holdout": 0,
        "training_data_code_adapter_hashes_verified": True,
        "evaluation_matches_frozen_protocol": True,
        "development_outcomes_identical_before_after_vectorization": equivalent,
        "limitations": "Exact normalized-line checks do not exclude near duplicates or external pretraining overlap."
    }
    (ROOT / "models/adaptation_integrity_checks.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
