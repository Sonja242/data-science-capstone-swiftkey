"""Explicitly try the development-selected experimental model. Sonja Sahebzad."""
import argparse
import json
from pathlib import Path
from learning_curve_experiment import ROOT, CurveRanker, frozen, read_json, verify_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen()
    # Require the completed study; never select on final scores or change default.
    summary = read_json(ROOT / "models/learning_curve_summary.json")
    selection = verify_selection()
    spec = {"kind": "curve", "seed": selection["representative_seed"],
            "steps": selection["selected"]["steps"],
            "shortlist": selection["selected"]["shortlist"]}
    request = read_json(args.request)
    ranker = CurveRanker(spec["seed"], spec["steps"])
    result = ranker.predict_widths(request["phrase"], request.get("ngram_candidates", []),
                                  request.get("choices", []), (spec["shortlist"],))[spec["shortlist"]]
    result.pop("full_ranking", None)
    result["selected_spec"] = spec
    result["experimental"] = not summary["promote"]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
