"""Independent CPU regressions for word-generation integrity. Sonja Sahebzad."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from word_generation_audit import (
    audit_reserved_hashes, score_map, ranking, derive_artifacts,
    select_generation, paired_counts, primary_comparison,
)


def score_pairs(values):
    return [[word, value] for word, value in values.items()]


def fixture(split="development", candidates=("beam16", "beam64")):
    records = []
    bases = [
        {"alpha": -1., "beta": -2., "gamma": -3., "delta": -4.},
        {"alpha": -1., "beta": -2., "target": -3., "delta": -4.},
        {"target": -.2, "alpha": -1., "beta": -2., "gamma": -3.},
        {"alpha": -1., "beta": -2., "gamma": -3., "target": -4.},
        {"alpha": -1., "beta": -2., "gamma": -3., "delta": -4.},
        {"alpha": -1., "target": -2., "gamma": -3., "delta": -4.},
    ]
    for i, base in enumerate(bases):
        additions = ({"target": -.5} if i == 0 else {"novel": -.5} if i == 1 else {})
        variants = {}
        for name in candidates:
            extra = {**additions, **({"target": -.5} if i == 4 and name == "beam64" else {})}
            variants[name] = {"scores": score_pairs(extra), "words": ranking({**base, **extra})[:3]}
        options = {"target": base.get("target", -.5), "alpha": -1., "beta": -2., "gamma": -3.}
        records.append({"source": ("blogs", "news", "twitter")[i // 2],
            "line_hash": hashlib.sha256(f"generation-{i}".encode()).hexdigest(),
            "actual": "target", "target_tokens": 1 if i % 2 else 2, "target_in_vocabulary": True,
            "baseline": {"scores": score_pairs(base), "words": ranking(base)[:3]},
            "options": {"scores": score_pairs(options), "words": ranking(options)},
            "candidates": variants})
    return {"split": split, "details": records}


class GenerationAuditTests(unittest.TestCase):
    def test_metrics_derived_from_canonical_union_and_options_kept_separate(self):
        artifacts = derive_artifacts(fixture())
        self.assertEqual(artifacts["baseline"]["summary"]["top3"], 3 / 6)
        self.assertEqual(artifacts["beam16"]["summary"]["top3"], 3 / 6)
        self.assertEqual(artifacts["beam64"]["summary"]["top3"], 4 / 6)
        for artifact in artifacts.values():
            self.assertEqual(artifact["summary"]["synthetic_choice_accuracy"], artifacts["baseline"]["summary"]["synthetic_choice_accuracy"])
        # Offered target remains absent from the smaller free list at case4.
        self.assertFalse(artifacts["beam16"]["details"][4]["shortlist_contains_target"])
        self.assertTrue(artifacts["beam16"]["details"][4]["choice_correct"])

    def test_added_words_cannot_overwrite_original_scores(self):
        raw = fixture()
        raw["details"][0]["candidates"]["beam16"]["scores"].append(["alpha", -.1])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            derive_artifacts(raw)

    def test_reported_top3_must_match_canonical_union_ranking(self):
        raw = fixture()
        raw["details"][3]["candidates"]["beam16"]["words"] = ["target", "alpha", "beta"]
        with self.assertRaisesRegex(ValueError, "canonical union"):
            derive_artifacts(raw)

    def test_reported_option_ranking_must_match_its_scores(self):
        raw = fixture()
        raw["details"][0]["options"]["words"].reverse()
        with self.assertRaisesRegex(ValueError, "Option ranking"):
            derive_artifacts(raw)

    def test_malformed_or_unsafe_score_pairs_rejected(self):
        for pairs in ([["x", float("nan")]], [["x", float("inf")]], [["x", True]],
                      [["x", .1]], [["Two words", -1]], [["x", -1], ["x", -2]]):
            with self.assertRaises(ValueError):
                score_map(pairs, "fixture")
        self.assertEqual(score_map([], "optional", allow_empty=True), {})

    def test_artifact_configuration_and_split_cannot_silently_change(self):
        raw = fixture()
        del raw["details"][0]["candidates"]["beam64"]
        with self.assertRaisesRegex(ValueError, "configuration grid"):
            derive_artifacts(raw)
        with self.assertRaisesRegex(ValueError, "Unexpected split"):
            select_generation(fixture("final"), {"beam16": 1., "beam64": 2.}, expected_cases=6)

    def test_shared_word_scores_use_preregistered_numerical_tolerance(self):
        raw = fixture()
        raw["details"][0]["candidates"]["beam64"]["scores"][0][1] += .015
        derive_artifacts(raw)
        with self.assertRaisesRegex(ValueError, "neutral tolerance"):
            derive_artifacts(raw, score_tolerance=.001)
        raw["details"][0]["candidates"]["beam64"]["scores"][0][1] += .01
        with self.assertRaisesRegex(ValueError, "neutral tolerance"):
            derive_artifacts(raw)

    def test_option_scores_match_free_canonical_scores_within_tolerance(self):
        raw = fixture()
        raw["details"][0]["options"]["scores"][0][1] += .03
        with self.assertRaisesRegex(ValueError, "Option/free canonical"):
            derive_artifacts(raw)

    def test_duplicate_or_reordered_identity_rejected(self):
        raw = fixture()
        expected = [(r["source"], r["line_hash"]) for r in raw["details"]]
        raw["details"].reverse()
        with self.assertRaisesRegex(ValueError, "Ordered"):
            derive_artifacts(raw, expected_keys=expected)
        raw["details"][1]["line_hash"] = raw["details"][0]["line_hash"]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            derive_artifacts(raw)

    def test_terminal_cap_is_enforced(self):
        raw = fixture()
        with self.assertRaisesRegex(ValueError, "terminal cap"):
            derive_artifacts(raw, terminal_cap=0)

    def test_canonical_token_group_requires_positive_integer(self):
        for bad in (0, -1, True, 1.5):
            raw = fixture()
            raw["details"][0]["target_tokens"] = bad
            with self.assertRaisesRegex(ValueError, "target_tokens"):
                derive_artifacts(raw)

    def test_selection_uses_exact_accuracy_before_speed(self):
        selected = select_generation(fixture(), {"beam16": 1., "beam64": 1000.}, expected_cases=6)
        self.assertEqual(selected["candidate"], "beam64")
        self.assertEqual(selected["selected"]["top3_correct"], 4)

    def test_selection_speed_and_fixed_order_break_exact_ties(self):
        raw = fixture()
        for row in raw["details"]:
            row["candidates"]["beam64"] = deepcopy(row["candidates"]["beam16"])
        self.assertEqual(select_generation(raw, {"beam16": 2., "beam64": 1.}, expected_cases=6)["candidate"], "beam64")
        self.assertEqual(select_generation(raw, {"beam16": 1., "beam64": 1.}, expected_cases=6)["candidate"], "beam16")

    def test_selection_requires_both_valid_independent_timings(self):
        for times in ({"beam16": 1.}, {"beam16": 1., "beam64": 0}, {"beam16": 1., "beam64": float("nan")}):
            with self.assertRaises(ValueError):
                select_generation(fixture(), times, expected_cases=6)

    def test_paired_transition_table_reconciles_and_groups_are_descriptive(self):
        result = primary_comparison(fixture("final", ("beam64",)), "beam64", expected_cases=6)
        primary = result["primary_top3"]
        self.assertEqual((primary["gains"], primary["losses"]), (2, 1))
        self.assertEqual(primary["gains_from_previously_missing_target"], 2)
        self.assertEqual(primary["unexpected_covered_target_gains"], 0)
        self.assertEqual(primary["lost_coverage"], 0)
        self.assertEqual(primary["expanded_top3_correct"] - primary["baseline_top3_correct"], 1)
        self.assertEqual(primary["bootstrap_seed"], 20261590)
        self.assertFalse(primary["promote"])
        groups = {row["group"]: row for row in result["descriptive_groups"]}
        self.assertEqual(groups["single_token"]["cases"], 3)
        self.assertEqual(groups["multi_token"]["cases"], 3)
        self.assertEqual(groups["multi_token"]["gains"], 2)

    def test_covered_target_gain_or_lost_coverage_cannot_be_hidden(self):
        artifacts = derive_artifacts(fixture())
        base = artifacts["baseline"]["details"]
        changed = deepcopy(artifacts["beam16"]["details"])
        changed[3]["rank"] = 1
        with self.assertRaisesRegex(ValueError, "superset"):
            paired_counts(base, changed)
        changed = deepcopy(base)
        changed[3]["shortlist_contains_target"] = False
        with self.assertRaisesRegex(ValueError, "superset"):
            paired_counts(base, changed)

    def test_close_boundary_diagnostic_does_not_change_primary_rule(self):
        raw = fixture("final", ("beam16",))
        row = raw["details"][0]
        row["candidates"]["beam16"]["scores"] = [["target", -2.99]]
        row["candidates"]["beam16"]["words"] = ["alpha", "beta", "target"]
        row["options"]["scores"][0][1] = -2.99
        row["options"]["words"] = ranking(dict(row["options"]["scores"]))
        result = primary_comparison(raw, "beam16", expected_cases=6)
        self.assertEqual(result["primary_top3"]["gains_at_close_expanded_boundary"], 1)
        self.assertEqual(result["primary_top3"]["gains"], 1)
        self.assertEqual(result["primary_top3"]["losses"], 1)
        self.assertFalse(result["primary_top3"]["promote"])

    def test_final_split_matches_the_frozen_evaluator_schema(self):
        raw = fixture("final", ("beam16",))
        self.assertEqual(derive_artifacts(raw, ("beam16",))["beam16"]["split"], "final")
        primary_comparison(raw, "beam16", expected_cases=6)
        raw["split"] = "test"
        with self.assertRaisesRegex(ValueError, "development or final"):
            derive_artifacts(raw, ("beam16",))

    def test_partial_artifacts_cannot_pass_protocol_case_counts(self):
        with self.assertRaisesRegex(ValueError, "Development case count"):
            select_generation(fixture(), {"beam16": 1., "beam64": 2.})
        with self.assertRaisesRegex(ValueError, "Final case count"):
            primary_comparison(fixture("final", ("beam16",)), "beam16")

    def test_fresh_hash_audit_requires_all_previous_sets_and_no_overlap(self):
        dev = fixture()["details"]
        reserved = deepcopy(dev)
        prior = {}
        for i, row in enumerate(reserved):
            row["line_hash"] = hashlib.sha256(f"reserved-{i}".encode()).hexdigest()
        for name in ("v3", "neural", "adaptation", "learning_curve"):
            prior[name] = [{"source": "blogs", "line_hash": hashlib.sha256(name.encode()).hexdigest()}]
        counts = {name: 1 for name in prior}
        result = audit_reserved_hashes(dev, reserved, prior, set(), expected_prior_counts=counts,
                                      development_per_source=2, final_per_source=2)
        self.assertEqual(result["excluded_prior_final_unique_lines"], 4)
        incomplete = deepcopy(prior)
        del incomplete["learning_curve"]
        with self.assertRaisesRegex(ValueError, "previously evaluated"):
            audit_reserved_hashes(dev, reserved, incomplete, set(), expected_prior_counts=counts,
                                  development_per_source=2, final_per_source=2)
        with self.assertRaisesRegex(ValueError, "training"):
            audit_reserved_hashes(dev, reserved, prior, {reserved[0]["line_hash"]}, expected_prior_counts=counts,
                                  development_per_source=2, final_per_source=2)
        prior["learning_curve"][0]["line_hash"] = reserved[0]["line_hash"]
        with self.assertRaisesRegex(ValueError, "previous evaluations"):
            audit_reserved_hashes(dev, reserved, prior, set(), expected_prior_counts=counts,
                                  development_per_source=2, final_per_source=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)