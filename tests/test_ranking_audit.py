"""CPU tests of development-only audit safeguards and diagnostic definitions."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "python/audit_ranking_development.py"
spec = importlib.util.spec_from_file_location("ranking_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RankingAuditTests(unittest.TestCase):
    def setUp(self):
        self.row = {"source": "blogs", "line_hash": "abc", "prefix": "a simple", "actual": "word"}
        self.detail = {"source": "blogs", "line_hash": "abc", "context_words": 2,
                       "target_characters": 4, "rank": 0, "shortlist_contains_target": True,
                       "choice_correct": True}
        self.artifact = {"split": "development", "data_sha256": "dataset", "details": [self.detail],
                         "summary": {"cases": 1, "top3": 0., "shortlist_recall": 1.}}

    def test_rank_zero_with_coverage_is_ranking_miss(self):
        self.assertEqual(audit.category(self.detail), "present_outside_top3")
        missing = {**self.detail, "shortlist_contains_target": False}
        self.assertEqual(audit.category(missing), "target_absent")

    def test_final_test_is_refused(self):
        artifact = {**self.artifact, "split": "test"}
        with self.assertRaisesRegex(AssertionError, "Only development"):
            audit.validate_artifact(artifact, [self.row], "dataset")

    def test_wrong_dataset_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "fingerprint"):
            audit.validate_artifact(self.artifact, [self.row], "wrong")

    def test_misaligned_rows_are_refused(self):
        row = {**self.row, "line_hash": "different"}
        with self.assertRaises(AssertionError):
            audit.validate_artifact(self.artifact, [row], "dataset")

    def test_top3_target_cannot_be_absent(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["details"][0].update(rank=1, shortlist_contains_target=False)
        with self.assertRaises(AssertionError):
            audit.validate_artifact(artifact, [self.row], "dataset")

    def test_coverage_conditional_denominator(self):
        details = [self.detail, {**self.detail, "rank": 1},
                   {**self.detail, "shortlist_contains_target": False}]
        result = audit.metrics(details)
        self.assertEqual(result["top3"], 1/3)
        self.assertEqual(result["coverage"], 2/3)
        self.assertEqual(result["top3_given_covered"], 1/2)
        self.assertEqual(result["present_outside_top3_count"], 1)
        self.assertEqual(result["target_absent_count"], 1)

    def test_nested_coverage_is_not_top3_gain(self):
        left = [{**self.detail, "shortlist_contains_target": False}]
        result = audit.paired_transition(left, [self.detail])
        self.assertEqual(result["newly_covered"], 1)
        self.assertEqual(result["new_top3"], 0)
        self.assertEqual(result["same_recorded_rank"], 1)

    def test_empty_subgroup_has_no_invented_rate(self):
        result = audit.metrics([])
        self.assertIsNone(result["top3"])
        self.assertIsNone(result["top3_given_covered"])


if __name__ == "__main__":
    unittest.main()
