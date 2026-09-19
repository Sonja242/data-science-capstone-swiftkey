"""Meaningful CPU regression tests for study integrity. Sonja Sahebzad."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learning_curve_audit import (
    SOURCES, SEEDS, audit_artifact, audit_details, audit_partitions,
    audit_prediction, confirmatory_comparison, keys, paired_bootstrap,
    select_on_development, select_development_summary, exact_paired_p,
)


def detail_fixture(ranks=(0, 1, 2, 3, 0, 1)):
    return [{"source": SOURCES[i // 2],
             "line_hash": hashlib.sha256(f"line-{i}".encode()).hexdigest(),
             "rank": rank, "choice_correct": i % 2 == 0,
             "shortlist_contains_target": bool(rank) or i == 0}
            for i, rank in enumerate(ranks)]


def result(seed=SEEDS[0], steps=256, width=64, ranks=(0, 1, 2, 3, 0, 1), latency=100):
    details = detail_fixture(ranks)
    return {"seed": seed, "steps": steps, "shortlist": width, "split": "development",
            "details": details, "summary": {**audit_details(details), "milliseconds": latency}}


class IntegrityChecks(unittest.TestCase):
    def test_metrics_recomputed_and_bounded(self):
        m = audit_details(detail_fixture())
        self.assertAlmostEqual(m["top1"], 2 / 6)
        self.assertAlmostEqual(m["top3"], 4 / 6)
        self.assertAlmostEqual(m["shortlist_recall"], 5 / 6)
        self.assertAlmostEqual(m["mrr_at3"], (1 + .5 + 1 / 3 + 1) / 6)

    def test_alignment_cannot_silently_zip_different_orders(self):
        rows = detail_fixture()
        with self.assertRaisesRegex(ValueError, "Ordered"):
            audit_details(list(reversed(rows)), keys(rows))

    def test_duplicate_hash_rejected_even_across_sources(self):
        rows = detail_fixture()
        rows[-1]["line_hash"] = rows[0]["line_hash"]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            audit_details(rows)

    def test_invalid_rank_and_false_coverage_rejected(self):
        for value in (-1, 4, True, 1.0):
            rows = detail_fixture()
            rows[0]["rank"] = value
            with self.assertRaises(ValueError):
                audit_details(rows)
        rows = detail_fixture()
        rows[1]["shortlist_contains_target"] = False
        with self.assertRaisesRegex(ValueError, "absent"):
            audit_details(rows)

    def test_forged_summary_rejected(self):
        artifact = result()
        artifact["summary"]["top3"] = .99
        with self.assertRaisesRegex(ValueError, "Stored metric"):
            audit_artifact(artifact)

    def test_hash_only_disjointness_checks(self):
        dev = detail_fixture()
        final = deepcopy(dev)
        for row in final:
            row["line_hash"] = hashlib.sha256(("fresh-" + row["line_hash"]).encode()).hexdigest()
        self.assertTrue(audit_partitions(dev, final, set(), set(), 2, 2)["passed"])
        with self.assertRaisesRegex(ValueError, "previous"):
            audit_partitions(dev, final, {final[0]["line_hash"]}, set(), 2, 2)
        with self.assertRaisesRegex(ValueError, "training"):
            audit_partitions(dev, final, set(), {final[0]["line_hash"]}, 2, 2)

    def test_options_do_not_enter_free_predictions(self):
        case = {**detail_fixture()[0], "actual": "world", "options": ["world", "best", "most", "universe"]}
        free = {"words": ["the", "a", "world"], "shortlist": ["the", "a", "world", "this"]}
        out = {**free, "choices": ["world", "best", "universe", "most"]}
        self.assertEqual(audit_prediction(case, out, free)["rank"], 3)
        corrupted = deepcopy(out)
        corrupted["words"] = ["world", "the", "a"]
        with self.assertRaisesRegex(ValueError, "altered"):
            audit_prediction(case, corrupted, free)
        case["options"][-1] = "world"
        with self.assertRaisesRegex(ValueError, "distinct"):
            audit_prediction(case, out)

    def test_unseen_target_can_be_missing_without_invalidating_case(self):
        case = {**detail_fixture()[0], "actual": "world", "options": ["world", "best", "most", "universe"]}
        out = {"words": ["the", "a", "this"], "shortlist": ["the", "a", "this"], "choices": case["options"]}
        row = audit_prediction(case, out)
        self.assertEqual(row["rank"], 0)
        self.assertTrue(row["choice_correct"])
        self.assertFalse(row["shortlist_contains_target"])

    def test_selection_uses_group_mean_not_best_seed(self):
        # Width64 has top3 values 2/6,3/6,4/6; width256 has 0,0,6/6.
        rows = []
        for i, seed in enumerate(SEEDS):
            rows.append(result(seed, width=64, ranks=tuple([1] * (2 + i) + [0] * (4 - i))))
            rows.append(result(seed, width=256, ranks=(1,) * 6 if i == 2 else (0,) * 6))
        chosen = select_on_development(rows, checkpoints=(256,), shortlists=(64, 256))
        self.assertEqual(chosen["selected"]["shortlist"], 64)
        self.assertEqual(chosen["representative_seed"], SEEDS[1])

    def test_selection_refuses_test_results_missing_or_duplicate_seeds(self):
        rows = [result(seed) for seed in SEEDS]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_on_development(rows[:-1], checkpoints=(256,))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_on_development(rows + [rows[0]], checkpoints=(256,))
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "development"):
            select_on_development(rows, checkpoints=(256,))

    def test_tie_breaks_are_deterministic(self):
        rows = [result(seed, step, width) for step in (256, 512) for width in (64, 256) for seed in SEEDS]
        chosen = select_on_development(rows, checkpoints=(256, 512), shortlists=(64, 256))
        self.assertEqual(chosen["selected"]["steps"], 256)
        self.assertEqual(chosen["selected"]["shortlist"], 64)
        self.assertEqual(chosen["representative_seed"], SEEDS[0])

    def test_summary_and_full_artifact_selection_agree(self):
        rows = [result(seed, step, width) for step in (256, 512) for width in (64, 256) for seed in SEEDS]
        flat = [{k: row[k] for k in ("seed", "steps", "shortlist", "split")} | row["summary"] for row in rows]
        self.assertEqual(select_on_development(rows, checkpoints=(256, 512), shortlists=(64, 256)),
                         select_development_summary(flat, checkpoints=(256, 512), shortlists=(64, 256)))
        flat[0]["cases"] += 1
        with self.assertRaises(ValueError):
            select_development_summary(flat, checkpoints=(256, 512), shortlists=(64, 256))

    def test_exact_paired_test_rejects_unpaired_and_nonbinary_input(self):
        with self.assertRaises(ValueError):
            exact_paired_p([1], [1, 0])
        with self.assertRaises(ValueError):
            exact_paired_p([.5], [1])
    def test_bootstrap_equal_source_estimand_and_reproducibility(self):
        sources = ["blogs"] * 2 + ["news"] * 3 + ["twitter"] * 5
        differences = [1] * 2 + [0] * 3 + [-1] * 5
        first = paired_bootstrap(differences, sources, repetitions=1000)
        second = paired_bootstrap(differences, sources, repetitions=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["difference"], 0)
        self.assertEqual(first["interval"], [0, 0])

    def test_identical_models_cannot_be_promoted(self):
        baseline = result()
        outcome = confirmatory_comparison(baseline, deepcopy(baseline))
        self.assertFalse(outcome["primary_top3"]["promote"])
        self.assertEqual(outcome["primary_top3"]["exact_paired_p"], 1)

    def test_clear_paired_improvement_is_detected(self):
        baseline, challenger = result(ranks=(0,) * 6), result(ranks=(1,) * 6)
        outcome = confirmatory_comparison(baseline, challenger)
        self.assertTrue(outcome["primary_top3"]["promote"])
        self.assertAlmostEqual(outcome["primary_top3"]["exact_paired_p"], .03125)

    def test_seed_mean_is_averaged_per_line_not_treated_as_extra_cases(self):
        baseline = result(ranks=(0,) * 6)
        replicas = [result(SEEDS[i], ranks=(1,) * 6 if i == 0 else (0,) * 6) for i in range(3)]
        outcome = confirmatory_comparison(baseline, replicas[1], replicas)
        self.assertFalse(outcome["primary_top3"]["promote"])
        self.assertAlmostEqual(outcome["secondary_seed_mean_top3"]["difference"], 1 / 3)
        self.assertEqual(outcome["secondary_seed_mean_top3"]["interval"], [1 / 3, 1 / 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)