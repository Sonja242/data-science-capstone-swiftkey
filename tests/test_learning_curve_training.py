"""CPU-only safety and reproducibility checks. Author: Sonja Sahebzad."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "python/learning_curve_training.py"
spec = importlib.util.spec_from_file_location("learning_curve_training", SCRIPT)
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


class ScheduleTests(unittest.TestCase):
    def test_unique_blocks_and_seed_reproduction(self):
        a = curve.make_schedule(11282, 20261410)
        b = curve.make_schedule(11282, 20261410)
        c = curve.make_schedule(11282, 20261411)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1024, 8))
        self.assertEqual(len(np.unique(a)), 8192)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(curve.schedule_sha256(a), curve.schedule_sha256(a.astype(">i8")))
        self.assertNotEqual(curve.schedule_sha256(a), curve.schedule_sha256(c))

    def test_short_stages_are_prefixes_of_one_trajectory(self):
        schedule = curve.make_schedule(11282, 20261410)
        seen = set()
        for stop in curve.CHECKPOINTS:
            now = set(schedule[:stop].ravel().tolist())
            self.assertTrue(seen.issubset(now))
            self.assertEqual(len(now), stop * 8)
            seen = now
        with self.assertRaises(ValueError):
            curve.make_schedule(8191, 20261410)

    def test_lr_warmup_and_shared_horizon(self):
        rates = [curve.learning_rate(s) for s in range(1, 1025)]
        self.assertAlmostEqual(rates[0], 1e-4 / 16)
        self.assertAlmostEqual(rates[15], 1e-4)
        self.assertEqual(rates[-1], 0.)
        self.assertTrue(all(a < b for a, b in zip(rates[:15], rates[1:16])))
        self.assertTrue(all(a >= b for a, b in zip(rates[15:], rates[16:])))
        self.assertGreater(rates[255], rates[511])
        self.assertGreater(rates[511], rates[1023])
        for invalid in (0, 1025):
            with self.assertRaises(ValueError):
                curve.learning_rate(invalid)

    def test_token_accounting_and_accumulation(self):
        self.assertEqual(curve.token_counts(256), {"consumed_blocks": 2048,
            "input_tokens": 262144, "supervised_token_positions": 260096})
        self.assertEqual(curve.token_counts(1024)["input_tokens"], 1048576)
        for micro in (2, 4, 8):
            plan = curve.validate_plan(20261410, 1024, micro, curve.CHECKPOINTS, None)
            hp = curve.hyperparameters(plan)
            self.assertEqual(hp["microbatch"] * hp["gradient_accumulation"], 8)


class PlanTests(unittest.TestCase):
    def test_official_and_pilot_outputs_are_isolated(self):
        official = curve.validate_plan(20261410, 1024, 2, curve.CHECKPOINTS, None)
        pilot = curve.validate_plan(20261410, 8, 8, (8,), "mb8")
        root = Path("project")
        self.assertNotEqual(curve.output_paths(root, official), curve.output_paths(root, pilot))
        self.assertEqual(pilot["purpose"], "pilot_training_only")
        self.assertEqual(curve.hyperparameters(pilot)["cosine_horizon_steps"], 1024)

    def test_invalid_or_ambiguous_plans_refused(self):
        invalid = [(20261410, 8, 2, (8,), None),
                   (1, 1024, 2, curve.CHECKPOINTS, None),
                   (20261410, 1024, 3, curve.CHECKPOINTS, None),
                   (20261410, 1024, 2, (512, 256, 1024), None),
                   (20261410, 1024, 2, (256, 256, 1024), None),
                   (20261410, 8, 2, (4,), "short"),
                   (20261410, 8, 2, (8,), "../../escape")]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                curve.validate_plan(*args)

    def test_metadata_counts_and_rejects_incomplete_history(self):
        plan = curve.validate_plan(20261410, 8, 2, (8,), "metadata")
        schedule = curve.make_schedule(11282, 20261410)
        history = [{"step": i} for i in range(1, 9)]
        metadata = curve.checkpoint_metadata(plan, 8, {"data": "hash"}, schedule,
                                           history, {"elapsed": 1.0}, {"adapter": "hash"})
        self.assertEqual(metadata["input_tokens"], 8192)
        self.assertEqual(metadata["supervised_token_positions"], 8128)
        self.assertEqual(metadata["consumed_schedule_sha256"], curve.schedule_sha256(schedule[:8]))
        self.assertFalse(metadata["contains_test_metrics"])
        with self.assertRaises(ValueError):
            curve.checkpoint_metadata(plan, 8, {}, schedule, history[:-1], {}, {})

    def test_exclusive_json_writes_protect_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            curve.json_write_new(path, {"status": "original"})
            with self.assertRaises(FileExistsError):
                curve.json_write_new(path, {"status": "overwrite"})
            self.assertEqual(json.loads(path.read_text())["status"], "original")


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / "data/adaptation"
        base = self.root / "models/neural/Qwen3-1.7B-Base"
        data.mkdir(parents=True)
        base.mkdir(parents=True)
        (data / "local_training.txt").write_text("Neutral training fixture.\n")
        (data / "development.csv").write_text("hash,target\nx,word\n")
        # A final-test file deliberately exists but must not be read or hashed.
        (data / "test.csv").write_text("UNUSED FINAL TEST")
        np.save(data / "local_training.npy", np.zeros((11282, 128), dtype=np.int32))
        self.patchers = [patch.object(curve, "TEXT_SHA256", curve.sha256(data / "local_training.txt")),
                         patch.object(curve, "CACHE_SHA256", curve.sha256(data / "local_training.npy")),
                         patch.object(curve, "DEVELOPMENT_SHA256", curve.sha256(data / "development.csv"))]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        (data / "local_training.meta.json").write_text(json.dumps({"text_sha256": curve.TEXT_SHA256,
            "seed": 20261311, "block_tokens": 128, "base_revision": curve.BASE_REVISION}))
        (base / "download_manifest.json").write_text(json.dumps({"repo": "Qwen/Qwen3-1.7B-Base", "revision": curve.BASE_REVISION}))
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
            (base / name).write_bytes(b"CPU test fixture, not a model")
        self.plan = curve.validate_plan(20261410, 1024, 2, curve.CHECKPOINTS, None)

    def test_valid_fixture_records_hashes_without_test_access(self):
        original = curve.sha256
        accessed = []
        def tracked(path):
            accessed.append(Path(path).name)
            return original(path)
        with patch.object(curve, "sha256", tracked):
            blocks, schedule, metadata = curve.preflight(self.root, self.plan)
        self.assertEqual(blocks.shape, (11282, 128))
        self.assertEqual(schedule.shape, (1024, 8))
        self.assertEqual(metadata["base_revision"], curve.BASE_REVISION)
        self.assertNotIn("test.csv", accessed)
        self.assertFalse(curve.output_paths(self.root, self.plan)[0].exists())

    def test_changed_training_data_or_base_revision_refused(self):
        datafile = self.root / "data/adaptation/local_training.txt"
        before = datafile.read_bytes()
        datafile.write_bytes(before + b"changed")
        with self.assertRaises(ValueError):
            curve.preflight(self.root, self.plan)
        datafile.write_bytes(before)
        manifest = self.root / "models/neural/Qwen3-1.7B-Base/download_manifest.json"
        manifest.write_text(json.dumps({"repo": "Qwen/Qwen3-1.7B-Base", "revision": "wrong"}))
        with self.assertRaises(ValueError):
            curve.preflight(self.root, self.plan)

    def test_existing_partial_or_completed_run_refused(self):
        destination, summary = curve.output_paths(self.root, self.plan)
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            curve.preflight(self.root, self.plan)
        destination.rmdir()
        summary.write_text("existing audit result")
        with self.assertRaises(FileExistsError):
            curve.preflight(self.root, self.plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
