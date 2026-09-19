"""CPU-only evaluator construction checks; never load weights or final data.

Set CAPSTONE_EVALUATOR_ROOT to test an evaluator in a separate worktree.
Author: Sonja Sahebzad.
"""
from contextlib import nullcontext, ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

sys.dont_write_bytecode = True
PROJECT = Path(os.environ.get("CAPSTONE_EVALUATOR_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "python"))
spec = importlib.util.spec_from_file_location("curve_evaluator_cpu", PROJECT / "python/learning_curve_experiment.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class CpuTensor:
    """Minimal read-only tensor interface used only by candidate construction."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, index):
        return CpuTensor(self.values[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def token_word(index):
    return "word" + chr(97 + index // 26) + chr(97 + index % 26)


def cpu_ranker():
    ranker = evaluator.CurveRanker.__new__(evaluator.CurveRanker)
    ranker.torch = SimpleNamespace(inference_mode=nullcontext)
    # Reversed eligible storage proves token IDs, not storage order, break ties.
    ranker.eligible = np.arange(320, dtype=np.int64)[::-1]
    ranker.eligible_cpu = ranker.eligible.copy()
    ranker.decoded = [" " + token_word(i) for i in range(320)]
    ranker.decoded[1] = ranker.decoded[0].upper()
    values = np.arange(320, 0, -1, dtype=float)
    values[64] = values[63]
    values[256] = values[255]
    prefill = SimpleNamespace(logits=CpuTensor(values[None, None, :]))
    ranker._prefill = lambda phrase: ([11, 12], prefill)
    ranker.calls = []

    def scores(prefix, cache, words):
        ranker.calls.append(list(words))
        return {word: len(word) + sum(map(ord, word)) / 10000 for word in words}

    ranker._score_candidates = scores
    return ranker, prefill


class CandidateConstructionTests(unittest.TestCase):
    def test_tied_logits_use_token_id_at_both_cutoffs(self):
        ranker, prefill = cpu_ranker()
        sets, metadata = ranker.free_sets(prefill, [], (64, 256))
        self.assertIn(token_word(63), sets[64])
        self.assertNotIn(token_word(64), sets[64])
        self.assertIn(token_word(255), sets[256])
        self.assertNotIn(token_word(256), sets[256])
        self.assertEqual(metadata[64]["raw_token_cutoff_ties"], 2)
        self.assertEqual(metadata[256]["raw_token_cutoff_ties"], 2)

    def test_raw_token_prefix_is_taken_before_alias_deduplication(self):
        ranker, prefill = cpu_ranker()
        sets, metadata = ranker.free_sets(prefill, ["can't", "the"], (64, 256))
        self.assertEqual(metadata[64]["unique_neural_words"], 63)
        self.assertEqual(metadata[256]["unique_neural_words"], 255)
        self.assertEqual(len(sets[64]), 65)
        self.assertEqual(len(sets[256]), 257)
        self.assertEqual(sets[64].count(token_word(0)), 1)
        self.assertNotIn(token_word(64), sets[64])
        self.assertTrue(set(sets[64]) <= set(sets[256]))

    def test_choices_never_change_free_candidates_scores_or_rankings(self):
        ranker, _ = cpu_ranker()
        plain = ranker.predict_widths("a neutral prefix", ["can't", "the"])
        choices = ["zebra", "uncharacteristically", token_word(2), "can't"]
        with_choices = ranker.predict_widths("a neutral prefix", ["can't", "the"], choices)
        for width in (64, 256):
            for field in ("words", "shortlist", "full_ranking", "log_scores"):
                self.assertEqual(plain[width][field], with_choices[width][field])
            self.assertNotIn("zebra", with_choices[width]["shortlist"])
            self.assertNotIn("uncharacteristically", with_choices[width]["shortlist"])
            self.assertEqual(set(with_choices[width]["choices"]), set(choices))
        self.assertEqual(with_choices[64]["choices"], with_choices[256]["choices"])

    def test_largest_free_batch_is_scored_first_and_missing_options_separately(self):
        ranker, _ = cpu_ranker()
        choices = [token_word(2), "zebra", "can't", "uncharacteristically"]
        result = ranker.predict_widths("a neutral prefix", ["can't", "the"], choices)
        self.assertEqual(len(ranker.calls), 2)
        self.assertEqual(ranker.calls[0], result[256]["shortlist"])
        self.assertEqual(ranker.calls[1], ["zebra", "uncharacteristically"])
        self.assertNotIn("zebra", ranker.calls[0])
        self.assertNotIn("uncharacteristically", ranker.calls[0])

    def test_direct_width_restricts_work_to_its_own_free_candidates(self):
        ranker, _ = cpu_ranker()
        choices = ["zebra", "can't", token_word(2), "uncharacteristically"]
        joint = ranker.predict_widths("a neutral prefix", ["can't", "the"], choices)
        ranker.calls.clear()
        direct = ranker.predict_widths("a neutral prefix", ["can't", "the"], choices, (64,))[64]
        self.assertEqual(ranker.calls[0], direct["shortlist"])
        self.assertEqual(len(ranker.calls[0]), 65)
        self.assertEqual(joint[64]["words"], direct["words"])
        self.assertEqual(joint[64]["choices"], direct["choices"])
        self.assertEqual(joint[64]["log_scores"], direct["log_scores"])

    def test_multiple_word_option_is_rejected(self):
        ranker, _ = cpu_ranker()
        with self.assertRaisesRegex(ValueError, "single normalized"):
            ranker.predict_widths("a neutral prefix", [], ["two words"])


def details_fixture():
    return [{"source": ("blogs", "news", "twitter")[i // 2],
             "line_hash": hashlib.sha256(f"cpu-fixture-{i}".encode()).hexdigest(),
             "rank": 1 if i % 2 else 0, "choice_correct": True,
             "shortlist_contains_target": True} for i in range(6)]


def artifact_fixture(seed=20261410, step=256, width=64, split="development"):
    return {"seed": seed, "steps": step, "shortlist": width, "split": split,
            "implementation": {"frozen": "cpu-fixture"}, "adapter_sha256": "adapter-fixture",
            "details": details_fixture(), "summary": {"milliseconds": 100 + step + width}}


class ArtifactBoundaryTests(unittest.TestCase):
    def patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(evaluator, "frozen", return_value={"implementation": {"frozen": "cpu-fixture"}}))
        cases = stack.enter_context(mock.patch.object(evaluator, "cases", return_value=details_fixture()))
        checkpoint = stack.enter_context(mock.patch.object(evaluator, "checkpoint", return_value=(None, {"adapter_sha256": "adapter-fixture"}, None)))
        stack.enter_context(mock.patch.object(evaluator, "protocol", return_value={"operational_adapter_sha256": "adapter-fixture"}))
        return stack, cases, checkpoint

    def test_expected_identity_rejects_wrong_seed_step_or_width_before_cases(self):
        for field, value in (("seed", 20261411), ("steps", 512), ("shortlist", 256)):
            stack, cases, _ = self.patches()
            with stack:
                with self.assertRaisesRegex(AssertionError, "identity mismatch"):
                    evaluator.verify_result(artifact_fixture(), "development", {field: value})
                cases.assert_not_called()

    def test_matching_identity_and_adapter_is_accepted(self):
        stack, _, checkpoint = self.patches()
        with stack:
            evaluator.verify_result(artifact_fixture(), "development", {"seed": 20261410, "steps": 256, "shortlist": 64})
            checkpoint.assert_called_once_with(20261410, 256)

    def test_adapter_mismatch_and_reordered_cases_are_rejected(self):
        artifact = artifact_fixture()
        artifact["adapter_sha256"] = "wrong-adapter"
        stack, _, _ = self.patches()
        with stack, self.assertRaises(AssertionError):
            evaluator.verify_result(artifact, "development")
        artifact = artifact_fixture()
        artifact["details"].reverse()
        stack, _, _ = self.patches()
        with stack, self.assertRaisesRegex(ValueError, "Ordered case"):
            evaluator.verify_result(artifact, "development")

    def test_operational_baseline_requires_unchanged_adapter_and_dimensions(self):
        artifact = artifact_fixture(seed=20261310, step=256, width=256, split="test")
        artifact["name"] = "prior_operational"
        stack, _, checkpoint = self.patches()
        with stack:
            evaluator.verify_result(artifact, "test", {"name": "prior_operational", "seed": 20261310})
            checkpoint.assert_not_called()
        for field, value in (("adapter_sha256", "wrong-adapter"), ("shortlist", 64), ("name", "other-baseline")):
            changed = deepcopy(artifact)
            changed[field] = value
            stack, _, _ = self.patches()
            with stack, self.assertRaises(AssertionError):
                evaluator.verify_result(changed, "test")

    def selection_fixture(self):
        artifacts = {}
        for seed in evaluator.SEEDS:
            for step in evaluator.STEPS:
                for width in evaluator.WIDTHS:
                    artifacts[f"learning_curve_dev_{seed}_{step}_{width}.json"] = artifact_fixture(seed, step, width)
        choice = evaluator.select_on_development(list(artifacts.values()), checkpoints=evaluator.STEPS,
                     shortlists=evaluator.WIDTHS, seeds=evaluator.SEEDS)
        choice["implementation"] = {"frozen": "cpu-fixture"}
        choice["artifact_sha256"] = {f"{r['seed']}_{r['steps']}_{r['shortlist']}": "sha-" + name
                                       for name, r in artifacts.items()}
        return artifacts, choice

    def selection_patches(self, artifacts, choice, tampered_name=None):
        values = {**artifacts, "learning_curve_selection.json": choice}
        stack = ExitStack()
        stack.enter_context(mock.patch.object(evaluator, "read_json", side_effect=lambda path: deepcopy(values[Path(path).name])))
        stack.enter_context(mock.patch.object(evaluator, "implementation", return_value={"frozen": "cpu-fixture"}))
        stack.enter_context(mock.patch.object(evaluator, "sha", side_effect=lambda path:
                    "tampered" if Path(path).name == tampered_name else "sha-" + Path(path).name))
        verified = stack.enter_context(mock.patch.object(evaluator, "verify_result"))
        return stack, verified

    def test_complete_selection_hashes_and_recomputed_choice_are_checked(self):
        artifacts, choice = self.selection_fixture()
        stack, verified = self.selection_patches(artifacts, choice)
        with stack:
            self.assertEqual(evaluator.verify_selection(), choice)
            self.assertEqual(verified.call_count, 18)
            for call in verified.call_args_list:
                result, split, identity = call.args
                self.assertEqual(split, "development")
                self.assertEqual(identity, {key: result[key] for key in ("seed", "steps", "shortlist")})

    def test_modified_development_file_or_selected_setting_is_rejected(self):
        artifacts, choice = self.selection_fixture()
        stack, _ = self.selection_patches(artifacts, choice, next(iter(artifacts)))
        with stack, self.assertRaises(AssertionError):
            evaluator.verify_selection()
        choice["selected"]["steps"] = 1024
        stack, _ = self.selection_patches(artifacts, choice)
        with stack, self.assertRaises(AssertionError):
            evaluator.verify_selection()

    def test_bad_selection_blocks_final_data_loading_and_model_creation(self):
        with mock.patch.object(evaluator, "frozen"), \
             mock.patch.object(evaluator, "verify_selection", side_effect=AssertionError("selection changed")), \
             mock.patch.object(evaluator, "cases") as cases, \
             mock.patch.object(evaluator, "CurveRanker") as ranker:
            with self.assertRaisesRegex(AssertionError, "selection changed"):
                evaluator.evaluate_final(evaluator.SEEDS[0])
            cases.assert_not_called()
            ranker.assert_not_called()

    def test_changed_frozen_inference_fingerprint_is_rejected(self):
        with mock.patch.object(evaluator, "protocol"), \
             mock.patch.object(evaluator, "read_json", return_value={"implementation": {"inputs": "original"}}), \
             mock.patch.object(evaluator, "implementation", return_value={"inputs": "changed"}):
            with self.assertRaisesRegex(AssertionError, "implementation changed"):
                evaluator.frozen()

    def test_fingerprint_includes_model_vocabulary_and_operational_adapter_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "models/neural/Qwen3-1.7B-Base"
            adapter = root / "models/neural/adaptation_local"
            vocabulary = root / "data/neural_evaluation/vocabulary.csv"
            base.mkdir(parents=True)
            adapter.mkdir(parents=True)
            vocabulary.parent.mkdir(parents=True)
            (base / "download_manifest.json").write_text(json.dumps({"revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"}))
            (base / "config.json").write_text("{}")
            (base / "model.safetensors").write_bytes(b"CPU fixture, not model weights")
            (base / "merges.txt").write_text("fixture")
            (adapter / "adapter_model.safetensors").write_bytes(b"CPU fixture, not adapter weights")
            config = {"r": 8, "lora_alpha": 16, "lora_dropout": .05, "target_modules": ["q_proj", "v_proj"]}
            config_path = adapter / "adapter_config.json"
            config_path.write_text(json.dumps(config))
            vocabulary.write_text("word\nexample\n")
            (root / "models/selected_predictor_v3.rds").write_bytes(b"CPU fixture, not R model")
            evaluator.inference_fingerprint.cache_clear()
            try:
                with mock.patch.object(evaluator, "ROOT", root):
                    fingerprint = evaluator.inference_fingerprint()
                    self.assertIn("data/neural_evaluation/vocabulary.csv", fingerprint)
                    self.assertIn("models/selected_predictor_v3.rds", fingerprint)
                    self.assertIn("models/neural/adaptation_local/adapter_config.json", fingerprint)
                    self.assertIn("models/neural/Qwen3-1.7B-Base/model.safetensors", fingerprint)
                    config["lora_alpha"] = 32
                    config_path.write_text(json.dumps(config))
                    evaluator.inference_fingerprint.cache_clear()
                    with self.assertRaises(AssertionError):
                        evaluator.inference_fingerprint()
            finally:
                evaluator.inference_fingerprint.cache_clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)