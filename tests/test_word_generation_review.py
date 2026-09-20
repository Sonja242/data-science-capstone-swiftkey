"""Independent CPU-only review tests. Author: Sonja Sahebzad.

Extract only the two class definitions with AST. No inference module import,
model loading, CUDA access, corpus examples or held-out outcomes are needed.
Set CAPSTONE_REVIEW_SOURCE to review a live source file from a separate worktree.
"""
import ast
from collections import Counter
from contextlib import nullcontext
import os
from pathlib import Path
import re
import types
import unittest

SOURCE = Path(os.environ.get("CAPSTONE_REVIEW_SOURCE", str(
    Path(__file__).resolve().parents[1] / "python/word_generation_experiment.py")))
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
         and node.name in ("WordTrie", "CompleteWordRanker")]
assert len(nodes) == 2
namespace = {"Counter": Counter, "WORD": re.compile(r"[a-z]+(?:'[a-z]+)*\Z"),
             "MAX_TOKENS": 6, "CAP": 128, "BEAMS": (16, 64),
             "AdaptedRanker": type("NoModelBase", (), {})}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
WordTrie = namespace["WordTrie"]
CompleteWordRanker = namespace["CompleteWordRanker"]


class WordTrieReviewTests(unittest.TestCase):
    def test_terminal_word_can_continue_to_longer_word(self):
        trie = WordTrie([("the", [1, 2]), ("there", [1, 2, 3])])
        completed = {}
        frontier = trie.expand([(0, 0.)], [[-.1]], 1, 16, completed)
        self.assertFalse(completed)
        frontier = trie.expand(frontier, [[-.2]], 2, 16, completed)
        self.assertAlmostEqual(completed["the"], -.3)
        self.assertEqual(len(frontier), 1)
        frontier = trie.expand(frontier, [[-.4]], 3, 16, completed)
        self.assertAlmostEqual(completed["there"], -.7)
        self.assertFalse(frontier)

    def test_partial_word_is_not_emitted(self):
        trie = WordTrie([("science", [1, 2, 3])])
        completed = {}
        frontier = trie.expand([(0, 0.)], [[-.1]], 1, 16, completed)
        trie.expand(frontier, [[-.2]], 2, 16, completed)
        self.assertFalse(completed)

    def test_beam_tie_is_resolved_by_token_path_not_insertion(self):
        trie = WordTrie([("zulu", [20, 5]), ("alpha", [10, 5])])
        frontier = trie.expand([(0, 0.)], [[-.1, -.1]], 1, 1, {})
        self.assertEqual(trie.paths[frontier[0][0]], (10,))

    def test_depth_limit_retains_six_and_excludes_seven(self):
        trie = WordTrie([("permitted", [1, 2, 3, 4, 5, 6]),
                         ("overlong", [1, 2, 3, 4, 5, 6, 7])])
        self.assertIn("permitted", trie.terminal)
        self.assertNotIn("overlong", trie.terminal)
        self.assertEqual(trie.statistics["over_depth_words"], 1)

    def test_invalid_and_single_token_words_are_not_added(self):
        trie = WordTrie([("UPPER", [1, 2]), ("two words", [3, 4]),
                         ("single", [5]), ("can't", [6, 7])])
        self.assertEqual({x for x in trie.terminal if x}, {"can't"})
        self.assertEqual(trie.statistics["filtered_words"], 2)
        self.assertEqual(trie.statistics["single_token_words"], 1)

    def test_duplicate_path_for_different_words_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "share a token path"):
            WordTrie([("first", [1, 2]), ("second", [1, 2])])

    def test_full_edge_log_likelihood_accumulates(self):
        trie = WordTrie([("word", [1, 2])])
        completed = {}
        frontier = trie.expand([(0, 0.)], [[-8.0]], 1, 16, completed)
        trie.expand(frontier, [[-3.0]], 2, 16, completed)
        self.assertEqual(completed["word"], -11.0)


class FreeChoiceSeparationTests(unittest.TestCase):
    def make_ranker(self):
        ranker = CompleteWordRanker.__new__(CompleteWordRanker)
        ranker.torch = types.SimpleNamespace(inference_mode=nullcontext)
        ranker._prefill = lambda phrase: ([1], object())
        ranker.baseline_set = lambda prefill, ng: ["the", "science", "old"]
        ranker.calls = []
        values = {"the": -1., "science": -2., "old": -4., "new": -.5,
                  "second": -3., "answer": -.01}
        def scorer(prefix, prefill, words):
            ranker.calls.append(tuple(words))
            return {w: values[w] for w in words}
        ranker._score_candidates = scorer
        ranker.proposals = lambda prefix, prefill, beam: (
            ["the", "new"] if beam == 16 else ["second", "new"], {"beam": beam})
        return ranker

    def test_options_cannot_change_free_sets_scores_or_ranking(self):
        ranker = self.make_ranker()
        free = ranker.predict_variants("neutral phrase")
        offered = ranker.predict_variants("neutral phrase", choices=["answer", "the"])
        self.assertEqual(free["baseline"], offered["baseline"])
        self.assertEqual(free["candidates"], offered["candidates"])
        self.assertEqual(offered["options"]["words"][0], "answer")
        self.assertNotIn("answer", offered["candidates"]["beam16"]["words"])

    def test_old_scores_are_preserved_and_only_additions_rescored(self):
        ranker = self.make_ranker()
        result = ranker.predict_variants("neutral phrase")
        self.assertEqual(ranker.calls, [("the", "science", "old"), ("new",), ("second", "new")])
        baseline = dict(result["baseline"]["scores"])
        for item in result["candidates"].values():
            union = {**baseline, **dict(item["scores"])}
            self.assertTrue(set(baseline) <= set(union))
            self.assertTrue(all(union[w] == value for w, value in baseline.items()))

    def test_new_word_can_legitimately_displace_old_top_three(self):
        result = self.make_ranker().predict_variants("neutral phrase")
        self.assertIn("old", result["baseline"]["words"])
        self.assertNotIn("old", result["candidates"]["beam16"]["words"])

    def test_empty_beams_gives_baseline_only(self):
        result = self.make_ranker().predict_variants("neutral phrase", beams=())
        self.assertFalse(result["candidates"])
        self.assertEqual(result["baseline"]["words"], ["the", "science", "old"])

    def test_multiword_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "single normalized"):
            self.make_ranker().predict_variants("neutral phrase", choices=["two words"])


if __name__ == "__main__":
    unittest.main()
