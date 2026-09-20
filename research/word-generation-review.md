# Complete-word generation: independent CPU review

**Author: Sonja Sahebzad**

**Date: 2026-09-20**

The reviewed implementation is suitable for the planned neutral GPU correctness checks. This CPU review found no critical partial-word, candidate-union, or answer-option leakage defect. It does not establish predictive improvement or numerical equivalence on a GPU.

## Verified behavior

Twelve semantic tests exercise the actual `WordTrie` and `CompleteWordRanker.predict_variants` class definitions. They are extracted with Python AST so the tests never import inference modules, load weights, or initialize CUDA. GPU-dependent helpers are replaced by deterministic toy values solely to test control flow and candidate handling.

* A valid terminal word is emitted while its longer valid continuations remain eligible.
* An incomplete token path is never emitted as a complete word.
* Six-token paths remain eligible; seven-token paths are excluded under the registered limit.
* Equal beam scores are ordered by canonical token path, independently of insertion order.
* The trie accumulates edge log-likelihoods without replacing them with local branch ranks.
* All old free candidates and their previously computed scores survive the union unchanged.
* Only new candidates are rescored. The original score dictionary is not mutated.
* Answer options are scored after free predictions and cannot change free sets, scores or rankings.
* Extra candidates can legitimately displace an old top-three prediction. Preserving the old candidate set does not guarantee unchanged accuracy.

Code inspection additionally confirms that proposal edge probabilities subtract `logsumexp` over the full model vocabulary before selecting trie edges. They are not renormalized separately over each node's allowed children. The final scorer remains the archived canonical-token likelihood plus word-boundary term. Neither targets nor answer options are passed to the proposal generator.

## Training-vocabulary construction

Using only the pinned local tokenizer and training vocabulary, the actual constructed trie has:

| Quantity | Count |
|---|---:|
| Training vocabulary words | 503,523 |
| Single-token words excluded from new proposals | 22,055 |
| Eligible multiple-token words | 476,776 |
| Words exceeding six tokens | 4,692 |
| Trie nodes | 655,490 |
| Terminal nodes with children | 60,188 |

The 4,692 excluded entries are 0.93% of vocabulary **types**, not 0.93% of observed next words. Existing baseline words remain available even when they exceed the new proposal-depth limit. No development or final-test outcomes were read during this review.

## Limitations to retain in the report

The beam and 128-word proposal cap use cumulative canonical token-path likelihood. The final boundary-mass term is applied later. This is a disclosed approximation: a completed word discarded before final scoring might have ranked higher after the boundary term. Beam search, the training lexicon, canonical tokenization, and the six-token bound constrain what can be proposed. Wider search therefore does not guarantee better final ranking or accuracy.

The existing baseline uses its operational `topk` implementation. Keeping that implementation preserves the intended control; neutral GPU checks must verify its exact shortlist and top-three agreement. Stable path ordering is already present in the new beam. CPU tests cannot validate GPU cache layout, precision, memory limits, or runtime.

One nonblocking generalization issue remains: `WordTrie.__init__` accepts a custom `max_tokens`, while `expand` uses the global `MAX_TOKENS`. These agree in the registered six-token experiment. If future callers raise the constructor limit, the expansion limit should be made consistent. This does not require changing the present fixed configuration.

## Evidence and reproduction

The reviewed canonical UTF-8 source SHA256 is `f72d8438c74651943696419e9d76aa8d5fb820ca47f629f29339ad7b1230863f`. [word_generation_cpu_review.json](word_generation_cpu_review.json) records the source, tokenizer and vocabulary fingerprints. The review did not modify the live evaluator or archived inference files.

From the isolated review worktree:

```powershell
$project = 'C:\Users\csj50\OneDrive\Documents\Sonja report\Data Science Capstone'
$env:CAPSTONE_REVIEW_SOURCE = "$project\python\word_generation_experiment.py"
& "$project\.venv-neural\Scripts\python.exe" -m unittest discover -s tests -p test_word_generation_review.py -v
```

The test evidence is **12 tests passed**. References are the reviewed [implementation](../python/word_generation_experiment.py), the [independent CPU tests](../tests/test_word_generation_review.py), and the earlier [ranking audit](ranking-audit.md). There is no new performance claim in this review.
