# Development-only ranking audit

**Author: Sonja Sahebzad**
**Date: 2026-09-20**

The archived development results support a ranking bottleneck, but do not demonstrate a faulty scorer. The locally adapted model places the observed next word in its shortlist for 467 of 600 examples (77.8%), and in its top three for 222 (37.0%). The remaining cases divide into 245 present-but-below-top-three targets (40.8%) and 133 absent targets (22.2%). Among covered targets, 47.5% reach the top three. These are exact next-word matches against one observed continuation, not judgments that every other continuation is implausible.

This audit reads only the fixed adaptation **development** set. Earlier final tests are already consumed evidence and cannot become fresh final tests. No earlier or newly reserved final-test files were opened. No GPU model, training, or inference was run.

## What changed between candidate lists?

| Development variant | Top-3 correct | Target present | Present but outside top 3 | Target absent |
|---|---:|---:|---:|---:|
| Base, width 64 | 211/600 | 411/600 | 200/600 | 189/600 |
| Base, width 256 | 211/600 | 463/600 | 252/600 | 137/600 |
| Local adaptation, width 256 | 222/600 | 467/600 | 245/600 | 133/600 |
| Dialogue mixture, width 256 | 217/600 | 472/600 | 255/600 | 128/600 |

Increasing baseline width newly covers 52 targets and loses none. Every one of those 52 still ranks below three. All 600 saved target ranks are unchanged. The archived ranks are censored: `rank=0` means outside the top three, including absent targets. We cannot infer rank 4 versus rank 200, unchanged full predictions, or full set nesting from those records.

Local adaptation gains 17 top-three matches and loses 6. All 17 gains concern targets already covered by the baseline; its 14 newly covered targets remain outside the top three. Ten previously covered targets disappear. This indicates that the observed gain mainly reflects ranking changes within the existing candidate pool, while adaptation can also change which words are proposed.

## Where are errors concentrated?

Results below use local adaptation and remain descriptive, rather than separate significance claims.

| Development group | Cases | Top-3 correct | Target present |
|---|---:|---:|---:|
| Blogs | 200 | 37.0% | 80.0% |
| News | 200 | 41.0% | 78.5% |
| Twitter | 200 | 33.0% | 75.0% |
| Context of 1 to 3 words | 158 | 28.5% | 68.4% |
| Context of 4 to 9 words | 150 | 36.0% | 80.7% |
| Context of 10 or more words | 292 | 42.1% | 81.5% |
| Target of 1 to 3 characters | 220 | 55.9% | 95.0% |
| Target of 4 to 6 characters | 250 | 28.8% | 75.6% |
| Target of 7 or more characters | 130 | 20.8% | 53.1% |

Groups overlap and are not controlled experiments. Longer words differ in frequency and content; these figures do not isolate a causal word-length penalty. The complete four-model breakdown is retained in [development_subgroups.csv](ranking_audit/development_subgroups.csv).

## Tokenization and candidate construction

The neural generator proposes whole words represented by one eligible token, and unions them with the R model's top 20 words. The final scorer can evaluate multiple-token words, but the neural generator does not enumerate them.

* Of 549 targets with one canonical token, 461 are covered and 220 reach the top three.
* Of 51 targets requiring multiple canonical tokens, only 6 are covered and 2 reach the top three.
* For 33 examples the target is neither represented by any eligible neural token nor included in the R top 20. Those targets cannot appear under this exact candidate-generation procedure, regardless of reranking. This is a structural limitation, not evidence that a broader generator would predict them correctly.
* There are 36,415 eligible token IDs but only 25,639 unique lowercase words. Case-fold aliases affect 9,023 words. After deduplication, a nominal width of 256 token IDs can yield fewer than 256 distinct neural words. Actual per-example candidate counts were not archived.

CPU tokenization checks compare the full prefix plus word against separately encoded prefix and space-prefixed word for the target and R candidates. All 12,316 comparisons agree. Only two development prefixes exceed the 128-token context limit. Neither finding supports a widespread prefix-boundary concatenation defect.

The scorer multiplies canonical next-token likelihoods and the probability of a following word boundary. Code inspection found aligned causal positions, masked padding, a final-word boundary term, and free-candidate separation from supplied choices. Its score is a **canonical token-path likelihood with a boundary term**, not the exact probability summed over every capitalization and tokenization of the normalized word. Lowercasing changes the canonical ID for 14,360 eligible token IDs. The practical ranking effect requires inference evidence and is not established here.

The boundary heuristic examines individually decoded tokens. There are 1,188 tokens beginning with a Unicode replacement character when decoded alone. That count flags a potential byte-decoding approximation, not a demonstrated scoring error or its size. The local task's ASCII normalization, partial bytes, punctuation, and special tokens must be considered before changing the boundary rule. Archived inference sources remain untouched.

## Prespecified GPU correctness check for the coordinator

The following is a design, not a completed GPU test. It can accompany the planned learning-curve experiment without tuning on a final test.

1. On fixed neutral phrases, include one-token words, split-token words, contractions, and word-prefix pairs such as `the` and `there`. Include a context longer than 128 tokens and verify identical truncation.
2. Use one prefill per phrase. Obtain neural widths 64 and 256 from the same logits, combine each with the identical R top 20, and record unique word counts. Check `S64` is a subset of `S256`. Record cutoff ties; use a documented deterministic tie rule.
3. Score `S256` once and rank each width only over its own set. Independently score `S64` to verify shared-candidate scores within a prespecified tolerance. Compare batched cached scores to an independent full-sequence calculation, including the boundary mass. Record numerical error and any ranking differences near ties.
4. Repeat with candidate order permuted, a different batch size, and supplied answer options. Free sets and free rankings must remain independent of the options. Options may be scored, but must not be injected into free candidates.
5. For development diagnostics, save exact full ranks, explicit absent-target status, context length, target token length, actual unique-list size, and source. Never insert the target to improve free coverage. A score calculated for an offered target remains separate from its free-list membership.

The parent study should freeze these checks and its selection rule before opening its newly reserved final test. All GPU experiments share one queue.

## Conclusion and next steps

The development evidence justifies the planned local-data learning curve and a smaller-shortlist speed comparison. It does not justify a promise of 85% or 99.8% free-text accuracy. A future, separately specified experiment could test bounded multiple-token word generation, because 45 of 51 multiple-token targets are currently absent. Beam generation, candidate aliases, alternative boundary handling, length normalization, and n-gram interpolation are **untested hypotheses**. Changing a score or adding candidates may lose correct predictions as well as gain them.

## Reproducibility and references

From this worktree, using the existing project Python environment:

```powershell
$project = 'C:\Users\csj50\OneDrive\Documents\Sonja report\Data Science Capstone'
& "$project\.venv-neural\Scripts\python.exe" python/audit_ranking_development.py --project-root $project --out-dir research/ranking_audit
& "$project\.venv-neural\Scripts\python.exe" -m unittest discover -s tests -p test_ranking_audit.py
```

The audit uses the CPU `tokenizers` library and Python standard library. It refuses final-test artifacts, checks dataset/source fingerprints and row alignment, and rejects output paths inside the live project. Aggregate outputs contain no source text. [development_audit.json](ranking_audit/development_audit.json) preserves input hashes, counts and transition tables. The archived inference-source SHA256 remains `5a4280640c49e427daaef398597492dc229af8ebc7b9ef7241bcdb012cf5b528`.

Evidence comes directly from [neural_predictor.py](../python/neural_predictor.py), [adaptation_experiment.py](../python/adaptation_experiment.py), [20_prepare_adaptation.R](../20_prepare_adaptation.R), and the four `models/adaptation_*_development.json` artifacts. The tokenizer is the locally pinned [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) tokenizer, SHA256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`. This report makes no new claim about the model publisher's training data or performance.
