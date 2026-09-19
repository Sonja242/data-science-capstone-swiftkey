"""Locally evaluated neural reranker. Author: Sonja Sahebzad.

Ranks whole-word candidates using canonical token likelihood and the probability
of a following word boundary. The shortlist does not contain the target unless
it is independently proposed by the neural model or the n-gram model.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models/neural/hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*\Z")
REPOS = {"0.6B": "Qwen/Qwen3-0.6B-Base", "1.7B": "Qwen/Qwen3-1.7B-Base"}


def download_model(size):
    from huggingface_hub import HfApi, snapshot_download
    repo = REPOS[size]
    folder = ROOT / "models/neural" / repo.split("/")[1]
    manifest_path = folder / "download_manifest.json"
    if manifest_path.exists():
        revision = json.loads(manifest_path.read_text())["revision"]
    else:
        revision = HfApi().model_info(repo, token=False).sha
    snapshot_download(repo_id=repo, revision=revision, token=False, local_dir=folder,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "*.md"], max_workers=2)
    manifest = {"repo": repo, "revision": revision, "license": "Apache-2.0"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


class NeuralWordRanker:
    def __init__(self, size, max_context=128, shortlist=64, batch_size=32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("This experiment expects CUDA. No CPU fallback is started silently.")
        torch.set_num_threads(4)
        torch.manual_seed(20261220)
        self.size = size
        self.max_context = max_context
        self.shortlist = shortlist
        self.batch_size = batch_size
        self.folder = ROOT / "models/neural" / REPOS[size].split("/")[1]
        self.manifest = json.loads((self.folder / "download_manifest.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.folder, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.folder, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
        with (ROOT / "data/neural_evaluation/vocabulary.csv").open(encoding="utf-8-sig", newline="") as f:
            vocabulary = {row["word"] for row in csv.DictReader(f)}
        self.decoded = self.tokenizer.batch_decode([[i] for i in range(len(self.tokenizer))],
                                                 clean_up_tokenization_spaces=False)
        eligible = []
        for i, token in enumerate(self.decoded):
            word = token.strip().lower()
            if token.startswith(" ") and WORD.fullmatch(word) and word in vocabulary:
                eligible.append(i)
        self.eligible = torch.tensor(eligible, device="cuda")
        # A token starting with another letter/apostrophe continues the same word.
        boundary = [i for i, token in enumerate(self.decoded)
                    if not token or not (token[0].isalpha() or token[0] == "'")]
        self.boundary = torch.tensor(boundary, device="cuda")
        self.token_cache = {}

    def _tokens(self, word):
        if word not in self.token_cache:
            self.token_cache[word] = self.tokenizer.encode(" " + word, add_special_tokens=False)
        return self.token_cache[word]

    def _prefill(self, phrase):
        torch = self.torch
        ids = self.tokenizer.encode(phrase.strip(), add_special_tokens=False)[-self.max_context:]
        if not ids:
            raise ValueError("Enter an English phrase.")
        inputs = torch.tensor([ids], device="cuda")
        output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            use_cache=True, logits_to_keep=1)
        return ids, output

    def _score_candidates(self, prefix_ids, prefill, words):
        torch = self.torch
        initial = torch.log_softmax(prefill.logits[0, -1].float(), dim=-1)
        scores = {}
        for start in range(0, len(words), self.batch_size):
            batch_words = words[start:start + self.batch_size]
            sequences = [self._tokens(word) for word in batch_words]
            width = max(map(len, sequences))
            batch = len(sequences)
            token_ids = torch.full((batch, width), self.tokenizer.eos_token_id, device="cuda", dtype=torch.long)
            mask = torch.zeros((batch, width), device="cuda", dtype=torch.long)
            for i, seq in enumerate(sequences):
                token_ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
                mask[i, :len(seq)] = 1
            cache = copy.deepcopy(prefill.past_key_values)
            cache.batch_repeat_interleave(batch)
            attention_mask = torch.cat((torch.ones((batch, len(prefix_ids)), device="cuda", dtype=torch.long), mask), dim=1)
            output = self.model(input_ids=token_ids, attention_mask=attention_mask,
                                past_key_values=cache, use_cache=True, logits_to_keep=0)
            logits = output.logits.float()
            normalizers = torch.logsumexp(logits, dim=-1)
            for i, (word, seq) in enumerate(zip(batch_words, sequences)):
                value = initial[seq[0]]
                if len(seq) > 1:
                    positions = torch.arange(len(seq)-1, device="cuda")
                    value = value + (logits[i, positions, torch.tensor(seq[1:], device="cuda")]
                                     - normalizers[i, positions]).sum()
                last = len(seq)-1
                value = value + torch.logsumexp(logits[i, last, self.boundary], dim=-1) - normalizers[i, last]
                scores[word] = float(value.item())
            del output, cache, logits, normalizers
        if not all(math.isfinite(v) for v in scores.values()):
            raise RuntimeError("A non-finite word score was produced.")
        return scores

    def predict(self, phrase, ngram_candidates=(), choices=()):
        torch = self.torch
        with torch.inference_mode():
            prefix_ids, prefill = self._prefill(phrase)
            eligible_logits = prefill.logits[0, -1, self.eligible]
            top = self.eligible[torch.topk(eligible_logits, self.shortlist).indices].tolist()
            free_words = list(dict.fromkeys([self.decoded[i].strip().lower() for i in top]
                                           + list(ngram_candidates)))
            free_words = [w for w in free_words if WORD.fullmatch(w)]
            options = list(dict.fromkeys(w.strip().lower() for w in choices))
            if any(not WORD.fullmatch(w) for w in options):
                raise ValueError("Each supplied option must be one English word.")
            union = list(dict.fromkeys(free_words + options))
            scores = self._score_candidates(prefix_ids, prefill, union)
            free_rank = sorted(free_words, key=lambda w: (-scores[w], w))
            choice_rank = sorted(options, key=lambda w: (-scores[w], w))
            return {"words": free_rank[:3], "shortlist": free_words,
                    "choices": choice_rank, "log_scores": {w: scores[w] for w in free_rank[:3] + choice_rank}}

    def check_scoring(self):
        """Compare cached scores with full-sequence scores, including a split word."""
        torch = self.torch
        phrase = "we are talking about"
        words = ["the", "science", "uncharacteristically"]
        with torch.inference_mode():
            prefix, prefill = self._prefill(phrase)
            cached = self._score_candidates(prefix, prefill, words)
            for word in words:
                seq = self._tokens(word)
                ids = torch.tensor([prefix + seq], device="cuda")
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)
                direct = sum(log_probs[len(prefix)-1+k, token] for k, token in enumerate(seq))
                direct = direct + torch.logsumexp(log_probs[-1, self.boundary], dim=-1)
                if abs(float(direct.item())-cached[word]) > .06:
                    raise AssertionError(f"Cached and direct word scores differ for {word}.")
        print("PASS: cached and full scoring agree, including word-boundary mass and multiple-token words.", flush=True)


def evaluate(ranker, split, limit=None):
    source = ROOT / "data/neural_evaluation" / f"{split}.csv"
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    details = []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        result = ranker.predict(row["prefix"], json.loads(row["ngram_candidates_json"]), json.loads(row["options_json"]))
        actual = row["actual"]
        details.append({"source": row["source"], "line_hash": row["line_hash"],
                        "rank": result["words"].index(actual)+1 if actual in result["words"] else 0,
                        "choice_correct": result["choices"][0] == actual,
                        "shortlist_contains_target": actual in result["shortlist"]})
        if (i+1) % 25 == 0:
            print(f"{ranker.size} {split}: {i+1}/{len(rows)}", flush=True)
    elapsed = time.perf_counter()-started
    summary = {"model": REPOS[ranker.size], "split": split, "cases": len(details),
               "top1": sum(d["rank"] == 1 for d in details)/len(details),
               "top3": sum(d["rank"] > 0 for d in details)/len(details),
               "synthetic_choice_accuracy": sum(d["choice_correct"] for d in details)/len(details),
               "shortlist_recall": sum(d["shortlist_contains_target"] for d in details)/len(details),
               "milliseconds": elapsed*1000/len(details), "torch": ranker.torch.__version__, "gpu": ranker.torch.cuda.get_device_name(), "peak_gpu_mib": ranker.torch.cuda.max_memory_allocated()/1024**2, "weight_mib": sum(p.stat().st_size for p in ranker.folder.glob("*.safetensors"))/1024**2}
    artifact = {"summary": summary, "details": details, "download": ranker.manifest,
                "configuration": {"context_tokens": ranker.max_context, "neural_shortlist": ranker.shortlist,
                                   "ngram_shortlist": 20, "batch_size": ranker.batch_size},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "limitations": ["Synthetic answer options are not actual quiz accuracy.",
                                "Unknown overlap with pretrained-model training data.",
                                "Free predictions rerank a finite shortlist; they are not exhaustive vocabulary search."]}
    out = ROOT / "models" / f"neural_{ranker.size}_{split}{'_pilot' if limit else ''}.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=REPOS, default="0.6B")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--evaluate", choices=("development", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--phrase")
    parser.add_argument("--choices", nargs="*", default=[])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.download:
        download_model(args.size)
    if args.evaluate or args.phrase or args.request:
        ranker = NeuralWordRanker(args.size)
        ranker.check_scoring()
        if args.evaluate:
            evaluate(ranker, args.evaluate, args.limit)
        if args.phrase or args.request:
            request = json.loads(args.request.read_text(encoding="utf-8-sig")) if args.request else {
                "phrase": args.phrase, "choices": args.choices, "ngram_candidates": []}
            prediction = ranker.predict(request["phrase"], request.get("ngram_candidates", []), request.get("choices", []))
            if args.output:
                args.output.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
            else:
                print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()

