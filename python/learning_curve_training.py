"""Controlled local-text learning trajectories. Author: Sonja Sahebzad.

This module performs training only. It never reads final-test text or evaluates
models. Importing it, --help, --check-only and the unit tests require no GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

import numpy as np


BASE_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SEEDS = (20261410, 20261411, 20261412)
CHECKPOINTS = (256, 512, 1024)
HORIZON = 1024
WARMUP = 16
BATCH = 8
BLOCK = 128
TEXT_SHA256 = "16472ddcc359ba584ccfc86d2a54e08616c6fea8cd5b780dbc9b8f05567fcb9d"
CACHE_SHA256 = "1e10149f99d4a397daeea41055cc8788cb15fa301a331e7c9aa009bd440648f9"
DEVELOPMENT_SHA256 = "fede576d6e22aa8553aec5c2b6c05314b78b29426ee95bf72d80fc79bd9571ab"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_code_sha256(path):
    # Universal newline decoding makes this stable after a Windows Git checkout.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def json_write_new(path, value):
    """Exclusive creation prevents silently replacing an audit artifact."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def learning_rate(step):
    """One-based update index; all trajectory stages share the 1024-step curve."""
    if not 1 <= step <= HORIZON:
        raise ValueError("step must be in 1..1024")
    if step <= WARMUP:
        return 1e-4 * step / WARMUP
    return 1e-4 * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (HORIZON - WARMUP)))


def make_schedule(block_count, seed):
    """The full 8192-block schedule is fixed before any optimization starts."""
    if block_count < HORIZON * BATCH:
        raise ValueError("Not enough distinct blocks for the full training trajectory")
    return np.random.default_rng(seed).permutation(block_count)[:HORIZON * BATCH].reshape(HORIZON, BATCH)


def schedule_sha256(schedule):
    # Explicit byte order and type avoid platform-dependent schedule hashes.
    return hashlib.sha256(np.asarray(schedule, dtype="<i8").tobytes(order="C")).hexdigest()


def token_counts(steps):
    if not 1 <= steps <= HORIZON:
        raise ValueError("steps must be in 1..1024")
    return {"consumed_blocks": steps * BATCH, "input_tokens": steps * BATCH * BLOCK,
            "supervised_token_positions": steps * BATCH * (BLOCK - 1)}


def validate_plan(seed, steps, microbatch, checkpoints, run_label):
    if seed not in SEEDS:
        raise ValueError(f"Use one of the predeclared seeds: {SEEDS}")
    if microbatch not in (2, 4, 8):
        raise ValueError("microbatch must be 2, 4 or 8")
    if not 1 <= steps <= HORIZON:
        raise ValueError("steps must be in 1..1024")
    checkpoints = tuple(checkpoints)
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be unique and strictly increasing")
    if checkpoints[-1] != steps or checkpoints[0] < 1:
        raise ValueError("The final checkpoint must equal steps, with all checkpoints positive")
    if run_label is None:
        if steps != HORIZON or checkpoints != CHECKPOINTS:
            raise ValueError("Official runs require 1024 steps and checkpoints 256, 512, 1024")
    elif not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,39}", run_label):
        raise ValueError("Pilot run labels require 1..40 lowercase letters, digits or underscores")
    return {"seed": seed, "steps": steps, "microbatch": microbatch,
            "checkpoints": list(checkpoints), "run_label": run_label,
            "purpose": "official_trajectory" if run_label is None else "pilot_training_only"}


def output_paths(root, plan):
    suffix = f"seed_{plan['seed']}"
    if plan["run_label"] is not None:
        suffix = f"pilot_{plan['run_label']}_{suffix}"
    return (root / "models/neural" / f"learning_curve_{suffix}",
            root / "models" / f"learning_curve_training_{suffix}.json")


def preflight(root, plan, hash_model=True):
    """Read-only CPU validation; hash the development file without inspecting cases."""
    root = Path(root).resolve(strict=True)
    data = root / "data/adaptation"
    expected = {"local_training.txt": TEXT_SHA256, "local_training.npy": CACHE_SHA256,
                "development.csv": DEVELOPMENT_SHA256}
    actual = {name: sha256(data / name) for name in expected}
    if actual != expected:
        raise ValueError("Training cache, source text or fixed development file changed")
    meta_path = data / "local_training.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    if meta != {"text_sha256": TEXT_SHA256, "seed": 20261311,
                "block_tokens": BLOCK, "base_revision": BASE_REVISION}:
        raise ValueError("Tokenization cache metadata does not match the pinned recipe")
    blocks = np.load(data / "local_training.npy", mmap_mode="r", allow_pickle=False)
    if blocks.shape != (11282, BLOCK) or blocks.dtype != np.int32:
        raise ValueError("Expected the original 11282 x 128 int32 packed blocks")
    if int(blocks.min()) < 0 or int(blocks.max()) >= 151936:
        raise ValueError("Token IDs outside the pinned model vocabulary")
    base = root / "models/neural/Qwen3-1.7B-Base"
    download = json.loads((base / "download_manifest.json").read_text(encoding="utf-8-sig"))
    if download.get("repo") != "Qwen/Qwen3-1.7B-Base" or download.get("revision") != BASE_REVISION:
        raise ValueError("Unexpected base model repository or immutable revision")
    model_files = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"]
    model_hashes = {name: sha256(base / name) for name in model_files} if hash_model else {}
    schedule = make_schedule(len(blocks), plan["seed"])
    destination, summary_path = output_paths(root, plan)
    if destination.exists() or summary_path.exists():
        raise FileExistsError(f"Existing run is preserved. No overwrite or resume: {destination}")
    fingerprint = {"data_sha256": actual, "cache_metadata_sha256": sha256(meta_path),
                   "base_repository": download["repo"], "base_revision": BASE_REVISION,
                   "base_file_sha256": model_hashes, "available_blocks": len(blocks),
                   "block_schedule_sha256": schedule_sha256(schedule),
                   "training_code_canonical_sha256": canonical_code_sha256(__file__)}
    return blocks, schedule, fingerprint


def hyperparameters(plan):
    return {"effective_batch": BATCH, "microbatch": plan["microbatch"],
            "gradient_accumulation": BATCH // plan["microbatch"], "block_tokens": BLOCK,
            "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05,
            "target_modules": ["q_proj", "v_proj"], "lora_bias": "none",
            "optimizer": "AdamW", "betas": [.9, .999], "epsilon": 1e-8,
            "weight_decay": .01, "gradient_clip_norm": 1., "peak_learning_rate": 1e-4,
            "warmup_steps": WARMUP, "cosine_horizon_steps": HORIZON,
            "dtype": "bfloat16", "attention": "sdpa", "context_tokens": BLOCK,
            "gradient_checkpointing": "use_reentrant=False", "external_data_fraction": 0.,
            "packing": "Original shuffled local lines, EOS-separated, 128-token blocks",
            "attention_can_cross_eos": True, "blocks_repeated_within_trajectory": False}


def checkpoint_metadata(plan, step, fingerprint, schedule, history, runtime, files):
    if step not in plan["checkpoints"] or len(history) != step or history[-1]["step"] != step:
        raise ValueError("Checkpoint history is incomplete or step was not predeclared")
    return {"schema_version": 1, "author": "Sonja Sahebzad", "plan": plan,
            "completed_step": step, **token_counts(step), "fingerprint": fingerprint,
            "consumed_schedule_sha256": schedule_sha256(schedule[:step]),
            "hyperparameters": hyperparameters(plan), "runtime": runtime,
            "adapter_files_sha256": files, "history": history,
            "training_only": True, "contains_test_metrics": False,
            "schedule_comparison_note": "256 is an intermediate point on a 1024-step LR schedule, not a repeat of the earlier 256-step cosine run."}


def train(root, plan):
    blocks, schedule, fingerprint = preflight(root, plan)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This fixed experiment requires a BF16-capable CUDA GPU")
    torch.set_num_threads(4)
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # SDPA kernels can still vary across hardware/software; record this limitation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    destination, summary_path = output_paths(root, plan)
    destination.mkdir(parents=True, exist_ok=False)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors")},
                   "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu_name": torch.cuda.get_device_name(0),
                   "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                   "deterministic_algorithms_warn_only": True,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "reproducibility_limit": "Identical seeds and data do not guarantee bitwise equality across platforms or nondeterministic CUDA kernels."}
    run = {"author": "Sonja Sahebzad", "plan": plan, "fingerprint": fingerprint,
           "hyperparameters": hyperparameters(plan), "environment": environment,
           "started_utc": utc_now(), "checkpoints": []}
    json_write_new(destination / "protocol.json", run)
    np.save(destination / "block_schedule.npy", np.asarray(schedule, dtype="<i8"), allow_pickle=False)
    try:
        base = root / "models/neural/Qwen3-1.7B-Base"
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                    trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        model.config.use_cache = False
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(model, config)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        parameters = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in parameters)
        if trainable != 1605632 or any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
            raise RuntimeError("Unexpected trainable parameter set")
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        saved_seconds = 0.
        history = []
        print(f"Seed {plan['seed']}: {trainable:,} trainable parameters; {plan['steps']} updates; microbatch {plan['microbatch']}; checkpoints {plan['checkpoints']}", flush=True)
        for step in range(1, plan["steps"] + 1):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step)
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            rows = np.asarray(blocks[schedule[step - 1]], dtype=np.int64)
            for offset in range(0, BATCH, plan["microbatch"]):
                batch = torch.from_numpy(rows[offset:offset + plan["microbatch"]]).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids=batch, attention_mask=torch.ones_like(batch), labels=batch)
                    loss = output.loss / (BATCH // plan["microbatch"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                value += float(loss.detach())
                del output, loss, batch
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if any(p.grad is None for p in parameters):
                raise RuntimeError("A trainable adapter parameter has no gradient")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("A frozen parameter unexpectedly received gradients")
            optimizer.step()
            history.append({"step": step, "loss": value, "learning_rate": learning_rate(step),
                            "gradient_norm_before_clipping": float(norm)})
            if step == 1 or step % 16 == 0 or step == plan["steps"]:
                print(f"Seed {plan['seed']} step {step}/{plan['steps']}: loss {value:.4f}, elapsed {time.perf_counter() - started:.1f}s", flush=True)
            if step in plan["checkpoints"]:
                torch.cuda.synchronize()
                checkpoint_start = time.perf_counter()
                if not any(bool(torch.any(p.detach() != 0)) for n, p in model.named_parameters() if "lora_B" in n):
                    raise RuntimeError("Adapter output parameters remained at initialization")
                if not all(bool(torch.isfinite(p.detach()).all()) for p in parameters):
                    raise FloatingPointError("Non-finite adapter parameters")
                folder = destination / f"step_{step}"
                folder.mkdir(exist_ok=False)
                model.save_pretrained(folder, safe_serialization=True)
                files = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
                runtime = {"trainable_parameters": trainable,
                           "training_seconds_excluding_checkpoint_io": checkpoint_start - started - saved_seconds,
                           "wall_seconds_since_training_start": checkpoint_start - started,
                           "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                           "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                           "recorded_utc": utc_now(), "environment": environment}
                metadata = checkpoint_metadata(plan, step, fingerprint, schedule, history, runtime, files)
                json_write_new(folder / "training_metadata.json", metadata)
                run["checkpoints"].append({"step": step, "relative_path": folder.relative_to(root).as_posix(),
                     "metadata_sha256": sha256(folder / "training_metadata.json"),
                     "adapter_sha256": files["adapter_model.safetensors"], **token_counts(step)})
                saved_seconds += time.perf_counter() - checkpoint_start
                print(f"Saved checkpoint: {folder}", flush=True)
        run.update({"completed_utc": utc_now(), "status": "complete", "history": history,
                    "trainable_parameters": trainable, **token_counts(plan["steps"]),
                    "checkpoint_io_seconds": saved_seconds})
        json_write_new(destination / "completion.json", run)
        json_write_new(summary_path, run)
        print(f"Complete training manifest: {summary_path}", flush=True)
    except BaseException as error:
        json_write_new(destination / "failure.json", {"status": "failed", "recorded_utc": utc_now(),
                       "error_type": type(error).__name__, "message": str(error),
                       "instruction": "Preserve this partial run. Automatic resume and overwrite are disabled."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--microbatch", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--steps", type=int, default=HORIZON,
                        help="Stop after this many steps. The LR horizon remains 1024; short runs require --run-label.")
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--run-label", help="Separate pilot output prefix; never used for an official trajectory")
    parser.add_argument("--check-only", action="store_true", help="Validate and print a plan without loading a model or writing files")
    args = parser.parse_args()
    points = args.checkpoints if args.checkpoints is not None else sorted(set([p for p in CHECKPOINTS if p <= args.steps] + [args.steps]))
    plan = validate_plan(args.seed, args.steps, args.microbatch, points, args.run_label)
    root = args.project_root.resolve(strict=True)
    if args.check_only:
        _, _, fingerprint = preflight(root, plan)
        print(json.dumps({"plan": plan, "fingerprint": fingerprint,
                          "hyperparameters": hyperparameters(plan), "output_paths": [str(p) for p in output_paths(root, plan)]}, indent=2))
    else:
        train(root, plan)


if __name__ == "__main__":
    main()
