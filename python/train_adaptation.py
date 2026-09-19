"""Two fixed-budget LoRA experiments. Author: Sonja Sahebzad."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from neural_predictor import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(tokenizer, filename, seed):
    cache = ROOT / "data/adaptation" / (filename + ".npy")
    path = ROOT / "data/adaptation" / (filename + ".txt")
    meta_path = cache.with_suffix(".meta.json")
    expected = {"text_sha256": digest(path), "seed": seed, "block_tokens": 128,
                "base_revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"}
    if cache.exists():
        assert json.loads(meta_path.read_text()) == expected
        return np.load(cache)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    np.random.default_rng(seed).shuffle(lines)
    encoded = tokenizer(lines, add_special_tokens=False)["input_ids"]
    flat = [token for row in encoded for token in row + [tokenizer.eos_token_id]]
    blocks = np.asarray(flat[:len(flat)//128*128], dtype=np.int32).reshape(-1,128)
    np.save(cache, blocks)
    meta_path.write_text(json.dumps(expected,indent=2),encoding="utf-8")
    return blocks


def train(recipe, steps=256, microbatch=2):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(4)
    torch.manual_seed(20261310)
    np.random.seed(20261310)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert 8 % microbatch == 0
    destination = ROOT / "models/neural" / ("adaptation_" + recipe)
    if destination.exists():
        raise FileExistsError(f"Preserve the existing run before starting another: {destination}")
    base = ROOT / "models/neural/Qwen3-1.7B-Base"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True,trust_remote_code=False)
    local = pack(tokenizer,"local_training",20261311)
    external = pack(tokenizer,"taskmaster_training",20261312)
    rng = np.random.default_rng(20261313)
    local_ids = rng.permutation(len(local))
    external_ids = rng.permutation(len(external))
    assert len(local_ids)>=steps*8 and len(external_ids)>=steps*2
    # Match the first six local blocks at each step across the two recipes.
    local_schedule = local[local_ids[:steps*8]].reshape(steps,8,128)
    if recipe=="augmented":
        local_schedule[:,6:,:] = external[external_ids[:steps*2]].reshape(steps,2,128)
    schedule = torch.from_numpy(local_schedule.astype(np.int64))
    model = AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda")
    model.config.use_cache=False
    config = LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
        target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM",bias="none")
    model = get_peft_model(model,config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all("lora_" in name for name,p in model.named_parameters() if p.requires_grad)
    trainable = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    history=[]; started=time.perf_counter()
    accumulation=8//microbatch
    print(f"{recipe}: {trainable:,} trainable adapter parameters; {steps} steps; effective batch 8 x 128 tokens",flush=True)
    for step in range(steps):
        warmup=16
        fraction=min((step+1)/warmup,1.)
        cosine=.5*(1+math.cos(math.pi*max(step-warmup,0)/max(steps-warmup,1)))
        lr=1e-4*fraction*cosine
        for group in optimizer.param_groups: group["lr"]=lr
        optimizer.zero_grad(set_to_none=True)
        value=0.
        for offset in range(0,8,microbatch):
            batch=schedule[step,offset:offset+microbatch].to("cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=model(input_ids=batch,attention_mask=torch.ones_like(batch),labels=batch)
                loss=output.loss/accumulation
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            loss.backward()
            value+=float(loss.detach())
            del output,loss
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm): raise RuntimeError("Non-finite adapter gradients")
        optimizer.step()
        history.append({"step":step+1,"loss":value,"lr":lr,"gradient_norm":float(norm)})
        if step==0 or (step+1)%16==0:
            seconds=time.perf_counter()-started
            print(f"{recipe} step {step+1}/{steps}: loss {value:.4f}, elapsed {seconds:.1f}s",flush=True)
    assert any(bool(torch.any(p.detach()!=0)) for n,p in model.named_parameters() if "lora_B" in n)
    model.save_pretrained(destination,safe_serialization=True)
    manifest={"recipe":recipe,"steps":steps,"effective_batch":8,"block_tokens":128,
        "input_tokens":steps*8*128,"supervised_token_positions":steps*8*127,
        "microbatch":microbatch,"rank":8,"alpha":16,"dropout":.05,
        "target_modules":["q_proj","v_proj"],"peak_lr":1e-4,"warmup_steps":16,
        "seed":20261310,"trainable_parameters":trainable,
        "local_blocks":int(len(local)),"external_blocks":int(len(external)),
        "external_block_fraction":.25 if recipe=="augmented" else 0.,
        "seconds":time.perf_counter()-started,"peak_gpu_mib":torch.cuda.max_memory_allocated()/1024**2,
        "base_revision":"ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "training_code_sha256":digest(__file__),
        "local_data_sha256":digest(ROOT/"data/adaptation/local_training.txt"),
        "external_data_sha256":digest(ROOT/"data/adaptation/taskmaster_training.txt"),
        "adapter_sha256":digest(destination/"adapter_model.safetensors"),"history":history}
    path=ROOT/"models"/f"adaptation_training_{recipe}.json"
    path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Saved adapter: {destination}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--recipe",choices=("local","augmented"),required=True)
    parser.add_argument("--steps",type=int,default=256)
    parser.add_argument("--microbatch",type=int,default=2)
    args=parser.parse_args()
    train(args.recipe,args.steps,args.microbatch)
