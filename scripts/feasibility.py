"""Is LoRA fine-tuning actually tractable on this machine?

Run before designing anything around it. Loads the real model, attaches the real
adapter, and times real training steps -- a guess about MPS throughput is not
worth building a project on.
"""
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.bfloat16
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"== {MODEL} {DTYPE} batch={BATCH} ==")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE)
print(f"loaded in {time.time() - t0:.0f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA trainable {trainable:,} / {total:,} = {100*trainable/total:.3f}%")

model.to(DEV).train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# Batch shaped like the real task: ~320 tokens, batch 4.
ids = torch.randint(0, tok.vocab_size, (BATCH, 320), device=DEV)
times = []
for i in range(6):
    s = time.time()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    if DEV == "mps":
        torch.mps.synchronize()
    times.append(time.time() - s)
    print(f"  step {i}: {times[-1]:.2f}s  loss {loss.item():.3f}")

steady = sum(times[2:]) / len(times[2:])
print(f"\nsteady-state {steady:.2f}s/step at batch {BATCH} x 320 tok")
if DEV == "mps":
    print(f"peak MPS alloc {torch.mps.driver_allocated_memory()/1e9:.1f} GB")
for n_ex, ep in ((1500, 3), (3000, 3)):
    steps = (n_ex * ep) / BATCH
    print(f"  {n_ex} examples x {ep} epochs = {steps:.0f} steps ~ {steps*steady/60:.0f} min")
