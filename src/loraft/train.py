"""LoRA fine-tuning on the extraction task.

An explicit loop rather than `Trainer`: the one thing that must be right here is
that loss is computed on the *answer* tokens only. If prompt tokens are left
unmasked the model spends most of its gradient learning to reproduce a system
prompt it is always given anyway, and the run still looks fine because the loss
curve goes down.
"""
from __future__ import annotations

import csv
import json
import math
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config, task

IGNORE = -100


class ExtractionDataset(Dataset):
    def __init__(self, rows: list[dict], tok):
        self.rows, self.tok = rows, tok

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        prompt = self.tok.apply_chat_template(
            task.build_messages(r["received"], r["text"]),
            tokenize=False, add_generation_prompt=True)
        answer = task.target_json(r["gold"]) + self.tok.eos_token

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: config.MAX_LEN]
        # Mask the prompt: gradient only from tokens the model must produce.
        labels = ([IGNORE] * len(p_ids) + a_ids)[: config.MAX_LEN]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [IGNORE] * d)
        mask.append([1] * len(b["input_ids"]) + [0] * d)
    return (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask))


def main() -> None:
    torch.manual_seed(config.SEED)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(config.DATA / "train.jsonl")]
    ds = ExtractionDataset(rows, tok)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=config.DTYPE)
    model = get_peft_model(model, LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT, task_type="CAUSAL_LM",
        target_modules=config.TARGET_MODULES,
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={config.LORA_R} trainable {trainable:,} / {total:,} "
          f"= {100*trainable/total:.3f}%")
    model.to(config.DEVICE).train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.LR, weight_decay=0.0)
    total_steps = len(dl) * config.EPOCHS
    warmup = max(10, int(0.03 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))  # cosine decay to zero

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    log = open(config.REPORTS / "train_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "epoch", "loss", "lr", "seconds"])

    step, t0 = 0, time.time()
    for epoch in range(config.EPOCHS):
        for ids, labels, mask in dl:
            for g in opt.param_groups:
                g["lr"] = config.LR * lr_at(step)
            out = model(input_ids=ids.to(config.DEVICE),
                        attention_mask=mask.to(config.DEVICE),
                        labels=labels.to(config.DEVICE))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                w.writerow([step, epoch, round(out.loss.item(), 4),
                            round(opt.param_groups[0]["lr"], 8), round(el, 1)])
                log.flush()
                eta = (total_steps - step) * el / step / 60
                print(f"  step {step}/{total_steps} ep{epoch} "
                      f"loss {out.loss.item():.4f} eta {eta:.0f}m", end="\r")
    print()
    log.close()

    config.ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.ADAPTER_DIR)
    size = sum(f.stat().st_size for f in config.ADAPTER_DIR.rglob("*") if f.is_file())
    print(f"saved adapter -> {config.ADAPTER_DIR} ({size/1e6:.1f} MB, "
          f"{time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
