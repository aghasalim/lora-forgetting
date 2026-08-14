"""Catastrophic-forgetting check: did narrow fine-tuning break general ability?

Two measurements, because "forgetting" hides two different failures that call
for different fixes:

**Knowledge retention** -- scored by log-likelihood over the answer options. No
generation involved, so a model that has lost the *habit* of answering multiple
choice still scores normally if it still knows the answer.

**Instruction following** -- the same questions asked in chat, answer parsed
from what it generates. A model fine-tuned to emit nothing but JSON can retain
every fact and still fail here, by replying `{"vendor": null, ...}` to "what is
the capital of France".

If knowledge holds and generation collapses, the model has not forgotten
anything -- it has been over-specialised into one output format, and the fix is
data mixing rather than a smaller learning rate. Reporting only one of these
numbers would point at the wrong remedy.
"""
from __future__ import annotations

import json
import re

import torch

from . import config, evaluate

N_MC = int(__import__("os").getenv("N_MC", "150"))

# Open-ended probes, kept deliberately simple: anything the base model cannot do
# is useless as a forgetting signal.
OPEN_PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("What is 17 plus 25?", ["42"]),
    ("Name the largest planet in our solar system.", ["jupiter"]),
    ("What gas do plants absorb from the air for photosynthesis?",
     ["carbon dioxide", "co2"]),
    ("In which continent is Brazil located?", ["south america"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("What language is primarily spoken in Japan?", ["japanese"]),
    ("Translate 'good morning' into French.", ["bonjour"]),
    ("What is the boiling point of water in Celsius at sea level?", ["100"]),
    ("Who painted the Mona Lisa?", ["da vinci", "davinci", "leonardo"]),
]


def load_arc(n: int = N_MC) -> list[dict]:
    """ARC-Easy: grade-school science multiple choice. Ungated on the Hub."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rows = []
    for r in ds:
        choices = r["choices"]["text"]
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels or not 2 <= len(choices) <= 5:
            continue
        rows.append({"q": r["question"], "choices": choices, "labels": labels,
                     "answer": labels.index(r["answerKey"])})
        if len(rows) >= n:
            break
    return rows


@torch.no_grad()
def mc_loglikelihood(tok, model, rows: list[dict]) -> float:
    """Pick the option with the highest length-normalised log-likelihood.

    Length normalisation matters: without it the model simply prefers whichever
    option has fewest tokens, which measures verbosity rather than knowledge.
    """
    correct = 0
    for i, r in enumerate(rows):
        ctx = f"Question: {r['q']}\nAnswer:"
        ctx_ids = tok(ctx, return_tensors="pt")["input_ids"]
        scores = []
        for c in r["choices"]:
            full = tok(ctx + " " + c, return_tensors="pt")["input_ids"].to(config.DEVICE)
            logits = model(full).logits[0, :-1].float().log_softmax(-1)
            tgt = full[0, 1:]
            start = ctx_ids.shape[1] - 1
            lp = logits[start:].gather(-1, tgt[start:].unsqueeze(-1)).sum().item()
            scores.append(lp / max(1, full.shape[1] - ctx_ids.shape[1]))
        correct += int(int(torch.tensor(scores).argmax()) == r["answer"])
        print(f"  mc {i+1}/{len(rows)}", end="\r")
    print()
    return correct / len(rows)


@torch.no_grad()
def mc_generative(tok, model, rows: list[dict]) -> tuple[float, float]:
    """Ask in chat form and parse the letter. Returns (accuracy, parse rate)."""
    correct = parsed = 0
    for i, r in enumerate(rows):
        opts = "\n".join(f"{l}. {c}" for l, c in zip(r["labels"], r["choices"]))
        msgs = [{"role": "user", "content":
                 f"{r['q']}\n{opts}\nAnswer with the letter only."}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(config.DEVICE)
        out = tok.decode(model.generate(**enc, max_new_tokens=8, do_sample=False,
                                        pad_token_id=tok.pad_token_id)[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        m = re.search(r"\b([A-E1-4])\b", out.upper())
        if m:
            parsed += 1
            if m.group(1) in r["labels"] and r["labels"].index(m.group(1)) == r["answer"]:
                correct += 1
        print(f"  gen {i+1}/{len(rows)}", end="\r")
    print()
    return correct / len(rows), parsed / len(rows)


@torch.no_grad()
def open_ended(tok, model) -> tuple[float, list[dict]]:
    hits, log = 0, []
    for q, accept in OPEN_PROBES:
        msgs = [{"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(config.DEVICE)
        out = tok.decode(model.generate(**enc, max_new_tokens=40, do_sample=False,
                                        pad_token_id=tok.pad_token_id)[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        ok = any(a in out.lower() for a in accept)
        hits += ok
        log.append({"q": q, "answer": out, "ok": ok})
    return hits / len(OPEN_PROBES), log


def main() -> None:
    import sys

    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    label = sys.argv[2] if len(sys.argv) > 2 else ("tuned" if adapter else "base")

    tok, model = evaluate.load(adapter)
    rows = load_arc()
    print(f"=== forgetting check: {label} (ARC-Easy n={len(rows)}) ===")

    ll = mc_loglikelihood(tok, model, rows)
    gen, parse = mc_generative(tok, model, rows)
    oe, oe_log = open_ended(tok, model)

    res = {"label": label, "n_mc": len(rows),
           "arc_loglikelihood": round(ll, 4),
           "arc_generative": round(gen, 4),
           "arc_parse_rate": round(parse, 4),
           "open_ended": round(oe, 4)}
    print(json.dumps(res, indent=2))

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS / f"forgetting_{label}.json", "w") as f:
        json.dump({**res, "open_ended_log": oe_log}, f, indent=2)


if __name__ == "__main__":
    main()
