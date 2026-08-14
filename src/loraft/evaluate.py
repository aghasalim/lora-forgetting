"""Run a model (base or adapted) against a dataset and score it.

The same function scores the base and the fine-tuned model on identical
prompts, because a before/after where the prompt changed measures prompt
engineering rather than fine-tuning.
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config, data, task


def load(adapter: str | None = None):
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    # Decoder-only batched generation needs left padding, otherwise the model
    # continues from pad tokens and the outputs are quietly garbage.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=config.DTYPE)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # fold LoRA in; faster inference
    return tok, model.to(config.DEVICE).eval()


@torch.no_grad()
def generate(tok, model, rows: list[dict], batch_size: int = 8) -> list[str]:
    outs = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(
                task.build_messages(r["received"], r["text"]),
                tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=config.MAX_LEN).to(config.DEVICE)
        gen = model.generate(
            **enc, max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,  # greedy: one right answer, so sampling only adds variance
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
        print(f"  {min(i+batch_size, len(rows))}/{len(rows)}", end="\r")
    print()
    return outs


def run(rows: list[dict], label: str, tok, model) -> dict:
    t0 = time.time()
    raws = generate(tok, model, rows)
    scored = [task.score_one(raw, r["gold"]) for raw, r in zip(raws, rows)]
    agg = task.aggregate(scored)
    agg["label"] = label
    agg["seconds"] = round(time.time() - t0, 1)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS / f"preds_{label}.jsonl", "w", encoding="utf-8") as f:
        for r, raw, s in zip(rows, raws, scored):
            f.write(json.dumps({
                "id": r.get("id"), "kind": r.get("kind"), "received": r["received"],
                "text": r["text"], "gold": r["gold"], "raw": raw, "score": s,
            }, ensure_ascii=False) + "\n")
    return agg


def by_kind(label: str) -> dict:
    """Accuracy broken out by the benchmark's difficulty categories."""
    rows = [json.loads(l) for l in open(config.REPORTS / f"preds_{label}.jsonl")]
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r.get("kind") or "n/a", []).append(r["score"]["all_correct"])
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(out.items())}


def main() -> None:
    import sys

    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    label = sys.argv[2] if len(sys.argv) > 2 else ("tuned" if adapter else "base")

    tok, model = load(adapter)
    bench = data.load_benchmark()
    held = [json.loads(l) for l in open(config.DATA / "heldout_synthetic.jsonl")]

    print(f"=== {label}: hand-written benchmark (n={len(bench)}) ===")
    a = run(bench, label, tok, model)
    print(json.dumps(a))
    print("by kind:", json.dumps(by_kind(label)))

    print(f"\n=== {label}: held-out synthetic (n={len(held)}) ===")
    b = run(held, f"{label}_synth", tok, model)
    print(json.dumps(b))

    with open(config.REPORTS / f"summary_{label}.json", "w") as f:
        json.dump({"benchmark": a, "synthetic": b, "by_kind": by_kind(label)}, f, indent=2)


if __name__ == "__main__":
    main()
