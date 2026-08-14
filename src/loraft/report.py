"""Render the measured JSON summaries into RESULTS.md.

Generated rather than hand-written so the tables cannot drift away from the
numbers actually produced by the last run.
"""
from __future__ import annotations

import json

from . import config

FIELDS = ["json_parsed", "schema_ok", "vendor", "amount", "currency", "date",
          "category", "all_correct"]


def _load(name):
    p = config.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    sb, st_ = _load("summary_base.json"), _load("summary_tuned.json")
    fb, ft = _load("forgetting_base.json"), _load("forgetting_tuned.json")
    if not (sb and st_):
        raise SystemExit("run `make baseline` and `make eval` first")

    L = ["# Results", "",
         f"Base model `{config.BASE_MODEL}`, LoRA r={config.LORA_R} "
         f"alpha={config.LORA_ALPHA} on {config.TARGET_MODULES}, "
         f"{config.EPOCHS} epochs, lr={config.LR}, batch {config.BATCH_SIZE}.", "",
         "## Target task", "",
         "### Hand-written benchmark (45 cases, vendors disjoint from training)", "",
         "| metric | base | fine-tuned | delta |", "|---|---|---|---|"]
    for k in FIELDS:
        b, t = sb["benchmark"][k], st_["benchmark"][k]
        L.append(f"| {k} | {b:.1%} | {t:.1%} | {t-b:+.1%} |")

    L += ["", "### By difficulty", "", "| kind | base | fine-tuned | delta |",
          "|---|---|---|---|"]
    for k in sorted(sb["by_kind"]):
        b, t = sb["by_kind"][k], st_["by_kind"].get(k, 0)
        L.append(f"| {k} | {b:.1%} | {t:.1%} | {t-b:+.1%} |")

    L += ["", "### Generalisation gap", "",
          "| set | base | fine-tuned |", "|---|---|---|",
          f"| held-out synthetic (same generator) | {sb['synthetic']['all_correct']:.1%} "
          f"| {st_['synthetic']['all_correct']:.1%} |",
          f"| hand-written benchmark (disjoint vendors) | {sb['benchmark']['all_correct']:.1%} "
          f"| {st_['benchmark']['all_correct']:.1%} |"]

    if fb and ft:
        L += ["", "## Catastrophic forgetting check", "",
              "| check | base | fine-tuned | delta |", "|---|---|---|---|"]
        for key, name in [("arc_loglikelihood", "ARC-Easy, log-likelihood (knowledge)"),
                          ("arc_generative", "ARC-Easy, generated (instruction following)"),
                          ("arc_parse_rate", "answer parseable at all"),
                          ("open_ended", "open-ended factual probes")]:
            L.append(f"| {name} | {fb[key]:.1%} | {ft[key]:.1%} | {ft[key]-fb[key]:+.1%} |")

    (config.REPORTS.parent / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"wrote {config.REPORTS.parent / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
