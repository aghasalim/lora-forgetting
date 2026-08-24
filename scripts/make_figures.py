"""Draw the README figures from reports/*.json.

Reads the saved evaluation only -- no model, no GPU, no inference.

    python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FIELDS = ["json_parsed", "schema_ok", "vendor", "amount", "currency", "date",
          "category", "all_correct"]
BASE, TUNED = "#bdbdbd", "#2166ac"


def load(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def task_gain(out: Path) -> Path:
    """Show what the adapter buys on the task it was trained for."""
    base, tuned = load("summary_base"), load("summary_tuned")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, split in zip(axes, ["benchmark", "synthetic"], strict=True):
        positions = np.arange(len(FIELDS))
        ax.barh(positions - 0.2, [base[split][f] * 100 for f in FIELDS], 0.4,
                label="base", color=BASE, edgecolor="0.3", lw=0.4)
        ax.barh(positions + 0.2, [tuned[split][f] * 100 for f in FIELDS], 0.4,
                label="LoRA tuned", color=TUNED, edgecolor="0.3", lw=0.4)
        ax.set_yticks(positions)
        ax.set_yticklabels(FIELDS, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("% correct")
        ax.set_title(f"{split}  (n={base[split]['n']})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Exact-match on every extracted field. `all_correct` requires every "
        "field right at once.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def by_kind(out: Path) -> Path:
    """Show the per-slice change that the aggregate hides.

    The headline gain is real, and two slices still get worse: ``written_amount``
    drops from 1.00 to 0.60 and ``currency`` from 0.80 to 0.60. Two more are flat.
    The adapter is trading accuracy between kinds rather than lifting all of them.
    """
    base, tuned = load("summary_base")["by_kind"], load("summary_tuned")["by_kind"]
    kinds = sorted(base, key=lambda k: tuned[k] - base[k])
    deltas = [(tuned[k] - base[k]) * 100 for k in kinds]
    positions = np.arange(len(kinds))

    figure, ax = plt.subplots(figsize=(10, 5.0))
    colours = ["#b2182b" if d < 0 else "#1a9850" for d in deltas]
    ax.barh(positions, deltas, color=colours, edgecolor="0.3", lw=0.4)
    ax.axvline(0, color="0.2", lw=1.1)
    ax.set_yticks(positions)
    ax.set_yticklabels(kinds, fontsize=9)
    ax.set_xlabel("change in accuracy after tuning (percentage points)")
    for index, kind in enumerate(kinds):
        ax.text(deltas[index] + (2 if deltas[index] >= 0 else -2), index,
                f"{base[kind]:.1f} -> {tuned[kind]:.1f}",
                va="center", ha="left" if deltas[index] >= 0 else "right",
                fontsize=7.5, color="0.35")
    ax.set_xlim(min(deltas) - 24, max(deltas) + 24)
    worse = sum(1 for d in deltas if d < 0)
    flat = sum(1 for d in deltas if d == 0)
    ax.set_title(
        f"{worse} slices get worse and {flat} are unchanged, "
        "while the aggregate improves.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def forgetting(out: Path) -> Path:
    """Check whether the adapter cost the model any general capability.

    This is the question the repository is named for, and the answer is no at this
    adapter size: ARC log-likelihood moves by 0.7 points and everything else is
    unchanged to the digit.
    """
    base, tuned = load("forgetting_base"), load("forgetting_tuned")
    metrics = ["arc_loglikelihood", "arc_generative", "arc_parse_rate", "open_ended"]
    positions = np.arange(len(metrics))

    figure, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(positions - 0.2, [base[m] * 100 for m in metrics], 0.4,
           label="base", color=BASE, edgecolor="0.3", lw=0.4)
    ax.bar(positions + 0.2, [tuned[m] * 100 for m in metrics], 0.4,
           label="LoRA tuned", color=TUNED, edgecolor="0.3", lw=0.4)
    for index, metric in enumerate(metrics):
        delta = (tuned[metric] - base[metric]) * 100
        ax.text(index, max(base[metric], tuned[metric]) * 100 + 2,
                f"{delta:+.1f}", ha="center", fontsize=9,
                color="#b2182b" if delta < -0.5 else "0.3")
    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylabel("%")
    ax.set_ylim(0, 112)
    ax.set_title(
        f"General capability before and after, n={base['n_mc']} multiple-choice "
        "items.\nNo measurable forgetting at this adapter size.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def arc_protocol(out: Path) -> Path:
    """Score the same benchmark two ways on the same model.

    Log-likelihood ranking and free generation disagree by 16.7 points on the
    identical items. Which protocol a paper used is therefore part of its number.
    """
    base, tuned = load("forgetting_base"), load("forgetting_tuned")
    labels = ["base", "LoRA tuned"]
    positions = np.arange(len(labels))

    figure, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(positions - 0.2,
           [base["arc_loglikelihood"] * 100, tuned["arc_loglikelihood"] * 100], 0.4,
           label="log-likelihood ranking", color="#9ecae1", edgecolor="0.3", lw=0.4)
    ax.bar(positions + 0.2,
           [base["arc_generative"] * 100, tuned["arc_generative"] * 100], 0.4,
           label="free generation", color="#2166ac", edgecolor="0.3", lw=0.4)
    gap = (base["arc_generative"] - base["arc_loglikelihood"]) * 100
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ARC accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Same items, same model, two scoring protocols: {gap:.1f} points apart.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def training(out: Path) -> Path:
    """Plot the training loss, for the record."""
    with (REPORTS / "train_log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]

    figure, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(steps, loss, color="#2166ac", lw=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(
        f"LoRA training, {len(rows)} logged steps, "
        f"{float(rows[-1]['seconds']) / 60:.0f} minutes total.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        task_gain(FIGURES / "task-gain.png"),
        by_kind(FIGURES / "by-kind.png"),
        forgetting(FIGURES / "forgetting.png"),
        arc_protocol(FIGURES / "arc-protocol.png"),
        training(FIGURES / "training.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
