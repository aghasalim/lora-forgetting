"""Draw the README figures from reports/*.json and reports/train_log.csv.

Reads the saved evaluation only, no model, no GPU, no inference. Every number
drawn here comes from a committed file, so a figure cannot disagree with the
tables in RESULTS.md.

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
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FIELDS = ["json_parsed", "schema_ok", "vendor", "amount", "currency", "date",
          "category", "all_correct"]

# Grey is the untuned model and blue is the adapter, in every figure that
# compares them, so the reader learns the pair once.
BASE, TUNED, LINK = "#9e9e9e", PALETTE[0], "#d6d6d6"
WORSE, BETTER = PALETTE[1], PALETTE[2]
# The two ARC protocols are not a before and after, so they get their own pair
# rather than borrowing the grey and blue.
LOGLIK, GENERATE = PALETTE[4], PALETTE[3]


def load(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def train_log() -> tuple[list[int], list[float], list[float], list[int]]:
    with (REPORTS / "train_log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return ([int(r["step"]) for r in rows], [float(r["loss"]) for r in rows],
            [float(r["seconds"]) for r in rows], [int(r["epoch"]) for r in rows])


def task_gain(out: Path) -> Path:
    """What the adapter buys on the task it was trained for, field by field."""
    base, tuned = load("summary_base"), load("summary_tuned")
    positions = np.arange(len(FIELDS))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    heads = (
        ("benchmark", "Every field gets better on the hand-written benchmark",
         "45 messages whose vendors never appear in training, exact match per field"),
        ("synthetic", "The synthetic split nearly saturates, so it flatters the gain",
         "150 held-out messages from the generator that wrote the training set"),
    )
    for ax, (split, title, subtitle) in zip(axes, heads, strict=True):
        before = np.array([base[split][f] for f in FIELDS]) * 100
        after = np.array([tuned[split][f] for f in FIELDS]) * 100
        ax.hlines(positions, before, after, color=LINK, lw=3.4, zorder=1)
        ax.plot(before, positions, "o", ms=8, color=BASE, zorder=2, label="base")
        ax.plot(after, positions, "o", ms=8, color=TUNED, zorder=3, label="LoRA tuned")
        for y, (b, a) in enumerate(zip(before, after, strict=True)):
            ax.text(max(a, b) + 2.5, y, f"{a - b:+.1f}", va="center", ha="left",
                    fontsize=8.5, color="#5a5a5a")
        ax.set_yticks(positions)
        ax.set_yticklabels(FIELDS)
        ax.invert_yaxis()
        ax.set_xlim(18, 119)
        ax.set_xlabel("cases correct (%), with the change in percentage points")
        ax.grid(False, axis="y")
        titled(ax, title, subtitle)
    axes[0].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def by_kind(out: Path) -> Path:
    """The per-slice change that the aggregate hides.

    The headline gain is real, and two slices still get worse: ``written_amount``
    drops from 1.00 to 0.60 and ``currency`` from 0.80 to 0.60. Two more are flat.
    The adapter is trading accuracy between kinds rather than lifting all of them.
    """
    base, tuned = load("summary_base")["by_kind"], load("summary_tuned")["by_kind"]
    kinds = sorted(base, key=lambda k: tuned[k] - base[k])
    deltas = [(tuned[k] - base[k]) * 100 for k in kinds]
    positions = np.arange(len(kinds))

    figure, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.barh(positions, deltas, color=[WORSE if d < 0 else BETTER for d in deltas],
            height=0.62, zorder=2)
    ax.axvline(0, color="#555555", lw=1.1, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(kinds)
    ax.set_xlabel("change in accuracy after tuning (percentage points)")
    for index, kind in enumerate(kinds):
        offset = 3 if deltas[index] >= 0 else -3
        ax.text(deltas[index] + offset, index,
                f"{base[kind] * 100:.0f}% to {tuned[kind] * 100:.0f}%",
                va="center", ha="left" if deltas[index] >= 0 else "right",
                fontsize=8.5, color="#5a5a5a")
    ax.set_xlim(min(deltas) - 32, max(deltas) + 32)
    ax.grid(False, axis="y")
    worse = sum(1 for d in deltas if d < 0)
    flat = sum(1 for d in deltas if d == 0)
    titled(ax,
           f"{worse} slices get worse and {flat} stand still while the total improves",
           "each kind is 5 of the 45 benchmark cases, so one case is worth 20 points")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def forgetting(out: Path) -> Path:
    """Whether the adapter cost the model any general capability.

    This is the question the repository is named for, and the answer is no at this
    adapter size: ARC log-likelihood moves by 0.7 points and everything else is
    unchanged to the digit.
    """
    base, tuned = load("forgetting_base"), load("forgetting_tuned")
    metrics = ["arc_loglikelihood", "arc_generative", "arc_parse_rate", "open_ended"]
    labels = ["ARC-Easy\nlog-likelihood", "ARC-Easy\ngenerated answer",
              "answer parseable\nat all", "open-ended\nfactual probes"]
    positions = np.arange(len(metrics))

    figure, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(positions - 0.2, [base[m] * 100 for m in metrics], 0.4,
           label="base", color=BASE, zorder=2)
    ax.bar(positions + 0.2, [tuned[m] * 100 for m in metrics], 0.4,
           label="LoRA tuned", color=TUNED, zorder=2)
    for index, metric in enumerate(metrics):
        delta = (tuned[metric] - base[metric]) * 100
        ax.text(index, max(base[metric], tuned[metric]) * 100 + 2, f"{delta:+.1f}",
                ha="center", fontsize=9, color=WORSE if delta < -0.5 else "#5a5a5a")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("items answered correctly (%)")
    ax.set_ylim(0, 124)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(False, axis="x")
    ax.legend(loc="upper left")
    titled(ax, "The adapter costs no general capability I can measure",
           f"{base['n_mc']} ARC-Easy items scored two ways, plus "
           f"{len(base['open_ended_log'])} open-ended probes, same weights either side")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def arc_protocol(out: Path) -> Path:
    """The same benchmark scored two ways on the same model.

    Log-likelihood ranking and free generation disagree by 16.7 points on the
    identical items. Which protocol a paper used is therefore part of its number.
    """
    base, tuned = load("forgetting_base"), load("forgetting_tuned")
    labels = ["base", "LoRA tuned"]
    positions = np.arange(len(labels))
    loglik = [base["arc_loglikelihood"] * 100, tuned["arc_loglikelihood"] * 100]
    generate = [base["arc_generative"] * 100, tuned["arc_generative"] * 100]

    figure, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset, values, colour, label in (
            (-0.19, loglik, LOGLIK, "rank the options by log-likelihood"),
            (0.19, generate, GENERATE, "generate an answer and parse it")):
        ax.bar(positions + offset, values, 0.38, color=colour, label=label, zorder=2)
        for x, value in zip(positions + offset, values, strict=True):
            ax.text(x, value + 1.8, f"{value:.1f}%", ha="center", fontsize=9,
                    color="#5a5a5a")
    # The gap is the whole point, so it gets drawn, in the clear space beside
    # each pair rather than across the bars.
    for x, (low, high) in enumerate(zip(loglik, generate, strict=True)):
        ax.annotate("", xy=(x + 0.45, high), xytext=(x + 0.45, low), arrowprops={
            "arrowstyle": "<->", "color": "#555555", "lw": 1.0})
        ax.text(x + 0.5, (low + high) / 2, f"{high - low:.1f}\npoints",
                fontsize=9, color="#555555", va="center", ha="left")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, 1.9)
    ax.set_ylabel("ARC-Easy accuracy (%)")
    ax.set_ylim(0, 124)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(False, axis="x")
    ax.legend(loc="upper left")
    titled(ax, "The scoring protocol moves ARC further than the fine-tuning does",
           f"same {base['n_mc']} questions, same weights, "
           "two ways of asking for the answer")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def _converged(steps: list[int], loss: list[float]) -> int:
    """First logged step at which the loss reaches its floor of 0.0001."""
    return steps[next(i for i, value in enumerate(loss) if value <= 0.0001)]


def _draw_run(ax, steps, loss, epochs) -> None:
    """Axis furniture shared by the training figure and the training GIF."""
    for index in range(1, len(epochs)):
        if epochs[index] != epochs[index - 1]:
            ax.axvline(steps[index], color="#bbbbbb", ls=":", lw=1.1, zorder=1)
            ax.text(steps[index] + 8, 0.86, f"epoch {epochs[index] + 1}",
                    transform=ax.get_xaxis_transform(), fontsize=8.5,
                    color="#888888", va="top", ha="left")
    ax.set_xlim(-15, steps[-1] + 15)
    ax.set_ylim(-0.006, max(loss) * 1.12)
    ax.set_xlabel("training step (of 1014)")
    ax.set_ylabel("loss on answer tokens (nats per token)")


def training(out: Path) -> Path:
    """The training loss, and how much of the run happened after it flattened."""
    steps, loss, secs, epochs = train_log()
    converged = _converged(steps, loss)
    minutes = secs[-1] / 60
    after = (secs[-1] - secs[steps.index(converged)]) / 60

    figure, ax = plt.subplots(figsize=(9.5, 4.8))
    _draw_run(ax, steps, loss, epochs)
    ax.axvspan(converged, steps[-1] + 15, color="#f2f2f2", zorder=0)
    ax.plot(steps, loss, color=PALETTE[0], lw=1.7, zorder=3)
    ax.text(steps[-1], 0.55,
            f"{after:.0f} of the {minutes:.0f} minutes\nwere spent inside this shading",
            transform=ax.get_xaxis_transform(), fontsize=9.5, color="#5a5a5a",
            va="center", ha="right")
    titled(ax, f"The loss was down to 0.0001 "
               f"{secs[steps.index(converged)] / 60:.0f} minutes into a "
               f"{minutes:.0f} minute run",
           "3 epochs of LoRA on 4 attention projections, logged every 10 steps, "
           "shading is everything after that point")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def anim_training(out: Path, frames: int = 84, hold: int = 16, fps: int = 16) -> Path:
    """Replay the committed training log against the wall clock.

    The static figure plots loss against step. This one adds the column the
    static figure cannot show at the same time: the seconds each step cost, so
    the reader watches most of the run go by after the loss stopped moving.
    Nothing is sampled here, so the GIF is identical on every run.
    """
    steps, loss, secs, epochs = train_log()
    converged = _converged(steps, loss)
    cut_index = steps.index(converged)
    minutes = secs[-1] / 60
    after = (secs[-1] - secs[cut_index]) / 60

    figure, ax = plt.subplots(figsize=(7.6, 4.4))
    _draw_run(ax, steps, loss, epochs)
    titled(ax, f"Most of the {minutes:.0f} minutes came after the loss reached 0.0001",
           "the committed training log replayed against its own wall clock")

    span = ax.axvspan(converged, converged, color="#f2f2f2", zorder=0)
    line, = ax.plot([], [], color=PALETTE[0], lw=1.7, zorder=3)
    head, = ax.plot([], [], "o", ms=6.0, color=PALETTE[0], zorder=4)
    clock = ax.text(0.98, 0.99, "", transform=ax.transAxes, fontsize=10,
                    color="#5a5a5a", ha="right", va="top")
    note = ax.text(steps[-1], 0.55, "", transform=ax.get_xaxis_transform(),
                   fontsize=9.5, color="#5a5a5a", va="center", ha="right")

    def draw(i: int):
        k = min(i, frames - 1) * (len(steps) - 1) // (frames - 1) + 1
        line.set_data(steps[:k], loss[:k])
        head.set_data(steps[k - 1:k], loss[k - 1:k])
        clock.set_text(f"step {steps[k - 1]}    {secs[k - 1] / 60:.0f} min elapsed")
        if k > cut_index:
            span.set_width(steps[k - 1] - converged)
            note.set_text(f"{(secs[k - 1] - secs[cut_index]) / 60:.0f} minutes so far\n"
                          f"since the loss got here")
        if i >= frames:
            note.set_text(f"{after:.0f} of the {minutes:.0f} minutes\n"
                          f"were spent inside this shading")
        return [line, head, clock, note, span]

    anim = FuncAnimation(figure, draw, frames=frames + hold, interval=1000 // fps,
                         blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    _shrink_gif(out)
    return out


def _shrink_gif(path: Path) -> None:
    """Rewrite every frame onto one shared palette. Usually halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], loop=0,
                      duration=durations, optimize=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        task_gain(FIGURES / "task-gain.png"),
        by_kind(FIGURES / "by-kind.png"),
        forgetting(FIGURES / "forgetting.png"),
        arc_protocol(FIGURES / "arc-protocol.png"),
        training(FIGURES / "training.png"),
        anim_training(FIGURES / "training.gif"),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
