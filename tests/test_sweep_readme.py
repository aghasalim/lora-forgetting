"""The seed, rank, epoch and target sweeps in README section 3 are recomputed
from the raw summaries in reports/sweep/ (RTX 3090) and reports/sweep_a5000/.
Editing a number by hand fails here."""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "reports" / "sweep"
A = ROOT / "reports" / "sweep_a5000"
FIRST = ["r16_s42", "r16_s1", "r16_s2", "r8_s42", "r32_s42"]
SECOND = {
    "attention, 3 epochs": ["r16_s1", "r16_s2", "r16_s3", "r16_s4"],
    "attention, 1 epoch": ["ep1_s42", "ep1_s1", "ep1_s2"],
    "attention and MLP, 3 epochs": ["mlp_s42", "mlp_s1", "mlp_s2"],
}


def summary(root, run, label="tuned"):
    return json.loads((root / run / f"summary_{label}.json").read_text())


def acc(root, run):
    return summary(root, run)["benchmark"]["all_correct"]


def pct(x):
    return f"{100 * x:.1f}%"


def readme():
    return " ".join((ROOT / "README.md").read_text().split())


def test_first_batch_table():
    text = (ROOT / "README.md").read_text()
    labels = {"r16_s42": "r=16, seed 42", "r16_s1": "r=16, seed 1", "r16_s2": "r=16, seed 2",
              "r8_s42": "r=8, seed 42", "r32_s42": "r=32, seed 42"}
    base = summary(G, "base", "base")
    rows = [("base", base)] + [(labels[r], summary(G, r)) for r in FIRST]
    for label, s in rows:
        k = s["by_kind"]
        row = (f"| {label} | {pct(s['benchmark']['all_correct'])} | "
               f"{k['written_amount']} | {k['currency']} |")
        assert row in text, row


def test_second_batch_table():
    text = (ROOT / "README.md").read_text()
    for setting, runs in SECOND.items():
        xs = [acc(A, r) for r in runs]
        seeds = ", ".join(r.split("_s")[1] for r in runs)
        row = f"| {setting} | {seeds} | {', '.join(pct(x) for x in xs)} | {pct(st.mean(xs))} |"
        assert row in text, row


def test_prose():
    text = readme()
    r16 = [acc(G, r) for r in FIRST[:3]]
    base = summary(G, "base", "base")
    every = [(G, r) for r in FIRST] + [(A, r) for runs in SECOND.values() for r in runs]
    worse_w = sum(summary(*x)["by_kind"]["written_amount"] < base["by_kind"]["written_amount"] for x in every)
    worse_c = sum(summary(*x)["by_kind"]["currency"] < base["by_kind"]["currency"] for x in every)
    attn = [(G, r) for r in FIRST[:3]] + [(A, r) for r in SECOND["attention, 3 epochs"]]
    worse_attn = sum(summary(*x)["by_kind"]["written_amount"] < 1.0 for x in attn)
    mlp = [acc(A, r) for r in SECOND["attention and MLP, 3 epochs"]]

    def secs(root, run):
        return float(list(csv.DictReader((root / run / "train_log.csv").open()))[-1]["seconds"])

    one = round(st.mean(secs(A, r) for r in SECOND["attention, 1 epoch"]))
    three = round(st.mean(secs(A, r) for r in SECOND["attention, 3 epochs"]))
    for want in [
        f"mean {pct(st.mean(r16))} and sd {100 * st.stdev(r16):.1f} points",
        f"the worst being {100 * (min(r16) - base['benchmark']['all_correct']):.1f} points",
        f"worse in {worse_w} of {len(every)} runs, `currency` in {worse_c} of {len(every)}",
        f"worse than base in {worse_w} and currency in {worse_c}",
        f"worse in {worse_attn} of {len(attn)}",
        f"between {pct(min(r16))} and {pct(max(r16))}",
        f"no seed below {pct(min(mlp))}",
        f"in {one} s of training against {three} s",
        f"Seed 1 gives {pct(acc(G, 'r16_s1'))} on the 3090 and {pct(acc(A, 'r16_s1'))}",
        f"seed 2 gives {pct(acc(G, 'r16_s2'))} and {pct(acc(A, 'r16_s2'))}",
    ]:
        assert want in text, want


def test_forgetting_reruns():
    text = readme()
    f = {r: json.loads((A / r / f"forgetting_{'base' if r == 'base' else 'tuned'}.json").read_text())
         for r in ("base", "r16_s1", "r16_s2")}
    ll = f"{pct(f['r16_s1']['arc_loglikelihood'])} and {pct(f['r16_s2']['arc_loglikelihood'])} against {pct(f['base']['arc_loglikelihood'])}"
    gen = f"{pct(f['r16_s1']['arc_generative'])} and {pct(f['r16_s2']['arc_generative'])} against {pct(f['base']['arc_generative'])}"
    assert ll in text, ll
    assert gen in text, gen
