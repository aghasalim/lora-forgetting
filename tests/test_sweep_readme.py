"""The seed and rank sweep in README section 3 is recomputed from the raw
summaries in reports/sweep/. Editing a number by hand fails here."""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "reports" / "sweep"
RUNS = ["r16_s42", "r16_s1", "r16_s2", "r8_s42", "r32_s42"]


def summary(run):
    name = "summary_base.json" if run == "base" else "summary_tuned.json"
    return json.loads((SWEEP / run / name).read_text())


def pct(x):
    return f"{100 * x:.1f}%"


def test_sweep_table_matches_the_summaries():
    readme = (ROOT / "README.md").read_text()
    labels = {"base": "base", "r16_s42": "r=16, seed 42", "r16_s1": "r=16, seed 1",
              "r16_s2": "r=16, seed 2", "r8_s42": "r=8, seed 42", "r32_s42": "r=32, seed 42"}
    for run, label in labels.items():
        s = summary(run)
        k = s["by_kind"]
        row = (f"| {label} | {pct(s['benchmark']['all_correct'])} | "
               f"{k['written_amount']} | {k['currency']} |")
        assert row in readme, row


def test_sweep_prose_matches_the_summaries():
    readme = " ".join((ROOT / "README.md").read_text().split())
    base = summary("base")["by_kind"]
    r16 = [summary(r)["benchmark"]["all_correct"] for r in RUNS[:3]]
    worse_w = sum(summary(r)["by_kind"]["written_amount"] < base["written_amount"] for r in RUNS)
    worse_c = sum(summary(r)["by_kind"]["currency"] < base["currency"] for r in RUNS)
    gap = min(r16) - summary("base")["benchmark"]["all_correct"]
    for want in [
        f"mean {pct(st.mean(r16))} and sd {100 * st.stdev(r16):.1f} points",
        f"the worst being {100 * gap:.1f} points",
        f"worse in every one of {len(RUNS)} runs" if worse_w == len(RUNS) else "unreachable",
        f"`currency` in {worse_c} of {len(RUNS)}",
        f"between {pct(min(r16))} and {pct(max(r16))}",
    ]:
        assert want in readme, want
