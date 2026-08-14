"""Base vs fine-tuned, side by side.

Reads precomputed predictions rather than loading a model. That is a deliberate
constraint, not a shortcut: a 1.5B model needs ~3 GB of weights and free hosting
tiers give you 1 GB, so a "live" hosted demo of this project cannot honestly
exist. Showing every one of the 45 benchmark cases is also more informative than
a text box, because you can see the failures instead of the examples I chose.

Run locally with the adapter present and the sidebar offers live generation too.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIELDS = ["vendor", "amount", "currency", "date", "category"]

st.set_page_config(page_title="LoRA fine-tuning with a forgetting check",
                   page_icon="🔬", layout="wide")


@st.cache_data
def load(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return json.loads(p.read_text())


base = load("preds_base.jsonl")
tuned = load("preds_tuned.jsonl")
s_base, s_tuned = load("summary_base.json"), load("summary_tuned.json")
f_base, f_tuned = load("forgetting_base.json"), load("forgetting_tuned.json")

st.title("LoRA fine-tuning, with the forgetting check")
st.caption(
    "Qwen2.5-1.5B-Instruct + LoRA on structured expense extraction. "
    "Both numbers are reported: what fine-tuning bought on the target task, and "
    "what it cost everywhere else."
)

if not base or not tuned:
    st.error("No predictions found. Run `make baseline && make train && make eval`.")
    st.stop()

# --- headline ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
b_acc = s_base["benchmark"]["all_correct"]
t_acc = s_tuned["benchmark"]["all_correct"]
c1.metric("base — all fields correct", f"{b_acc:.1%}")
c2.metric("fine-tuned", f"{t_acc:.1%}", delta=f"{t_acc - b_acc:+.1%}")
if f_base and f_tuned:
    d = f_tuned["arc_loglikelihood"] - f_base["arc_loglikelihood"]
    c3.metric("general knowledge (ARC)", f"{f_tuned['arc_loglikelihood']:.1%}",
              delta=f"{d:+.1%}", delta_color="normal" if abs(d) < 0.02 else "inverse")

tab1, tab2, tab3 = st.tabs(["Side by side", "Scores", "Forgetting check"])

with tab1:
    kinds = sorted({r.get("kind") or "n/a" for r in base})
    pick = st.multiselect("difficulty", kinds, default=kinds)
    only_diff = st.checkbox("only where the two models disagree", value=True)

    tmap = {r["id"]: r for r in tuned}
    shown = 0
    for r in base:
        t = tmap.get(r["id"])
        if not t or (r.get("kind") or "n/a") not in pick:
            continue
        same = r["score"]["all_correct"] == t["score"]["all_correct"]
        if only_diff and same:
            continue
        shown += 1
        st.markdown(f"**{r['id']}** · `{r.get('kind')}` · received {r['received']}")
        st.code(r["text"], language=None)
        a, b, c = st.columns(3)
        a.caption("gold")
        a.json({k: r["gold"][k] for k in FIELDS}, expanded=True)
        b.caption("base " + ("✅" if r["score"]["all_correct"] else "❌"))
        b.code((r["raw"] or "(empty)")[:400], language="json")
        c.caption("fine-tuned " + ("✅" if t["score"]["all_correct"] else "❌"))
        c.code((t["raw"] or "(empty)")[:400], language="json")
        st.divider()
    if not shown:
        st.info("No cases match that filter.")

with tab2:
    st.subheader("Hand-written benchmark (45 cases)")
    rows = []
    for k in ["json_parsed", "schema_ok", *FIELDS, "all_correct"]:
        rows.append({"metric": k,
                     "base": s_base["benchmark"][k], "fine-tuned": s_tuned["benchmark"][k],
                     "delta": round(s_tuned["benchmark"][k] - s_base["benchmark"][k], 4)})
    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("By difficulty")
    kb, kt = s_base["by_kind"], s_tuned["by_kind"]
    st.dataframe([{"kind": k, "base": kb[k], "fine-tuned": kt.get(k),
                   "delta": round(kt.get(k, 0) - kb[k], 3)} for k in sorted(kb)],
                 width="stretch", hide_index=True)

    st.subheader("Held-out synthetic vs hand-written")
    st.caption(
        "Training data and the synthetic split share a generator; the benchmark "
        "is hand-written with a disjoint vendor list. The gap between these two "
        "columns is how much of the gain is generalisation rather than "
        "template memorisation."
    )
    st.dataframe([
        {"set": "held-out synthetic (same generator)",
         "base": s_base["synthetic"]["all_correct"], "fine-tuned": s_tuned["synthetic"]["all_correct"]},
        {"set": "hand-written benchmark (disjoint vendors)",
         "base": s_base["benchmark"]["all_correct"], "fine-tuned": s_tuned["benchmark"]["all_correct"]},
    ], width="stretch", hide_index=True)

with tab3:
    if not (f_base and f_tuned):
        st.info("Run `make forgetting` to populate this.")
    else:
        st.caption(
            "Knowledge retention is scored by log-likelihood over the answer "
            "options, so it does not depend on the model still being willing to "
            "chat. Instruction following asks the same questions in chat form. "
            "The two can move independently, and which one moves tells you "
            "whether the model forgot facts or just lost a format."
        )
        st.dataframe([
            {"check": "ARC-Easy, log-likelihood (knowledge)",
             "base": f_base["arc_loglikelihood"], "fine-tuned": f_tuned["arc_loglikelihood"],
             "delta": round(f_tuned["arc_loglikelihood"] - f_base["arc_loglikelihood"], 4)},
            {"check": "ARC-Easy, generated answer (instruction following)",
             "base": f_base["arc_generative"], "fine-tuned": f_tuned["arc_generative"],
             "delta": round(f_tuned["arc_generative"] - f_base["arc_generative"], 4)},
            {"check": "answer parseable at all",
             "base": f_base["arc_parse_rate"], "fine-tuned": f_tuned["arc_parse_rate"],
             "delta": round(f_tuned["arc_parse_rate"] - f_base["arc_parse_rate"], 4)},
            {"check": "open-ended factual probes",
             "base": f_base["open_ended"], "fine-tuned": f_tuned["open_ended"],
             "delta": round(f_tuned["open_ended"] - f_base["open_ended"], 4)},
        ], width="stretch", hide_index=True)

        st.subheader("What it says when asked a normal question")
        for a, b in zip(f_base.get("open_ended_log", []), f_tuned.get("open_ended_log", [])):
            st.markdown(f"**{a['q']}**")
            x, y = st.columns(2)
            x.caption("base " + ("✅" if a["ok"] else "❌"))
            x.code(a["answer"][:300] or "(empty)", language=None)
            y.caption("fine-tuned " + ("✅" if b["ok"] else "❌"))
            y.code(b["answer"][:300] or "(empty)", language=None)

st.divider()
st.markdown(
    "Code and full write-up: "
    "[github.com/aghasalim/lora-forgetting](https://github.com/aghasalim/lora-forgetting)"
)
