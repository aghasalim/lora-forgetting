"""Prompt construction and scoring for the extraction task.

Scoring is deliberately strict about structure and lenient about surface form:
a model that writes "PRET" instead of "Pret" has extracted the vendor, while one
that emits prose around its JSON has failed the task it was asked to do. Getting
that boundary wrong in either direction makes the headline number meaningless.
"""
from __future__ import annotations

import json
import re
import unicodedata

from . import config


def build_messages(received: str, text: str) -> list[dict]:
    """Chat messages for one example.

    `received` is in the prompt because without it "yesterday" has no answer,
    and a task whose gold label is unknowable cannot be scored.
    """
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Message received on {received}.\n\n{text}"},
    ]


def target_json(gold: dict) -> str:
    """Canonical target string, key order fixed so training is consistent."""
    return json.dumps({k: gold.get(k) for k in config.FIELDS}, ensure_ascii=False)


# --- parsing --------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Tolerates markdown fences and leading prose, because whether the model wraps
    its answer is a formatting habit rather than an extraction failure. Does NOT
    tolerate missing or extra keys, that is a schema failure and is scored as
    one.
    """
    if not raw:
        return None
    m = _FENCE.search(raw)
    candidate = m.group(1) if m else raw
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(candidate[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(candidate[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def schema_ok(obj: dict | None) -> bool:
    return obj is not None and set(obj.keys()) == set(config.FIELDS)


# --- field normalisation --------------------------------------------------

_PUNCT = re.compile(r"[^\w\s.&'-]")


def _norm_text(v) -> str | None:
    if v is None:
        return None
    s = unicodedata.normalize("NFKD", str(v)).strip().lower()
    s = _PUNCT.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s or None


def _norm_amount(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = re.sub(r"[^\d.,-]", "", str(v))
    if not s:
        return None
    # "1.204,55" (European) vs "1,204.55" (Anglo): whichever separator comes
    # last is the decimal point.
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) == 2 else s.replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _norm_currency(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().upper()
    return s if re.fullmatch(r"[A-Z]{3}", s) else None


def _norm_date(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else None


def _norm_category(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    return s if s in config.CATEGORIES else None


NORMALISERS = {
    "vendor": _norm_text, "amount": _norm_amount, "currency": _norm_currency,
    "date": _norm_date, "category": _norm_category,
}


def score_one(raw: str, gold: dict) -> dict:
    """Score a single response. Returns per-field correctness plus structure flags."""
    obj = extract_json(raw)
    ok = schema_ok(obj)
    out = {"json_parsed": obj is not None, "schema_ok": ok}
    for f in config.FIELDS:
        pred = NORMALISERS[f](obj.get(f)) if ok else None
        want = NORMALISERS[f](gold.get(f))
        # A schema failure scores every field wrong, including the null ones --
        # otherwise a model that emits nothing scores well on sparse examples.
        out[f] = bool(ok and pred == want)
    out["all_correct"] = ok and all(out[f] for f in config.FIELDS)
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    keys = ["json_parsed", "schema_ok", *config.FIELDS, "all_correct"]
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys} | {"n": n}
