"""Tests for scoring. Every headline number in this repo is produced by
`score_one`, so a bug here silently rewrites the results rather than crashing.
"""
import pytest

from src.loraft import config, data, task

GOLD = {"vendor": "Pret", "amount": 12.4, "currency": "GBP",
        "date": "2025-01-03", "category": "meals"}


def _resp(**over):
    import json
    return json.dumps({**GOLD, **over})


def test_perfect_response_scores_all_correct():
    assert task.score_one(_resp(), GOLD)["all_correct"]


def test_surface_form_is_forgiven():
    """Casing and stray punctuation are not extraction failures."""
    assert task.score_one(_resp(vendor="  PRET. ", currency="gbp"), GOLD)["all_correct"]


def test_markdown_fence_is_forgiven():
    assert task.score_one(f"```json\n{_resp()}\n```", GOLD)["all_correct"]


def test_prose_around_json_is_forgiven():
    assert task.score_one(f"Sure! Here you go:\n{_resp()}\nHope that helps.",
                          GOLD)["all_correct"]


def test_missing_key_fails_schema_and_every_field():
    import json
    bad = json.dumps({k: v for k, v in GOLD.items() if k != "category"})
    s = task.score_one(bad, GOLD)
    assert not s["schema_ok"]
    # A response missing a key must not score points on the keys it did emit,
    # otherwise emitting fewer fields is a scoring strategy.
    assert not any(s[f] for f in config.FIELDS)


def test_extra_key_fails_schema():
    import json
    assert not task.score_one(json.dumps({**GOLD, "notes": "x"}), GOLD)["schema_ok"]


def test_unparseable_output_scores_zero():
    s = task.score_one("I'm not sure what you mean.", GOLD)
    assert not s["json_parsed"] and not s["all_correct"]


@pytest.mark.parametrize("written,expected", [
    ("12.40", 12.4), ("12,40", 12.4), ("1,204.55", 1204.55),
    ("1.204,55", 1204.55), ("£12.40", 12.4), (12.4, 12.4), ("18000", 18000.0),
])
def test_amount_normalisation(written, expected):
    """European and Anglo separators both appear in the benchmark."""
    assert task.NORMALISERS["amount"](written) == expected


def test_invalid_category_is_not_credited():
    """A category outside the closed set is wrong, not merely unusual."""
    assert not task.score_one(_resp(category="food"), GOLD)["category"]


def test_null_is_scored_not_ignored():
    gold = {**GOLD, "vendor": None}
    assert task.score_one(_resp(vendor=None), gold)["vendor"]
    assert not task.score_one(_resp(vendor="Pret"), gold)["vendor"]


def test_bad_date_format_rejected():
    assert not task.score_one(_resp(date="3 January 2025"), GOLD)["date"]


def test_benchmark_is_wellformed():
    rows = data.load_benchmark()
    assert len(rows) >= 40, "brief asks for 30-50 hand-written examples"
    for r in rows:
        assert set(r["gold"].keys()) == set(config.FIELDS), r["id"]
        assert task.NORMALISERS["date"](r["received"]), r["id"]
        cat = r["gold"]["category"]
        assert cat is None or cat in config.CATEGORIES, r["id"]
        if r["gold"]["date"]:
            assert task.NORMALISERS["date"](r["gold"]["date"]), r["id"]


def test_train_and_benchmark_vendors_stay_disjoint():
    """The headline number is only meaningful if the model cannot have
    memorised the vendor list."""
    rows = data.load_benchmark()
    train_v = {v for vs in data.VENDORS.values() for v in vs}
    bench_v = {r["gold"]["vendor"] for r in rows if r["gold"]["vendor"]}
    assert not (train_v & bench_v)


def test_generator_produces_null_examples():
    rows = data.generate(n=200)
    n_null = sum(1 for r in rows if all(r["gold"][f] is None for f in config.FIELDS))
    assert n_null > 0, "without all-null training rows the model never learns to abstain"
