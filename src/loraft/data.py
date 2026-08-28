"""Synthetic training data for the extraction task.

Two deliberate separations from the held-out benchmark, both there to stop the
headline number measuring the wrong thing:

1. **Disjoint vendors.** No company appearing in `eval/eval_set.jsonl` appears
   here. A model that learns "Figma means software" would score well without
   learning to extract anything.
2. **Disjoint phrasings.** The templates below are structurally unlike the
   hand-written benchmark messages. Generating both from one generator would
   produce a benchmark that measures template memorisation, and it would look
   excellent.

The benchmark is hand-written and messier than anything here. The gap between
in-distribution and hand-written accuracy is therefore a real quantity, and the
report shows both rather than only the flattering one.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta

from . import config, task

# Deliberately disjoint from every vendor in the hand-written benchmark.
VENDORS = {
    "travel": ["Ryanair", "FlixBus", "DB Bahn", "Hertz", "NS", "Lyft", "easyJet",
               "Premier Inn", "Booking.com", "Gett", "Deutsche Bahn", "TfL"],
    "meals": ["Nando's", "Wagamama", "Leon", "Costa", "Paul", "Itsu", "Greggs",
              "Caffe Nero", "Subway", "Honest Burgers", "Pizza Pilgrims"],
    "software": ["GitHub", "Linear", "Vercel", "Datadog", "Sentry", "Miro",
                 "Zoom", "Atlassian", "Cloudflare", "Postman", "1Password"],
    "hardware": ["Dell", "Belkin", "Sandisk", "Razer", "Corsair", "Elgato",
                 "Kensington", "Seagate", "Brother", "Epson"],
    "office": ["Staples", "Office Depot", "Muji", "Flying Tiger", "Lyreco",
               "Nespresso", "Tesco Express"],
    "other": ["Eventbrite", "WWF", "Oxfam", "Meetup", "Red Cross", "Kickstarter"],
}
CURRENCIES = ["EUR", "GBP", "USD", "CHF", "SEK", "CAD", "JPY", "DKK", "NOK", "PLN"]
SYMBOL = {"EUR": "€", "GBP": "£", "USD": "$", "JPY": "¥"}
WORD = {"EUR": "euros", "GBP": "pounds", "USD": "dollars", "CHF": "francs",
        "SEK": "kronor", "CAD": "Canadian dollars", "JPY": "yen",
        "DKK": "kroner", "NOK": "kroner", "PLN": "zloty"}

# Structurally unlike the benchmark phrasings, which lead with the purchase.
TEMPLATES = [
    "Expense claim: {vendor}, {amt}, {d}.",
    "{vendor}, {amt}, {d}",
    "Please reimburse {amt} paid to {vendor} on {d}.",
    "Card ending 4417 charged {amt} by {vendor}, {d}.",
    "{d}: {vendor} {amt}",
    "Receipt attached. {vendor}. {amt}. Date of purchase {d}.",
    "Submitting {amt} for {vendor} ({d}).",
    "Company card, {vendor}, {amt} on {d}",
]
NON_EXPENSE = [
    "Standup is cancelled tomorrow, I'll post notes instead.",
    "Does anyone have the link to the onboarding doc?",
    "Deploy finished, {n} tests green, no rollbacks needed.",
    "Welcome to the team! Your laptop should arrive next week.",
    "The office wifi is down on floor {n}, IT has been notified.",
    "Sprint {n} retro moved to Friday at 11.",
    "Can someone review my PR when they get a minute?",
    "Reminder that the office closes early on the {n}th.",
    "We're at {n}% of the quarterly target with three weeks left.",
    "Parking barrier is broken again, use the side entrance.",
]


def _fmt_amount(amount: float, cur: str, rng: random.Random) -> str:
    style = rng.choice(["symbol", "code_before", "code_after", "word", "comma"])
    whole = amount == int(amount)
    n = f"{int(amount)}" if whole and rng.random() < 0.4 else f"{amount:.2f}"
    if style == "symbol" and cur in SYMBOL:
        return f"{SYMBOL[cur]}{n}"
    if style == "code_before":
        return f"{cur} {n}"
    if style == "word":
        return f"{n} {WORD[cur]}"
    if style == "comma":
        return f"{n.replace('.', ',')} {cur}"
    return f"{n} {cur}"


def _fmt_date(d: date, received: date, rng: random.Random) -> str:
    delta = (received - d).days
    styles = ["iso", "slash", "long", "short"]
    if delta == 0:
        styles += ["today"] * 3
    elif delta == 1:
        styles += ["yesterday"] * 3
    elif 2 <= delta <= 6:
        styles += ["ndays"] * 3
    s = rng.choice(styles)
    if s == "today":
        return "today"
    if s == "yesterday":
        return "yesterday"
    if s == "ndays":
        return f"{delta} days ago"
    if s == "iso":
        return d.isoformat()
    if s == "slash":
        return d.strftime("%d/%m/%Y")
    if s == "long":
        return d.strftime("%-d %B %Y")
    return d.strftime("%-d %b %Y")


def generate(n: int | None = None, seed: int = config.SEED) -> list[dict]:
    n = n or config.N_TRAIN
    rng = random.Random(seed)
    rows = []
    base = date(2024, 1, 1)

    while len(rows) < n:
        received = base + timedelta(days=rng.randint(0, 700))

        # ~12% non-expense, so the model learns that all-null is a real answer
        # rather than something to avoid.
        if rng.random() < 0.12:
            text = rng.choice(NON_EXPENSE).format(n=rng.randint(2, 40))
            gold = {f: None for f in config.FIELDS}
            rows.append({"received": received.isoformat(), "text": text, "gold": gold})
            continue

        cat = rng.choice(config.CATEGORIES)
        vendor = rng.choice(VENDORS[cat])
        cur = rng.choice(CURRENCIES)
        amount = round(rng.choice([
            rng.uniform(3, 60), rng.uniform(60, 400), rng.uniform(400, 2500),
        ]), 2)
        if cur == "JPY":
            amount = float(rng.randint(500, 40000))
        d = received - timedelta(days=rng.randint(0, 14))

        gold = {"vendor": vendor, "amount": amount, "currency": cur,
                "date": d.isoformat(), "category": cat}
        amt_str = _fmt_amount(amount, cur, rng)
        date_str = _fmt_date(d, received, rng)
        text = rng.choice(TEMPLATES).format(vendor=vendor, amt=amt_str, d=date_str)

        # ~18% drop a field, matching the benchmark's missing-field cases.
        if rng.random() < 0.18:
            drop = rng.choice(["vendor", "amount", "date"])
            if drop == "vendor":
                text = text.replace(vendor, "the supplier")
                gold["vendor"] = None
            elif drop == "amount":
                text = text.replace(amt_str, "").strip(" —-,.")
                text = f"{text} (amount to follow)"
                gold["amount"] = gold["currency"] = None
            else:
                text = text.replace(date_str, "an unknown date")
                gold["date"] = None

        rows.append({"received": received.isoformat(), "text": text, "gold": gold})
    return rows


def write(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_benchmark() -> list[dict]:
    with open(config.EVAL_SET, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    rows = generate()
    # A held-out slice from the *same* generator, so the report can show the
    # in-distribution number next to the hand-written one.
    split = int(len(rows) * 0.9)
    write(rows[:split], config.DATA / "train.jsonl")
    write(rows[split:], config.DATA / "heldout_synthetic.jsonl")

    bench = load_benchmark()
    train_vendors = {v for vs in VENDORS.values() for v in vs}
    bench_vendors = {b["gold"]["vendor"] for b in bench if b["gold"]["vendor"]}
    overlap = train_vendors & bench_vendors
    assert not overlap, f"vendor leak between train and benchmark: {overlap}"

    n_null = sum(1 for r in rows[:split] if all(r["gold"][f] is None for f in config.FIELDS))
    print(f"train={split}  heldout_synthetic={len(rows)-split}  benchmark={len(bench)}")
    print(f"vendors: {len(train_vendors)} train, {len(bench_vendors)} benchmark, overlap 0")
    print(f"non-expense in train: {n_null/split:.1%}")
    print(f"example target: {task.target_json(rows[0]['gold'])}")


if __name__ == "__main__":
    main()
