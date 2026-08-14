# LoRA fine-tuning, with the forgetting check that usually gets skipped

[![ci](https://github.com/aghasalim/lora-forgetting/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/lora-forgetting/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tuning `Qwen2.5-1.5B-Instruct` with LoRA to pull structured JSON out of
informal expense messages, by a third-year Applied Computer Science (AI)
student. Trained on a MacBook Pro — no CUDA, no cloud GPU.

"I fine-tuned a model and the loss went down" is not a result. The two things
that make it one are a baseline you measured before you started, and a check
that you did not quietly break everything else. Both are here, and both numbers
lead.

---

## Both numbers, up front

**Target task**, 45 hand-written cases whose vendors never appear in training:

| | base | fine-tuned | delta |
|---|---|---|---|
| valid JSON | 93.3% | **100%** | +6.7 |
| every field correct | 46.7% | **75.6%** | **+28.9** |
| date | 66.7% | 93.3% | +26.7 |
| category | 71.1% | 91.1% | +20.0 |

**General capability**, same model, checked two different ways:

| | base | fine-tuned | delta |
|---|---|---|---|
| ARC-Easy, log-likelihood (knowledge) | 72.0% | 71.3% | **−0.7** |
| ARC-Easy, generated answer (instruction following) | 88.7% | **88.7%** | **0.0** |
| answer parseable at all | 100% | 100% | 0.0 |
| open-ended factual probes | 100% | 100% | 0.0 |

**No catastrophic forgetting** — and I want to be careful about how that reads,
because it is a real result rather than a relieved shrug. This adapter is 0.28%
of the model's parameters, trained to complete convergence (final loss 0.0000)
on a narrow task whose every answer is a JSON object. That is roughly the recipe
you would design if you *wanted* to over-specialise a model. It still answers
"what is the capital of France" in prose, and it still scores identically on 150
multiple-choice science questions.

The one movement, −0.7 points on log-likelihood, is one question out of 150. I
am not going to call that degradation.

---

## Why the forgetting check is two measurements

"Forgetting" hides two failures that need different fixes, and one number cannot
tell them apart:

- **Knowledge** is scored by log-likelihood over the answer options — no
  generation at all. A model that has lost the *habit* of answering multiple
  choice still scores normally here if it still knows the answer.
- **Instruction following** asks the same questions in chat and parses what
  comes out. A model fine-tuned to emit only JSON can retain every fact and
  still fail this, by replying `{"vendor": null, ...}` to a history question.

If knowledge holds and generation collapses, nothing was forgotten — the model
was over-specialised into one output format, and the fix is mixing general data
back into training, not lowering the learning rate. Reporting one number would
have pointed at the wrong remedy. Here both held, so neither fix is needed.

---

## What the aggregate number hides

+28.9 points looks like an unambiguous win. Broken out by difficulty it is not:

| kind | base | fine-tuned | delta |
|---|---|---|---|
| not an expense at all | 0% | **100%** | +100 |
| missing fields | 0% | 60% | +60 |
| ambiguous category | 40% | **100%** | +60 |
| relative dates | 0% | 80% | +80 |
| direct | 80% | 100% | +20 |
| distractor numbers | 80% | 80% | 0 |
| messy formatting | 40% | 40% | 0 |
| **currency** | **80%** | **60%** | **−20** |
| **written-out amounts** | **100%** | **60%** | **−40** |

**Two categories got worse.** Chasing them down was the most useful hour of the
project.

The base model's biggest weakness was not extraction, it was *abstention*: 0% on
messages that are not expenses, and 0% on messages with missing fields. It
always filled every slot in. Fine-tuning taught it that `null` is a real answer,
which is where most of the gain comes from.

The regressions are narrower than they look. Inspecting all four broken cases,
three are the same failure: **category falling back to `"other"` for a vendor
the training data never contained** — Slack and Namecheap are both obviously
software, and the base model knew that from pretraining. Net across the
benchmark, category went 32/45 → 41/45, so the fine-tune **fixed 12 and broke
3**. It is a real regression on a real mechanism, and it is still a large net
gain. Both halves of that sentence belong in the report.

For balance, one thing I suspected and had to drop: I thought the tuned model
was inventing vendors where the gold answer is null. It does it once in eight
cases — and so does the base model. Identical. Not a regression.

The base model's failures were mostly *schema* failures rather than reading
failures. It emitted categories that do not exist in the schema it was given —
`transportation`, `hotel`, `donation` — and produced invalid JSON 6.7% of the
time. Fine-tuning fixed the format completely (100% valid JSON, 100% schema
conformance) and that accounts for much of the headline gain.

---

## The generalisation gap I built the experiment to see

| set | base | fine-tuned |
|---|---|---|
| held-out synthetic (same generator as training) | 28.0% | **95.3%** |
| hand-written benchmark (disjoint vendors, messier) | 46.7% | **75.6%** |

Had I generated the benchmark from the same script as the training data, this
project would report **95.3%** and be measuring template memorisation. The
20-point gap between those rows is the part of the gain that does not
generalise, and it is only visible because the two sets were built separately —
hand-written messages, and a vendor list that shares nothing with training. A
test asserts that disjointness so it cannot silently regress.

Worth noting the base model scores *worse* on the synthetic set (28.0%) than on
the hand-written one (46.7%). Not a paradox: the benchmark uses famous vendors,
so "Figma is software" is answerable from pretraining, while the synthetic set
mixes obscure vendors with ten currencies including DKK, NOK and PLN.

---

## Running it

```bash
make setup && make data && make baseline
```

```bash
make train && make eval && make forgetting && make report
```

That reproduces every number above. `make baseline` before `make train` is the
order on purpose: a baseline measured after you already have a fine-tuned model
is a baseline you can talk yourself out of.

```bash
make app
```

---

## Notes on training this on a laptop

`make feasibility` measures step time and memory before committing to a run, and
it changed the project twice.

**float32 was unusable.** At fp32 the run needed 19.5 GB and managed 69 s/step.
In bfloat16 it needed 14.2 GB and managed 4.2 s/step — a 16× speedup that is
mostly *memory pressure*, not arithmetic: fp32 sits close enough to the 24 GB
ceiling that the machine starts swapping.

**And my feasibility number was still wrong.** It measured fixed-length dummy
batches. Real variable-length batches ran at 7–15 s/step, and the run degraded
over time until I found a Docker VM from an earlier project still holding 6 GB.
Freeing it restored the speed. The whole run took 74 minutes against a predicted
26. A benchmark on synthetic batches predicts synthetic throughput — which is
the same lesson this repo reports about synthetic evaluation data, arriving from
an unexpected direction.

Final loss was 0.0001 by step 140 of 1014, so **3 epochs was roughly 3× more
than this task needed**. I would use one epoch next time. The over-training is
what makes the forgetting result meaningful rather than lucky, so it is reported
as run, not quietly re-run with better settings.

---

## What is not here

- **No rank or target-module sweep.** `r=16` on attention projections was chosen
  up front and never varied. One run is 74 minutes on this hardware, so a sweep
  was out of budget. Nothing in this repo claims those values are optimal.
- **No hosted live demo.** The comparison app reads precomputed predictions
  because a 1.5B model needs ~3 GB against a 1 GB free tier. Showing all 45
  benchmark cases is more informative than a text box anyway — you can see the
  failures rather than the examples I would have picked.
- **No QLoRA comparison.** `bitsandbytes` has no MPS backend, so 4-bit
  quantisation is not available on this machine at all.

## Layout

```
src/loraft/
  config.py       every knob, with the measurement that justified it
  task.py         prompt construction and scoring
  data.py         training generator, vendors disjoint from the benchmark
  train.py        LoRA loop; loss masked to answer tokens only
  evaluate.py     identical prompts for base and tuned
  forgetting.py   knowledge vs instruction-following, measured separately
eval/eval_set.jsonl   45 hand-written cases
tests/                20 tests, no model or network needed
RESULTS.md            generated from the measured JSON, not hand-typed
```

## License

MIT — see [LICENSE](LICENSE).
