# Methods and detail

Long form detail moved out of the README.


## 2. Why the forgetting check is two measurements


![the same ARC items scored two ways](../reports/figures/arc-protocol.png)

"Forgetting" hides two failures that need different fixes, and one number cannot
tell them apart:

- **Knowledge** is scored by log-likelihood over the answer options, no
  generation at all. A model that has lost the *habit* of answering multiple
  choice still scores normally here if it still knows the answer.
- **Instruction following** asks the same questions in chat and parses what
  comes out. A model fine-tuned to emit only JSON can retain every fact and
  still fail this, by replying `{"vendor": null, ...}` to a history question.

If knowledge holds and generation collapses, nothing was forgotten, the model
was over-specialised into one output format, and the fix is mixing general data
back into training, not lowering the learning rate. Reporting one number would
have pointed at the wrong remedy. Here both held, so neither fix is needed.

---


## 3. What the aggregate number hides


![per-slice change after tuning](../reports/figures/by-kind.png)

Two slices get worse and two are unchanged while the aggregate improves. The
adapter is redistributing accuracy between question kinds rather than lifting all
of them, which is invisible in a single exact-match number.

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
the training data never contained**, Slack and Namecheap are both obviously
software, and the base model knew that from pretraining. Net across the
benchmark, category went 32/45 → 41/45, so the fine-tune **fixed 12 and broke
3**. It is a real regression on a real mechanism, and it is still a large net
gain. Both halves of that sentence belong in the report.

For balance, one thing I suspected and had to drop: I thought the tuned model
was inventing vendors where the gold answer is null. It does it once in eight
cases, and so does the base model. Identical. Not a regression.

The base model's failures were mostly *schema* failures rather than reading
failures. It emitted categories that do not exist in the schema it was given `transportation`, `hotel`, `donation`, and produced invalid JSON 6.7% of the
time. Fine-tuning fixed the format completely (100% valid JSON, 100% schema
conformance) and that accounts for much of the headline gain.

---


## 4. The generalisation gap I built the experiment to see


| set | base | fine-tuned |
|---|---|---|
| held-out synthetic (same generator as training) | 28.0% | **95.3%** |
| hand-written benchmark (disjoint vendors, messier) | 46.7% | **75.6%** |

Had I generated the benchmark from the same script as the training data, this
project would report **95.3%** and be measuring template memorisation. The
20-point gap between those rows is the part of the gain that does not
generalise, and it is only visible because the two sets were built separately
hand-written messages, and a vendor list that shares nothing with training. A
test asserts that disjointness so it cannot silently regress.

Worth noting the base model scores *worse* on the synthetic set (28.0%) than on
the hand-written one (46.7%). Not a paradox: the benchmark uses famous vendors,
so "Figma is software" is answerable from pretraining, while the synthetic set
mixes obscure vendors with ten currencies including DKK, NOK and PLN.

---


## 6. Notes on training this on a laptop


![training loss](../reports/figures/training.png)

`make feasibility` measures step time and memory before committing to a run, and
it changed the project twice.

**float32 was unusable.** At fp32 the run needed 19.5 GB and managed 69 s/step.
In bfloat16 it needed 14.2 GB and managed 4.2 s/step, a 16× speedup that is
mostly *memory pressure*, not arithmetic: fp32 sits close enough to the 24 GB
ceiling that the machine starts swapping.

**And my feasibility number was still wrong.** It measured fixed-length dummy
batches. Real variable-length batches ran at 7 to 15 s/step, and the run degraded
over time until I found a Docker VM from an earlier project still holding 6 GB.
Freeing it restored the speed. The whole run took 74 minutes against a predicted
26. A benchmark on synthetic batches predicts synthetic throughput, which is
the same lesson this repo reports about synthetic evaluation data, arriving from
an unexpected direction.

Final loss was 0.0001 by step 140 of 1014, so **3 epochs was roughly 3× more
than this task needed**. I would use one epoch next time. The over-training is
what makes the forgetting result meaningful rather than lucky, so it is reported
as run, not quietly re-run with better settings.

---
