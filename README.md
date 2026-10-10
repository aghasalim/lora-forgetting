# LoRA fine-tuning, with the forgetting check that usually gets skipped

[![ci](https://github.com/aghasalim/lora-forgetting/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/lora-forgetting/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23003658.svg)](https://doi.org/10.5281/zenodo.23003658)

I fine-tuned `Qwen2.5-1.5B-Instruct` with LoRA to pull structured JSON out of
casual expense messages. I trained it on a MacBook Pro, without CUDA or a cloud
GPU.

Saying "the loss went down" doesn't tell you much. I wanted two more things. One
is a baseline measured before I started training. The other is a check that I
didn't quietly break the rest of the model. I put both of them at the top.

---

## Abstract

People usually report LoRA fine-tuning as a gain on the target task. I wanted
to report the gain and the cost side by side. So I fine-tuned a small language
model to extract structured expense data, and then checked whether its general
ability survived.

The gain on the task is big. Exact match on all fields at once goes from 46.7%
to 75.6% on the held-out benchmark. The forgetting check found no cost I could
measure at this adapter size. ARC log-likelihood moves by 0.7 points, and ARC
generative accuracy and open-ended answering don't change at all.

I learned more from the breakdown by slice. The overall improvement hides two
slices that get *worse*. `written_amount` falls from 1.00 to 0.60 and
`currency` from 0.80 to 0.60, and two more don't change. When I reran with
more seeds, only one of those turned out to be real. `written_amount` is worse in 11 of 15
runs, `currency` in 5 of 15. The headline number itself moves between 66.7% and
82.2% depending on the training seed.

I also found something about the forgetting check itself. Scoring ARC by
ranking log-likelihoods and scoring it by letting the model generate an answer
differ by 16.7 points, on the same items with the same model. So if someone
makes a claim about forgetting, you need to know which method they used.

What's in the repo:

- Task gain and retained ability measured on the same adapter.
- A breakdown by slice that shows changes the overall number hides.
- Evidence that the way you score ARC moves the number more than the
  fine-tuning does.

---

## 1. Both numbers, up front

**Target task.** I wrote 45 hand-written cases whose vendors never appear in training.

| | base | fine-tuned | delta |
|---|---|---|---|
| valid JSON | 93.3% | **100%** | +6.7 |
| every field correct | 46.7% | **75.6%** | **+28.9** |
| date | 66.7% | 93.3% | +26.7 |
| category | 71.1% | 91.1% | +20.0 |

**General capability.** Same model, checked in two different ways.

| | base | fine-tuned | delta |
|---|---|---|---|
| ARC-Easy, log-likelihood (knowledge) | 72.0% | 71.3% | **−0.7** |
| ARC-Easy, generated answer (instruction following) | 88.7% | **88.7%** | **0.0** |
| answer parseable at all | 100% | 100% | 0.0 |
| open-ended factual probes | 100% | 100% | 0.0 |

I didn't find catastrophic forgetting, and I think that's a real result worth
taking seriously. This adapter is 0.28% of the model's parameters. I trained it
until it fully converged (final loss 0.0000) on a narrow task where every answer
is a JSON object. If I *wanted* to over-specialise a model, that's pretty much
how I'd do it. It still answers "what is the capital of France" in normal
sentences, and it still scores identically on 150
multiple-choice science questions.

The one movement, −0.7 points on log-likelihood, is one question out of 150. I
wouldn't call that getting worse.

---

![task gain on every extracted field](reports/figures/task-gain.png)

![general capability before and after](reports/figures/forgetting.png)

## 2. Why the forgetting check is two measurements
"Forgetting" can mean two different problems that need different fixes, and
one number can't tell you which one you have. I score knowledge by
log-likelihood over the answer options, without generating anything. For
instruction following I ask the same questions in chat and parse whatever comes
back. On the same 150 ARC items and the same base model, the two protocols disagree by 16.7 points, 72.0% ranked against 88.7%
generated. Both stayed put after tuning (72.0% to 71.3% and 88.7% to 88.7%).
So the model kept its facts and it still answers in normal sentences.

![the same ARC items scored two ways](reports/figures/arc-protocol.png)

I go through the two protocols and the 16.7 point gap between them on the same items in [the notes](notes/METHODS.md#1-why-the-forgetting-check-is-two-measurements).
## 3. What the aggregate number hides
The overall number goes up, but two slices get worse. In those, written-out amounts drop from
100% to 60%, currency from 80% to 60%. I read all four broken cases and three are
the same failure, category falling back to "other" for a vendor that never
showed up in the training data. Over the whole benchmark the category still went
32/45 to 41/45, so the fine-tune fixed 12 cases and broke 3.

![per-slice change after tuning](reports/figures/by-kind.png)

I go through all four broken cases one by one in [the notes](notes/METHODS.md#2-what-the-aggregate-number-hides).

### Is it the seed?

Each slice only has 5 cases, so one case is worth 20 points. From a single run
you can't tell a real regression from bad luck. So I reran training on a rented
RTX 3090. I added two more seeds at `r=16`, plus `r=8` and `r=32` at the
original seed, keeping alpha at twice the rank. The data, prompts and 45 cases
were all the same. The raw predictions and logs are in [`reports/sweep/`](reports/sweep/).

| run | every field correct | `written_amount` | `currency` |
| --- | ---: | ---: | ---: |
| base | 48.9% | 1.0 | 0.8 |
| r=16, seed 42 | 75.6% | 0.6 | 0.6 |
| r=16, seed 1 | 66.7% | 0.4 | 0.6 |
| r=16, seed 2 | 82.2% | 0.8 | 0.8 |
| r=8, seed 42 | 77.8% | 0.8 | 0.6 |
| r=32, seed 42 | 75.6% | 0.8 | 0.8 |

Seed 42 on the GPU gave exactly the same result as the published run, 75.6%
with the same numbers per slice. The base model scores 48.9% on the GPU against
46.7% on the Mac. That's one case out of 45 flipping on different hardware, so I
compare the GPU rows with the GPU base.

This first batch changed how I read section 3 in two ways. First, the headline
moves a lot with the seed. At `r=16` the three seeds give 66.7%, 75.6% and 82.2%, mean
74.8% and sd 7.8 points, and the published 75.6% just happened to be the middle
one. Every seed still beats the base, the worst being 17.8 points, but "+28.9"
is only one draw from a range. Second, `r=8` and `r=32` land at 77.8% and 75.6%,
which is inside that spread. With one seed each I can't say which is better.

### A second batch, on a different GPU

I ran the second batch on an RTX A5000, because the 3090 wasn't available
anymore ([`reports/sweep_a5000/`](reports/sweep_a5000/)). I wanted to know if
more seeds change the picture, if I really need three epochs, and if it helps to
put LoRA on the MLP projections too. Every row is `r=16`.

| setting | seeds | every field correct, per seed | mean |
| --- | --- | --- | ---: |
| attention, 3 epochs | 1, 2, 3, 4 | 71.1%, 80.0%, 68.9%, 80.0% | 75.0% |
| attention, 1 epoch | 42, 1, 2 | 71.1%, 80.0%, 75.6% | 75.6% |
| attention and MLP, 3 epochs | 42, 1, 2 | 77.8%, 82.2%, 77.8% | 79.3% |

The hardware changes the number as well. Seed 1 gives 66.7% on the 3090 and 71.1%
on the A5000, and seed 2 gives 82.2% and 80.0%. The code, seed and data are the
same, and still one or two of the 45 cases flip between GPUs. So even with a
fixed seed, a single run on this benchmark only reproduces to within about 4
points.

One epoch turned out to be enough. It scores about the same as three (75.6%
against 75.0%), in 50 s of training against 152 s. In section 6 I guessed from
the loss curve that three epochs was about three times more than the task
needed, and this backs that up.

Adding the MLP projections is the only change that looks like it helps. The mean
is 79.3%, and there's no seed below 77.8%. That's still inside the spread of the
attention runs, so three seeds aren't enough to be sure. It's the setting I'd
try first if I had more compute.

The `written_amount` drop looks mostly real, and the `currency` one looks like
noise. Across all 15 tuned runs written-out amounts are worse than base in 11 and currency in
5. With the original attention-only setup it's worse in 6 of 7. The MLP runs
keep it at the base level in 2 of 3. That would fit the idea that the
attention-only adapter is too narrow for that slice, but 3 runs can't show it.

Forgetting still doesn't show up with more seeds. I reran the ARC check for
seeds 1 and 2 on the A5000. Log-likelihood came out at 71.3% and 72.0% against 71.3% for the base
model on the same GPU, and generated answers at 88.0% and 88.7% against 88.7%.
That's the same no-forgetting result as section 1, now on three seeds.

## 4. The generalisation gap I built the experiment to see
| set | base | fine-tuned |
|---|---|---|
| held-out synthetic (same generator as training) | 28.0% | **95.3%** |
| hand-written benchmark (disjoint vendors, messier) | 46.7% | **75.6%** |

If I had generated the benchmark with the same script as the training data, I
would be reporting **95.3%** and really just measuring how well the model
memorised my templates. The gap between those two rows is 19.8 points. That's
the part of the gain that disappears on messages my generator never wrote. The
base model goes the other way here, scoring worse on the synthetic set (28.0%) than on the hand-written
one (46.7%). I think that's because the benchmark uses famous vendors it already
knew from pretraining, while the synthetic set mixes obscure ones with ten
currencies.

How I kept the benchmark separate from the training generator is in [the notes](notes/METHODS.md#3-the-generalisation-gap-i-built-the-experiment-to-see).
## 5. Running it

```bash
make setup && make data && make baseline
```

```bash
make train && make eval && make forgetting && make report
```

Those commands rebuild the whole results table. I run `make baseline` before
`make train` on purpose. If you measure the baseline after you already have a
fine-tuned model, it's too easy to talk yourself out of it. The separate
checkers in `verify/` also recompute each figure from the raw prediction and log
files, and the build fails if they don't match.

```bash
make app
```

---

## 6. Notes on training this on a laptop
`make feasibility` measures step time and memory before I commit to a run, and
it changed the project twice. First, it ruled out float32. At fp32 the run
needed 19.5 GB and 69 s/step, while bfloat16 needed 14.2 GB and 4.2 s/step.
Second, its own estimate was off, because it timed dummy batches of fixed
length. The real run on variable-length ones took 73.9 minutes. Loss was
already down to 0.003 by step 140 of 1014 and first touched 0.0001 at step 200,
so three epochs was about three times more than this task needed.

![training loss](reports/figures/training.png)

![the same training run replayed against the wall clock](reports/figures/training.gif)

*The whole run, 73.9 minutes of it, against the wall clock. Look at the pace more than the shape. Most of the drop happens in the first quarter, which is why the feasibility check mattered more to me than the loss curve.*

The feasibility numbers, the Docker stall and where the 73.9 minutes went are in [the notes](notes/METHODS.md#4-notes-on-training-this-on-a-laptop).
I also read the run back from the log and traced every setting to its line in
[notes/TRAINING.md](notes/TRAINING.md).
## 7. Limitations

The sweep is thin. I ran `r=8` and `r=32` once each and the MLP targets three
times (section 3). None of them stands clearly apart from the seed spread of
`r=16`, and I'm not claiming any of these values are the best ones.

The forgetting check only has three seeds. Section 1 is the laptop run, and I
rechecked seeds 1 and 2 on a GPU (section 3). I only scored the other runs on
the target task.

There's no hosted live demo. The comparison app reads predictions I computed
ahead of time, because a 1.5B model needs ~3 GB and the free tier only gives
1 GB. I actually think showing all 45 benchmark cases is more useful than a text
box, since you see the failures too and not only examples I picked.

I didn't compare against QLoRA. `bitsandbytes` has no MPS backend, so 4-bit
quantisation doesn't work on this machine at all.

## 8. Repository layout

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
verify/               every RESULTS.md row, re-derived from the raw JSON
RESULTS.md            generated from the measured JSON, not hand-typed
```

## 9. Licence

MIT, see [LICENSE](LICENSE).

## References

The papers behind the method, the effect it risks, and the tooling.

- **Hu, Shen, Wallis et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) the adaptation method.
- **Kirkpatrick, Pascanu, Rabinowitz et al. Overcoming catastrophic forgetting in neural networks. PNAS 114, 2017.** [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) the forgetting this repo measures.
- **McCloskey, Cohen. Catastrophic Interference in Connectionist Networks. Psychology of Learning and Motivation 24, 1989.** the original description of the effect.
- **Wolf, Debut, Sanh et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020.** [arXiv:1910.03771](https://arxiv.org/abs/1910.03771) the library.
