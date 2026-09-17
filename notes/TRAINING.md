# The training run, read back from its log

One run, 2026-08-14, never repeated. What survives of it is
`reports/train_log.csv` and the adapter it saved, and this page reads
both back and names the line of code behind every setting. The prose
claims in README section 6 are held to the same log by
`verify/trainlog.js`.

## When and on what

The log and the adapter weights were committed together in d10f02f on
2026-08-14 at 15:39 (+0400). The log has no date column, so that commit
is the upper bound. MacBook Pro with an Apple M4 and 24 GB, the MPS
backend, bfloat16 (README, opening paragraph; `config.py:23` to `:26`,
`:45`, `:49`).

## Every setting, with its line

| setting | value | where |
|---|---|---|
| base model | `Qwen/Qwen2.5-1.5B-Instruct` | `config.py:21` |
| dtype | bfloat16 | `config.py:45` |
| seed | 42 | `config.py:55`, applied at `train.py:60` |
| rank, alpha, dropout | 16, 32, 0.05 | `config.py:72` to `:74`; `artifacts/adapter/adapter_config.json` |
| target modules | q_proj, k_proj, v_proj, o_proj | `config.py:82` |
| trainable share | 0.28% | `config.py:76` |
| training rows | 1,350, the first 90% of 1,500 generated | `config.py:89`, `data.py:178` |
| batch | 4 | `config.py:86` |
| epochs | 3 | `config.py:85` |
| steps | 1,014, 338 per epoch | 1,350 / 4 rounded up, times 3 |
| max length | 384 tokens | `config.py:88` |
| optimiser | AdamW, lr 1e-4, weight decay 0 | `config.py:87`, `train.py:83` |
| warmup | 30 steps, linear | `train.py:85`, `:88` to `:89` |
| decay | cosine to zero | `train.py:91` |
| gradient clip | 1.0 | `train.py:107` |
| loss | answer tokens only, prompt masked with -100 | `train.py:23`, `:43` to `:44` |
| logging | step 1, then every 10 steps | `train.py:111` |

The step count is not in the log, which stops at 1010. It is bracketed
by it: the `epoch` column turns over between steps 330 and 340 and
again between 670 and 680, so an epoch is 338 steps and three of them
are 1,014.

## What the clock column says

The last logged row is step 1010 at 4,434.5 s, which is 73.9 minutes,
and the README rounds that to 74. Averaged over the whole run that is
4.39 s per step, close to the 4.2 s the feasibility check predicted
(`config.py:24`). The average hides the shape. Dividing the gap in
`seconds` by the gap in `step` between consecutive rows:

| steps | s per step |
|---|---:|
| 1 to 100 | 2.6 |
| 100 to 200 | 4.6 |
| 200 to 300 | 8.7 |
| 300 to 400 | 12.2 |
| 400 to 500 | 3.9 |
| 500 to 1010 | 2.4 |

The slowest ten-step interval is 360 to 370 at 16.0 s per step. From
step 420 on, every interval sits at 2.3 to 2.5 s. This is the stall
described in `notes/METHODS.md` section 4, a Docker VM holding 6 GB
until it was found and stopped, and the log places the recovery at
about step 420, 27 minutes in. The 7 to 15 s per step quoted there is
the stalled stretch, not the run. Free of it, the machine ran a third
faster than the fixed-length feasibility batches predicted.

## What the loss column says

| | step | loss |
|---|---:|---:|
| first row | 1 | 0.127 |
| highest logged | 10 | 0.1419 |
| peak learning rate, 9.998e-05 | 40 | 0.0605 |
| the README's step 140 | 140 | 0.0027 |
| first at or under 0.001 | 160 | 0.001 |
| first 0.0001 | 200 | 0.0001 |
| first exactly 0.0 | 600 | 0.0 |
| last row | 1010 | 0.0 |

It does not stay down once it gets there. After step 200 the log has 23
rows above 0.001, the largest 0.0402 at step 290, all of them single
batches of four, and the last one above 0.001 is step 900. Six rows in
the whole log read exactly 0.0.

## What it wrote

- `reports/train_log.csv`, 102 rows, columns step, epoch, loss, lr,
  seconds (`train.py:96`).
- `artifacts/adapter/`, by `save_pretrained` (`train.py:123`, path at
  `config.py:15`). `adapter_model.safetensors` is 17,462,432 bytes: 224
  tensors, 4,358,144 values, stored as float32 (its own header).
  `adapter_config.json` records peft 0.20.0, r 16, alpha 32, dropout
  0.05. Both are committed, and `.gitignore` says why.
- Nothing else. The loop saves once, at the end, and never writes a
  `checkpoint-*` directory; the ignore rule for those is for a `Trainer`
  this repo does not use. There is no evaluation during training and no
  memory reading in the log.

## Not in the repo

- The date and time the run started. The commit is the only clock.
- Memory during the run. The 14.2 GB in `config.py:24` is from the
  feasibility check, not from this training.
- The exact total step count as a logged number. 1,014 is derived above.
- `notes/METHODS.md` section 4 says the loss was 0.0001 by step 140; the
  log says 0.0027 at 140 and 0.0001 first at 200, which is what the
  README says. The METHODS sentence is the one that is off.
