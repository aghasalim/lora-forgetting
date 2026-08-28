---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
---

# LoRA adapter for structured JSON extraction

A LoRA adapter over `Qwen2.5-1.5B-Instruct`, trained to pull structured JSON out
of short free-text descriptions. It is the artifact produced by
[lora-forgetting](https://github.com/aghasalim/lora-forgetting), which exists to
check whether fine-tuning of this kind quietly costs general ability.

## What it does

Given a short description, it returns a JSON object with the fields the task
defines. On the held-out set built from the same generator it scores 95.3%
against the base model's 28.0%.

## What it costs

The point of the parent repo is the second measurement, not the first. General
ability was checked on ARC before and after, and the adapter shows no
significant drop. That result comes with a caveat worth reading: scoring ARC by
log-likelihood ranking and by free generation disagree by 16.7 points, so the
answer depends on how you ask. The parent repo documents that.

## Training

| setting | value |
|---|---|
| rank | 16 |
| alpha | 32 |
| dropout | 0.05 |
| target modules | q_proj, k_proj, v_proj, o_proj |
| task | causal language modelling |

No rank or target-module sweep was run, so these are a reasonable default
rather than a tuned choice.

## Limitations

Trained on synthetic descriptions from one generator, so it is fitted to that
distribution and has not been tested on text from anywhere else. The forgetting
check covers ARC only, which is one benchmark and not a general guarantee.
