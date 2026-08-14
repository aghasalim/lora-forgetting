# Results

Base model `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 alpha=32 on ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 3 epochs, lr=0.0001, batch 4.

## Target task

### Hand-written benchmark (45 cases, vendors disjoint from training)

| metric | base | fine-tuned | delta |
|---|---|---|---|
| json_parsed | 93.3% | 100.0% | +6.7% |
| schema_ok | 93.3% | 100.0% | +6.7% |
| vendor | 82.2% | 93.3% | +11.1% |
| amount | 86.7% | 100.0% | +13.3% |
| currency | 82.2% | 93.3% | +11.1% |
| date | 66.7% | 93.3% | +26.7% |
| category | 71.1% | 91.1% | +20.0% |
| all_correct | 46.7% | 75.6% | +28.9% |

### By difficulty

| kind | base | fine-tuned | delta |
|---|---|---|---|
| ambiguous_category | 40.0% | 100.0% | +60.0% |
| currency | 80.0% | 60.0% | -20.0% |
| direct | 80.0% | 100.0% | +20.0% |
| distractor | 80.0% | 80.0% | +0.0% |
| messy | 40.0% | 40.0% | +0.0% |
| missing_field | 0.0% | 60.0% | +60.0% |
| not_expense | 0.0% | 100.0% | +100.0% |
| relative_date | 0.0% | 80.0% | +80.0% |
| written_amount | 100.0% | 60.0% | -40.0% |

### Generalisation gap

| set | base | fine-tuned |
|---|---|---|
| held-out synthetic (same generator) | 28.0% | 95.3% |
| hand-written benchmark (disjoint vendors) | 46.7% | 75.6% |

## Catastrophic forgetting check

| check | base | fine-tuned | delta |
|---|---|---|---|
| ARC-Easy, log-likelihood (knowledge) | 72.0% | 71.3% | -0.7% |
| ARC-Easy, generated (instruction following) | 88.7% | 88.7% | +0.0% |
| answer parseable at all | 100.0% | 100.0% | +0.0% |
| open-ended factual probes | 100.0% | 100.0% | +0.0% |
