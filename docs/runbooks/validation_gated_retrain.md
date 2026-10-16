# Validation-gated retrain

This workflow compares a new model (`candidate`) with the previous production model
(`baseline`) on the same holdout and decides whether the candidate may become the active model.

## Why this is needed

Training scripts save a new model **straight away as active** and move the previous one to
`archive/`. That means an unverified model starts serving inference before any
validation; this is exactly how a working SOL GRU model was once replaced by a noticeably worse one.

The gate restores control over this transition:

```text
<bucket>/<stem>.keras                      active   — what predict.py sees
<bucket>/staging/<stem>.keras              staged   — candidate, not yet verified
<bucket>/archive/<stem>_<timestamp>.keras  archived — history, read-only
```

`predict.py` never offers anything from `staging/` or `archive/` for inference, so a
staged candidate physically cannot serve forecasts.

## Automatic mode

`auto_retrain.py` runs the whole sequence by itself. The gate is enabled by default:

```bash
.venv/bin/python controller/auto_retrain.py --models gru
```

What happens for each trained model:

1. snapshot of the active artifacts before training;
2. training;
3. snapshot comparison, which determines which bundle the training run created or replaced
   (the accuracy category is not known in advance, so the path is computed, not guessed);
4. `stage`: the candidate moves from active to `staging/`;
5. `rollback`: the baseline is restored from the archive to active;
6. the gate compares baseline and candidate and applies the verdict:
   `PASS`: the candidate becomes active and the previous baseline goes to the archive;
   `REJECT`: the candidate goes to the archive tagged `rejected_` and the previous model stays active.

The verdict is sent to Telegram and written to `logs/auto_retrain.log`.

Special cases:

| Verdict | Meaning |
| --- | --- |
| `NO_BASELINE` | first training run for this pair, nothing to compare against. The candidate stays in `staging` and no active model is assigned; a human makes the decision |
| `STAGE_FAILED` / `ROLLBACK_FAILED` | artifacts are not in the expected state and the gate did not run. Investigate manually with `--action status` |
| `APPLY_FAILED` | the verdict was computed, but the artifacts could not be moved. The gate report is saved |

To disable the safeguard:

```bash
.venv/bin/python controller/auto_retrain.py --models gru --no-gate
```

The new model then becomes active without any comparison. The script sends a warning about this to Telegram.
The `--dry-run` flag also disables the gate: no training runs, so there is nothing to move.

## Manual mode

### 1. Collect data

Run the pipeline with the required symbol, exchange, timeframe and period. For a fair
comparison, baseline and candidate must be evaluated on the same final CSV.

### 2. Train

```bash
.venv/bin/python models/train_gru.py \
  --config configs/config.yaml \
  --file /path/to/merged.csv
```

### 3. Move the candidate out of active and restore the baseline

```bash
BUCKET='models/saved_models/gru_saved_model/95-100%'
STEM='SOLUSDT_SOLUSDC_15m_30d_gru'

# What is where right now
.venv/bin/python controller/model_staging.py --action status --bucket "$BUCKET" --stem "$STEM"

# Candidate → staging
.venv/bin/python controller/model_staging.py --action stage --bucket "$BUCKET" --stem "$STEM"

# Baseline from archive → active
.venv/bin/python controller/model_staging.py --action rollback --bucket "$BUCKET" --stem "$STEM"
```

Any operation can be previewed first with `--dry-run`: the command prints the planned
moves and does not touch a single file.

### 4. Compare and apply the verdict

```bash
.venv/bin/python controller/validation_gate.py \
  --baseline-model  "$BUCKET/$STEM.keras" \
  --baseline-scaler "$BUCKET/${STEM}_scaler.pkl" \
  --candidate-model  "$BUCKET/staging/$STEM.keras" \
  --candidate-scaler "$BUCKET/staging/${STEM}_scaler.pkl" \
  --data /path/to/merged.csv \
  --test-size 0.1 \
  --apply
```

Without `--apply`, the gate stays fully read-only, as in the original MVP.

`--apply` requires the candidate to be in `<bucket>/staging/` and the baseline to be
the current active pair with the same stem. Otherwise a verdict computed on one set of files
would move another, so the operation is rejected.

### Criteria

The gate compares the first forecast horizon against four mandatory conditions:

- candidate MAPE is no worse than baseline;
- candidate RMSE is no worse than baseline;
- candidate R² is no lower than baseline;
- candidate directional accuracy is no lower than baseline.

By default, any regression results in `REJECT`.

### Comparison with persistence: required reading

The gate always computes the metrics of the naive "price will not change" forecast and prints them
next to the model metrics. This is not a formality.

15-minute prices are autocorrelated almost perfectly. On real SOL/USDT data,
persistence on the holdout gives:

```
MAPE 0.3458%   RMSE 0.7227   R² 0.999797
```

That is an "accuracy" of 99.65%, and by the project's rules such a model would land in the
`95-100%` category without predicting anything. A model that has learned the function
"return the last input value" will show exactly the same impressive numbers.

So a model's absolute MAPE means nothing on its own. What matters is
`skill_vs_persistence` in the report:

```json
{
  "persistence": {"mape_pct": 0.3458, "rmse": 0.7227, "r2": 0.999797},
  "candidate": {
    "skill_vs_persistence": {
      "mape_pct_skill_pct": -0.96,
      "rmse_skill_pct": -1.10,
      "r2_delta": -0.000004,
      "beats_persistence": false
    }
  }
}
```

Positive skill means the model is better than the naive forecast. Negative skill means it is worse than
doing nothing.

`directional_accuracy` is deliberately not computed for persistence: its forecast equals
the entry price, so it has no view on direction and there is nothing to compare.

To make the check mandatory:

```bash
--require-beat-persistence
```

A candidate that loses to the naive forecast on MAPE and RMSE then gets `REJECT`,
even if it is better than the previous model. The flag is off by default: the comparison
is printed and written to the report but does not affect the verdict.

### Exit codes

```text
0 = PASS
2 = REJECT
3 = verdict computed, but it could not be applied to the artifacts
1 = startup error or incompatible artifacts
```

## Tolerances

If the strict gate turns out to be too sensitive, tolerances can be set explicitly:

```bash
--max-mape-regression-pct 1.0
--max-rmse-regression-pct 1.0
--max-r2-drop 0.01
--min-directional-accuracy-delta -1.0
```

Loosening the thresholds without a measured justification is not recommended.

## Guarantees when moving artifacts

`controller/model_staging.py` enforces the following without exception:

- nothing is deleted, only moved, so every operation is reversible;
- model and scaler always move as a pair; an incomplete pair is never created;
- an existing destination is never overwritten;
- an operation is first planned and validated in full, and only then executed;
  if any step fails, the moves already made are rolled back;
- every mutation is written to the append-only journal `logs/model_staging_journal.jsonl`,
  so the move history can be reconstructed manually.

Versions tagged `rejected_` or `demoted_` remain on disk but are not offered as the
default rollback target. Such a version can only be restored with an explicit `--timestamp`.

## Manual rollback at any time

If the active model turns out to be worse than the previous one, regardless of the gate:

```bash
# Most recent clean archived version
.venv/bin/python controller/model_staging.py --action rollback --bucket "$BUCKET" --stem "$STEM"

# A specific version
.venv/bin/python controller/model_staging.py --action rollback --bucket "$BUCKET" --stem "$STEM" \
  --timestamp 2026-05-31_17-09-23
```

The current active pair is then moved to the archive with a `demoted_` tag instead of being deleted.
