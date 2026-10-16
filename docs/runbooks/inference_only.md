# Inference-only runs of saved models

`predict.py` runs an already trained model and scaler without any training. The script supports this workflow:

1. pick an available model from `models/saved_models`;
2. feed it a CSV with features;
3. get the forecast as JSON.

Training scripts are neither run nor modified.

## List available models

```bash
.venv/bin/python predict.py --list-models
```

`model-id` format:

```text
<model_type>:<quality_bucket>:<pair>
```

Example:

```text
gru:95-100%:merged_ltc_bybit
```

## Run a forecast by model-id

```bash
.venv/bin/python predict.py \
  --model-id 'gru:95-100%:merged_ltc_bybit' \
  --data data/LTC/merged_ltc_bybit.csv \
  --output /tmp/nexgen_prediction.json
```

By default, the forecast is made only for the last available sequence.

## Interactive selection by number

```bash
.venv/bin/python predict.py \
  --interactive \
  --data data/LTC/merged_ltc_bybit.csv \
  --output /tmp/nexgen_prediction.json
```

This is the terminal equivalent of a drop-down list: the script lists the models and asks you to enter a number.

## GUI with a drop-down list

If a local graphical session is available, you can open the model selection window:

```bash
.venv/bin/python predict_gui.py
```

In the window you can:

- pick a model from the drop-down list;
- choose the CSV data file;
- choose the path for the JSON result;
- enable forecasts for all sequences and cap the output with `limit`;
- run prediction without training.

On a headless server without an X11/desktop session the GUI may fail to open; use the CLI commands above instead.

## Forecast for all sequences

```bash
.venv/bin/python predict.py \
  --model-id 'gru:95-100%:merged_ltc_bybit' \
  --data data/LTC/merged_ltc_bybit.csv \
  --all \
  --limit 20 \
  --output /tmp/nexgen_predictions.json
```

With `--all`, `--limit` caps the number of most recent predictions in the JSON output.

## What predict.py checks

- that a matching `.keras` model and `.pkl` scaler pair exists;
- that the CSV contains a numeric `close` column;
- feature order: `close` first, as in the training scripts;
- scaler compatibility by feature count;
- model shape compatibility: `window`, `features`, `steps`;
- no `NaN` / `inf` and no negative/zero prices after the inverse transform.

For a legacy scaler without `feature_names_in_`, the script warns that only the feature count and the model shape were strictly checked.
