# Project Architecture

This document describes the current architecture of Nexgen Crypto Pro: its subsystems, data flow, model design, storage layout, deployment and known technical debt. For a quick start and a summary of results, see [README.md](README.md).

## Overview

Nexgen Crypto Pro is a Python project that does six things:

- collects cryptocurrency OHLCV data
- builds datasets from that data
- trains neural forecasting models
- evaluates the models with honest baselines
- serves them in inference-only mode
- runs part of its operations through a Telegram bot

The project consists of several independent subsystems:

- **`pipeline/`**
  - Collects data from exchanges, checks data quality and merges the results.
  - Builds plots and sends Telegram reports.
  - Hosts the Telegram bot that manages per-user collection sessions.
- **`models/`** trains the LSTM, GRU and hybrid models and saves the trained artifacts.
- **`utils/`** holds shared helpers: feature engineering, preprocessing, prediction targets, volatility, metrics, plots, logging and Telegram messaging.
- **`controller/`** holds the orchestration scripts:
  - sequential training and automatic retraining
  - the validation gate and model staging
  - the volatility baseline
  - the forecast-journal producer and publisher
- **Root scripts** handle inference (`predict.py`, `predict_gui.py`), backtesting (`backtest.py`) and summary reports (`generate_report*.py`).
- **Runtime directories** hold data, logs, sessions, state, models, plots and run archives. They are git-ignored.

## Repository structure

```text
.
├── configs/
│   ├── config.yaml                       # default training config (JSON body)
│   ├── config_gpu_sol.yaml               # GPU training config for SOL datasets (YAML)
│   ├── config_gpu_sol_returns.yaml       # same config, log-return target
│   ├── forecast_sol_15m_pipeline.yaml    # collector config for the forecast cycle
│   ├── sol_usdt_15m_full_pipeline.yaml   # SOL/USDT 15m, full history
│   ├── sol_usdc_15m_full_pipeline.yaml   # SOL/USDC 15m, full history (contains a listing gap)
│   └── sol_usdc_15m_clean_pipeline.yaml  # SOL/USDC 15m, continuous segment after relisting
├── controller/
│   ├── run_models.py
│   ├── auto_retrain.py
│   ├── validation_gate.py
│   ├── model_staging.py
│   ├── volatility_baseline.py
│   ├── forecast_producer.py
│   ├── run_forecast_cycle.py
│   └── publish_forecast_journal.py
├── models/
│   ├── train_lstm.py
│   ├── train_gru.py
│   ├── train_both.py
│   ├── train_gru11.py                    # archived Keras Tuner experiment
│   ├── LSTM2v4.py                        # archived early monolithic script
│   └── ktuner_visualize.py
├── pipeline/
│   ├── crypto_pipeline_v4.3.py
│   ├── bot_pipeline_control.py
│   ├── pipeline_config.yaml
│   ├── access.json.example
│   └── test_run/test_config.yaml
├── utils/
│   ├── data_utils.py
│   ├── features.py
│   ├── prep_utils.py
│   ├── targets.py
│   ├── volatility.py
│   ├── metrics.py
│   ├── plots.py
│   ├── logger.py
│   └── telegram_utils.py
├── tests/                                # unittest suite (9 modules)
├── docs/runbooks/                        # operational runbooks
├── deploy/                               # systemd unit, VPS setup script, GPU Dockerfile
├── scripts/with_gpu.sh
├── betatest/                             # early backtest and GUI experiments
├── qci/qci_v_0.01.py                     # standalone decision-support prototype
├── backtest.py
├── predict.py
├── predict_gui.py
├── generate_report.py
├── generate_report_V_2.py
├── requirements.txt                      # combined reference list (not a lock file)
├── requirements-ml.txt                   # pinned .venv (ML stack)
├── requirements-pipeline.txt             # pinned pipeline/.venv (pipeline/bot stack)
├── .env.example
├── README.md
└── PROJECT_ARCHITECTURE.md
```

The following runtime directories are git-ignored and created on first use:

```text
data/                     # training datasets (data/data_base/ is the default training input)
logs/                     # root logs: errors, auto-retrain, validation reports, staging journal
state/                    # forecast-cycle inputs, lock file and forecast journal
sessions/                 # per-user Telegram sessions
models/saved_models/      # trained models, scalers, metadata, plots, training logs
pipeline/base_data/       # raw CSV per exchange and symbol
pipeline/full_data_base/  # merged dataset (merged.csv)
pipeline/plot/            # collector plots
pipeline/archive/         # archived outputs of previous collector runs
pipeline/logs/            # collector and bot logs, per-user activity logs
```

### Key directories

| Path | Role |
| --- | --- |
| `pipeline/` | Async market-data collection, quality checks, merge, plots, Telegram bot control, per-user sessions. |
| `models/` | LSTM/GRU/hybrid training scripts, a Keras Tuner helper and archived experimental scripts. |
| `utils/` | Shared feature engineering, scaling, sequence generation, targets, volatility, metrics, plotting and Telegram helpers. |
| `controller/` | Training orchestration, validation gate, artifact staging, volatility baseline, forecast journal. |
| `configs/` | Training configs and dedicated collector configs for research datasets and the forecast cycle. |
| `data/` | Historical and prepared datasets for the ML side. |
| `sessions/` | Per-user Telegram sessions with their own config, logs, plots, archives and datasets. |
| `models/saved_models/` | Saved models, scalers, metadata sidecars, training plots, prediction plots and training logs. |

## Entry points

| Command | Environment | Purpose |
| --- | --- | --- |
| `pipeline/crypto_pipeline_v4.3.py` | `pipeline/.venv` | OHLCV collection via `ccxt.async_support`, cleaning, merge, indicators, plots, Telegram summary. |
| `pipeline/bot_pipeline_control.py` | `pipeline/.venv` | aiogram Telegram bot: roles, access, settings menu, starting/stopping the collector, personal sessions. |
| `models/train_lstm.py` | `.venv` | Trains a bidirectional LSTM with attention; saves the model, scaler, metadata and plots. |
| `models/train_gru.py` | `.venv` | Trains a bidirectional GRU with attention; saves the model, scaler, metadata and plots. |
| `models/train_both.py` | `.venv` | Trains the hybrid BiLSTM → BiGRU model with attention. |
| `controller/run_models.py` | `.venv` | Runs training scripts sequentially (only the LSTM entry is enabled in `MODELS_TO_RUN`). |
| `controller/auto_retrain.py` | `.venv` | Retrains the selected models on the newest dataset and routes each new model through the validation gate. |
| `controller/validation_gate.py` | `.venv` | Compares baseline and candidate models on the same holdout; with `--apply`, promotes or rejects the candidate. |
| `controller/model_staging.py` | `.venv` | Moves model bundles between active, staging and archive (`status`, `stage`, `promote`, `reject`, `rollback`, `archive-active`). |
| `controller/volatility_baseline.py` | `.venv` | Fits and evaluates the HAR realized-volatility baseline. |
| `controller/run_forecast_cycle.py` | `.venv` (spawns both) | Collects fresh data, runs the forecast producer and optionally publishes the journal. |
| `predict.py` / `predict_gui.py` | `.venv` | Inference-only predictions from saved model bundles (CLI / Tk GUI). |
| `backtest.py` | `.venv` | Evaluates a saved price-level model on historical data with a virtual long-only trader. |
| `generate_report.py` / `generate_report_V_2.py` | `.venv` | Scans training logs and plots and builds a Markdown summary (`SUMMARY_REPORT.md`). |

Typical commands:

```bash
# Data pipeline and Telegram bot
pipeline/.venv/bin/python pipeline/crypto_pipeline_v4.3.py
pipeline/.venv/bin/python pipeline/bot_pipeline_control.py

# Training
.venv/bin/python models/train_lstm.py --config configs/config.yaml --file path/to/dataset.csv
.venv/bin/python models/train_gru.py  --config configs/config.yaml --file path/to/dataset.csv
.venv/bin/python models/train_both.py --config configs/config.yaml --file path/to/dataset.csv

# Orchestration and gated retraining
.venv/bin/python controller/run_models.py
.venv/bin/python controller/auto_retrain.py --models gru,lstm,hybrid

# Inference, backtest and reports
.venv/bin/python predict.py --list-models
.venv/bin/python backtest.py --data path/to/dataset.csv \
  --model  "models/saved_models/lstm_saved_model/<bucket>/<stem>_lstm.keras" \
  --scaler "models/saved_models/lstm_saved_model/<bucket>/<stem>_lstm_scaler.pkl" \
  --window 30 --steps 2
.venv/bin/python generate_report.py
```

## Python environments

The project uses two Python environments:

| Environment | Role | Dependency file | Key packages |
| --- | --- | --- | --- |
| `.venv/` | ML training, inference, backtest, reports, tests | `requirements-ml.txt` | tensorflow 2.21, keras 3.14, scikit-learn 1.8, pandas 3.0.2, numpy 2.4.4, matplotlib, joblib, python-dotenv, PyYAML |
| `pipeline/.venv/` | Telegram bot and data collection | `requirements-pipeline.txt` | aiogram 3.22, ccxt 4.5.40, psutil 7.2.2, aiohttp 3.12, pandas 3.0.1, numpy 2.4.2, matplotlib, python-dotenv, PyYAML |

The two environments do not interfere, because each process runs with its own interpreter and its own site-packages. The pipeline environment is light and suits the long-running bot. The ML environment is heavy and is only needed for ML jobs.

```bash
# ML environment (.venv/)
python3 -m venv .venv
.venv/bin/pip install -r requirements-ml.txt
# Optional, for hyperparameter search:
# .venv/bin/pip install "keras-tuner>=1.4.0"

# Pipeline / Telegram environment (pipeline/.venv/)
python3 -m venv pipeline/.venv
pipeline/.venv/bin/pip install -r requirements-pipeline.txt
```

`requirements-ml.txt` and `requirements-pipeline.txt` are exact freezes of the real environments, so use them to reproduce a setup. The root `requirements.txt` is a combined reference list only.

### Production rules

- Do not run production processes with a bare `python3` unless the right environment has been activated.
- In systemd units, cron jobs, shell scripts and bot-spawned subprocesses, use the absolute path of the right interpreter.
  - The bot launches the collector with `PIPELINE_PYTHON` (env), then `pipeline_python` (config), then its own `sys.executable`.
  - `run_forecast_cycle.py` takes `--pipeline-python` and `--ml-python`.
- Keep the ML and pipeline dependency sets separate.
- On GPU hosts where CUDA libraries come from pip wheels, wrap ML commands in `scripts/with_gpu.sh`. It adds every `nvidia-*` package `lib` directory to `LD_LIBRARY_PATH`. Child processes inherit the setting, so `auto_retrain.py` subprocesses also see the GPU.

## Data flow

```text
Exchanges
  │
  └── CCXT async
        │
        ▼
pipeline/crypto_pipeline_v4.3.py
  ├── internet and exchange health checks
  ├── safe_fetch OHLCV (semaphore, retries, adaptive delay)
  ├── base_data/*.csv
  ├── time-series cleaning and quality checks
  ├── merge_and_analyze()
  ├── full_data_base/merged.csv and/or merged.json (output_format)
  └── plot/*.png + Telegram summary
        │
        ▼
models/train_lstm.py / train_gru.py / train_both.py
  ├── numeric feature matrix, close first
  ├── train-only scaling (utils/prep_utils.py)
  ├── sliding windows + targets (utils/prep_utils.py, utils/targets.py)
  ├── chronological train/val/test split
  ├── bidirectional RNN + multi-head attention
  ├── metrics vs persistence, prediction and attention plots
  └── models/saved_models/<model_type>_saved_model/<accuracy_bucket>/
        │
        ├──► controller/validation_gate.py + model_staging.py
        ├──► predict.py / predict_gui.py / controller/forecast_producer.py
        ├──► backtest.py
        └──► generate_report.py / Telegram reports
```

### Collection pipeline stages

1. **Configuration.** The collector reads its config once. It uses `PIPELINE_CONFIG` if that is set and otherwise falls back to `pipeline/pipeline_config.yaml`.
2. **Period selection.** The collection period comes from `collection_mode`:
   - `auto`: the last `history_days` days
   - `full`: since 2010, so the whole listing history is fetched
   - `days` / `hours`: fixed offsets
   - `custom`: an explicit start and end
3. **Fetching.** OHLCV data is fetched concurrently for every enabled exchange × symbol pair. A semaphore caps the number of concurrent requests. Requests have per-request timeouts and retries, and the delay between requests grows on failures and shrinks after successes.
4. **Raw storage.** Raw candles are saved as `<exchange>_<SYMBOL>_<timeframe>_<date>.csv`.
5. **Merge.**
   - Raw files are concatenated, and zero-volume rows and duplicates are dropped.
   - Each `(symbol, exchange)` series is reindexed onto the full timeframe grid.
   - A series is dropped if more than 30% of its bars are missing.
6. **Patching.** If Binance data is present together with KuCoin or Bybit data, gaps in the Binance series are filled from the alternate exchange.
7. **Quality report.** The collector logs missing timestamps and outliers (bar-to-bar moves greater than 15%), then forward- and back-fills the remaining NaNs.
8. **Indicators** are computed, and rows left incomplete by the indicator warm-up are dropped.
9. **Outputs.** The merged CSV and/or JSON is written, along with one price plot per pair and a summary. When Telegram is enabled, the summary, the plots and, depending on the role, the dataset are sent there.
10. **Cache and archiving.** With `use_cache: false`, previous merged outputs and raw CSV files are moved to `archive/run_<timestamp>/` before a fresh run.

### Features

The collector adds these indicators to the raw OHLCV columns:

| Group | Columns |
| --- | --- |
| Trend | `SMA_14`, `EMA_14` |
| Return and momentum | `RETURN`, `MOM_10` |
| Oscillators | `RSI_14`, `CCI_20`, `WILLR_14` |
| MACD | `MACD`, `MACD_signal`, `MACD_hist` |
| Bollinger Bands (20) | `BB_MID`, `BB_STD`, `BB_UPPER`, `BB_LOWER` |
| Volatility | `STD_14`, `ATR_14` |

`utils/features.py` contains a separate set of feature helpers: technical indicators, lag features for 1, 2, 3, 7 and 14 periods, and `make_features()`. The current training scripts import these helpers but do not call them. They train on the columns already present in the input CSV.

### Training preprocessing

- **Feature matrix.**
  - Columns are lower-cased, and `date` and all non-numeric columns are dropped.
  - `close` is moved to index 0 and the matrix is cast to `float32`.
  - Because `symbol` and `exchange` are dropped, a training CSV should contain a single series.
- **Length check.** A series shorter than `window_size + steps_ahead` is skipped.
- **Scaling.** `scale_data_train_only()` fits a `MinMaxScaler` only on the rows used by training sequences and then transforms the full array. Validation and test values may therefore fall outside `[0, 1]`, which is expected. When an existing model is fine-tuned, its saved scaler is reused.
- **Windowing.** `create_sequences()` produces tensors of shape `(samples, window_size, features)` and targets of shape `(samples, steps_ahead)`.
- **Splitting.** `split_data()` splits chronologically into train, validation and test (`val_size`, `test_size`) with no shuffling. `model.fit(..., shuffle=False)` keeps batches in time order as well.
- **Augmentation.** A copy of the training windows with Gaussian noise (σ = 0.01) is appended to the training set.

### Forecast journal and downstream consumer

The forecast producer (NEXGEN) and the downstream consumer run on different machines. Forecasts are not exchanged through either side's code repository.

```text
run_forecast_cycle.py
  ├── fresh SOL/USDT 15m OHLCV (configs/forecast_sol_15m_pipeline.yaml)
  ├── forecast_producer.py (inference only)
  └── state/forecast-journal/forecasts.jsonl
        │ append + fsync + exclusive file lock
        ▼
publish_forecast_journal.py
  └── private forecast-journal repository, branch "journal"
        │ fast-forward Git delivery
        ▼
consumer read-only clone
  └── state/forecasts.jsonl symlink → validator → skill test vs persistence
```

- **Single writer.** The producer is the only writer.
- **Timing.**
  - Inference must finish within 10 minutes of the input bar closing.
  - A commit is published right after each successful 15m inference.
  - The consumer only runs `pull --ff-only`, every minute, and records the first-seen commit SHA with its own UTC time.
- **Append-only history.** History rewrites, backfills, force-pushes and automatic merges or rebases are forbidden.
- **Use of short horizons.** The 15- and 30-minute horizons serve only as timing signals. They are not a basis for choosing the width of a liquidity-provision range.
- **Record format.** Each record follows contract v1: `v`, `ts` (the timestamp of the last fully closed input bar, never the time the process ran), `model`, `asset`, `quote`, `timeframeMin`, `windowBars`, `executionPrice` and `steps[{h, price}]`.
- **Refusals.** The producer refuses to append when:
  - inference fails
  - the CSV does not contain the confirmed asset/quote, or mixes exchanges and `--exchange` is not given
  - there is no fully closed bar
  - more than 10 minutes have passed since the last bar closed (checked before and after inference)
  - a prediction contains NaN/inf or a non-positive price
  - an equal or newer record already exists for the same model series
  - the existing JSONL contains a corrupted line
- **Durability.** Appends run under an exclusive file lock with `flush` and `fsync`. The producer never edits or rewrites the journal.

Deployment and operational commands are in [docs/runbooks/forecast_journal_producer.md](docs/runbooks/forecast_journal_producer.md).

### Main artifacts

- **Raw CSV:** `pipeline/base_data/`; research datasets go to `data/<dataset>/`.
- **Merged dataset:** `pipeline/full_data_base/merged.csv`.
- **Per-user session dataset:** `sessions/<telegram_id>/<timestamp>/full_data_base/merged.csv`.
- **Plots:** `pipeline/plot/`, `sessions/<telegram_id>/<timestamp>/plots/` and `models/saved_models/**`.
- **Saved models:** `models/saved_models/{lstm,gru,hybrid}_saved_model/<bucket>/`.
- **Volatility baseline:** `models/saved_models/volatility_har/<stem>_har_h<horizon>.json`.
- **Reports:**
  - `SUMMARY_REPORT.md` (generated)
  - session `SUMMARY.txt`
  - backtest JSON
  - `logs/validation_gate_report.json`
- **Journals:**
  - `logs/model_staging_journal.jsonl`
  - `state/forecasts.jsonl`, or `state/forecast-journal/forecasts.jsonl` when publishing

## Configuration

### `pipeline/pipeline_config.yaml` and `configs/*_pipeline.yaml`

These YAML files configure the data pipeline. Relative paths resolve against the directory of the config file, so dedicated configs can write to `data/<dataset>/` or `state/forecast_sol_15m/`.

| Group | Keys |
| --- | --- |
| Paths | `base_path`, `data_path`, `plot_path`, `log_file`, `merged_file`, optional `archive_path` |
| Markets | `symbols` (format `LTC/USDT`), `exchanges` (a list, or a dict of switches) |
| Period | `collection_mode` (`auto`, `full`, `days`, `hours`, `custom`), `history_days`, `delta_days`, `delta_hours`, `custom_start`, `custom_end` |
| Fetching | `timeframe`, `limit`, `async.base_delay`, `async.max_concurrent_requests`, `async.max_retries`, `async.request_timeout` |
| Notifications | `alerts.telegram_enabled`, optional `alerts.telegram_connect_timeout` / `alerts.telegram_read_timeout` |
| Output | `output_format` (`csv`, `json`, `both`), `use_cache` |
| Bot | `pipeline_python`, optional menu lists (`available_symbols`, `available_exchanges`, `available_timeframes`, `available_history_days`), `max_symbols_per_user` |

### `configs/config.yaml`

The file has a `.yaml` extension, but its body is JSON with a commented-out YAML prologue. The training scripts read it with `yaml.safe_load`, which accepts JSON. The GPU configs (`config_gpu_sol*.yaml`) use plain YAML and are read the same way.

| Group | Keys |
| --- | --- |
| `global_config` | `val_size`, `test_size`, `steps_ahead`, `window_size`, `enable_logging`, `log_errors_separately`, `retrain_existing`, optional `target` (`level` or `log_return_pct`) and `return_scale` |
| `exchanges`, `symbols` | Switches read by the Keras Tuner and legacy scripts. The current training scripts take their data from CSV files. |
| `model_config` | `num_layers`, `units`, `dropout_0`, `batch_size`, `epochs`, `validation_split`, `patience_early_stop`, `patience_reduce_lr`, `reduce_lr_factor`, `lr`, `min_lr` |
| `paths` | Relative paths for the error and training logs and the model search root |

### `.env`

This is the secrets file in the project root. It holds Telegram tokens and chat/admin IDs and must never be committed. `.env.example` is the template.

| Variable | Used by |
| --- | --- |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_ID` | Telegram bot |
| `TELEGRAM_TOKEN`, `CHAT_ID` | Collector notifications, `utils/telegram_utils.py` (training scripts, auto-retrain) |
| `TELEGRAM_EXTRA_CHAT_ID` | Collector; the bot sets it to the session owner's ID |
| `DOTENV_PATH` | Bot: explicit `.env` location (otherwise the nearest `.env` searching up from the working directory) |
| `PIPELINE_CONFIG`, `SESSION_ROOT`, `PIPELINE_SEND_DATASET`, `PIPELINE_PYTHON` | Collector/bot runtime overrides |

## Telegram subsystem

The Telegram subsystem lives in `pipeline/bot_pipeline_control.py`. It does the following:

- runs the aiogram bot
- reads `.env`, `pipeline_config.yaml` and `pipeline/access.json`
- enforces the role model:
  - `admin`: the `TELEGRAM_ADMIN_ID` user plus any granted admins
  - `operator`: can run personal sessions
  - `model`: can run sessions and download datasets
  - `viewer`: read-only panel
  - guests: no access
- **Guest flow:** a guest can see their Telegram ID and submit an access request.
- **Admin flow:** admins can list participants and requests, grant or change roles (`/allow`, `/role`) and remove users (`/deny`).
- **Session flow:** `/menu`, `/my_new`, `/my_config`, `/my_run`, `/my_status`, `/my_stop`, `/my_log`, `/my_report`, `/my_reset`, `/download`.
- **Collector launch:** the bot runs `pipeline/crypto_pipeline_v4.3.py` as a subprocess with an isolated per-session config. It passes `PIPELINE_CONFIG`, `SESSION_ROOT`, `TELEGRAM_EXTRA_CHAT_ID` and `PIPELINE_SEND_DATASET` through the environment.
- **Stopping:** the bot sends SIGINT, waits, then sends SIGTERM. `/my_reset` force-resets a stuck process.
- **Isolation:** each session's results live in `sessions/<telegram_id>/<timestamp>/`.
- **Delivery:** logs, reports, plots and the merged dataset are sent according to role.
- **Logging and limits:** each user's activity is logged to `pipeline/logs/user_activity/`, and callbacks are rate-limited.

Key files:

- `pipeline/bot_pipeline_control.py`
- `pipeline/access.json` (git-ignored; template: `pipeline/access.json.example`)
- `pipeline/pipeline_config.yaml`
- `.env`
- `sessions/`
- `pipeline/logs/`

The bot manages the collector as an external process, so the stability of paths, configs and access rights directly affects user sessions.

## ML subsystem

The ML subsystem consists of the training scripts, the shared utilities and the saved artifacts.

### Training scripts

- `models/train_lstm.py`: bidirectional LSTM model.
- `models/train_gru.py`: bidirectional GRU model.
- `models/train_both.py`: hybrid BiLSTM → BiGRU model.
- `models/train_gru11.py`: archived Keras Tuner variant of the GRU script. It is not called anywhere and has not been updated for the new targets.
- `models/LSTM2v4.py`: older monolithic experimental script.
- `models/ktuner_visualize.py`: plots Keras Tuner results.

**CLI.** Each script accepts `--config` (default `configs/config.yaml`) and `--file`. `--file` takes a CSV file or a folder. Without it, the script uses `data/data_base/`. When training from a folder, files that were already trained are moved to the model type's `ignored/` directory.

**Retraining.** If an active bundle for the same stem already exists, it is loaded. It is retrained only when `retrain_existing` is true.

### Model design

All three scripts build the same head on top of their recurrent block:

```text
Input(window_size, features)
  → recurrent block
      LSTM:   Bidirectional(LSTM(units, return_sequences)) → Dropout
      GRU:    Bidirectional(GRU(units, return_sequences))  → Dropout
      Hybrid: Bidirectional(LSTM) → Dropout → Bidirectional(GRU) → Dropout
  → LayerNormalization
  → MultiHeadAttention(num_heads=2, key_dim=units), self-attention
  → Dropout → Dense(units, ReLU) → GlobalAveragePooling1D
  → Dense(steps_ahead, float32)
```

- **Regularization:** L2 (1e-3) kernel regularization on the recurrent layers, plus dropout (`dropout_rate`, default 0.3).
- **Optimization:** MSE loss with an MAE metric. Adam is wrapped in a `LossScaleOptimizer` under a global `mixed_float16` policy, and the output layer stays `float32`.
- **Callbacks:** `EarlyStopping` (`patience_early_stop`, best weights restored) and `ReduceLROnPlateau` (`patience_reduce_lr`, `reduce_lr_factor`, `min_lr`).
- **Units:** `configs/config.yaml` sets 64. The script fallback is 256.
- **Attention weights** are captured and plotted for interpretability.
- **Evaluation:** after training, each script compares the model with persistence on the test split. It reports MAPE, accuracy (100 − MAPE) and directional accuracy, and warns when the model does not beat persistence.
- **Test metrics:** MAE, RMSE, MSE, R², explained variance, accuracy and MAPE.

### Prediction targets

`utils/targets.py` defines two target formulations. `global_config.target` selects one of them.

- **`level`** (legacy and default): the scaled absolute `close` for each of the next `steps_ahead` bars.
- **`log_return_pct`**: `y[k] = return_scale · ln(close[t + k + 1] / close[t])`, where `t` is the last bar of the input window. Prices are reconstructed as `entry · exp(y / return_scale)`.

Level and return models are incompatible, so a metadata sidecar `<stem>_<type>_target.json` is saved next to each model. It records the target, window, horizon, `return_scale` and the hyperparameters that matter for comparability (`lr`, `dropout_rate`, `batch_size`, `units`, `num_layers`). A model without a sidecar is treated as a legacy level model. The validation gate warns when the baseline and candidate were trained with different hyperparameters.

### Shared utilities

- `utils/features.py`: preprocessing, lag features and technical indicators.
- `utils/prep_utils.py`: `scale_data`, `scale_data_train_only`, `augment_data`, `create_sequences` and chronological splits.
- `utils/data_utils.py`: CSV/parquet loading, NaN/inf cleaning, chunked processing of large files, and a facade that re-exports the `features` and `prep_utils` functions so preprocessing has a single source.
- `utils/targets.py`: target definitions, conversions and metadata sidecars.
- `utils/volatility.py`: log returns, realized volatility, HAR features and fit, and price bands.
- `utils/metrics.py`: accuracy (100 − MAPE), MAPE, MSE, R² and explained variance.
- `utils/plots.py`: loss, attention and prediction-vs-actual plots.
- `utils/telegram_utils.py`: synchronous Telegram message sending.
- `utils/logger.py`: event logging to the console, files and, optionally, Telegram.

### Model artifacts and lifecycle

Saved models are grouped into buckets by accuracy (100 − MAPE on the test split):

- `below_0%`
- `0-25%`
- `25-50%`
- `50-75%`
- `75-95%`
- `95-100%`

Each bucket directory may also contain `staging/` and `archive/`. Each model type directory also has an `ignored/` directory for input files that were already trained.

The bucket reflects price-level error, not skill. A persistence forecast on 15-minute data reaches about 99.65% "accuracy" and would land in `95-100%`.

Each model is saved as an active bundle under a stable name. The `<stem>` is the input CSV file name without its extension:

```text
<bucket>/<stem>_<type>.keras                      # active model (what predict.py serves)
<bucket>/<stem>_<type>_scaler.pkl                 # fitted MinMaxScaler
<bucket>/<stem>_<type>_target.json                # target metadata sidecar
<bucket>/<stem>_log_<timestamp>.txt               # metrics and settings of the run
<bucket>/<stem>_train_loss_<timestamp>.png
<bucket>/<stem>_prediction_vs_actual_<timestamp>.png
<bucket>/<stem>_attention_<timestamp>.png
<bucket>/staging/<stem>_<type>.keras              # candidate awaiting the validation gate
<bucket>/archive/<stem>_<type>_<timestamp>.keras  # history, read-only
```

When a model is saved, the previous model and scaler with the same name are moved to `archive/`, whether they were in another bucket or in the same one. Training logs and plots for the same stem in other buckets are archived as well.

## Validation gate and staging

A training script saves a new model as active immediately. Without a gate, an unvalidated model would start serving inference right away. `controller/model_staging.py` and `controller/validation_gate.py` put this transition back under control.

**Staging guarantees** (`model_staging.py`):

- Nothing is deleted. Files are only moved, so every operation is reversible.
- The model, scaler and metadata always move together, and an incomplete pair is never created.
- An existing destination is never overwritten.
- Each operation is planned and validated in full before it runs. If any step fails, the moves already made are rolled back.
- Every mutation is appended to `logs/model_staging_journal.jsonl`.
- Rejected and demoted versions are archived with a `rejected_` or `demoted_` tag in front of the timestamp. They are never chosen as the default rollback target and can only be restored with an explicit `--timestamp`.

**Gate criteria** (`validation_gate.py`). Both models are scored on the same holdout (`--test-size`, default 0.1), using the first forecast step. The candidate must not be worse than the baseline on any of these:

- MAPE
- RMSE
- R²
- directional accuracy

Each criterion has an explicit tolerance flag (`--max-mape-regression-pct`, `--max-rmse-regression-pct`, `--max-r2-drop`, `--min-directional-accuracy-delta`), and all of them default to 0. Persistence metrics and skill against persistence are always computed and reported. `--require-beat-persistence` makes beating persistence mandatory. Exit codes: `0` PASS, `2` REJECT, `3` verdict computed but artifacts could not be moved, `1` error.

**Automatic flow** (`auto_retrain.py`, gate enabled by default):

1. Take a snapshot of the active bundles.
2. Train.
3. Diff the snapshots to find the bundles that changed.
4. Move each new model to staging.
5. Restore the baseline from the archive.
6. Run the gate with `--apply`.

If no archived baseline exists, the result is `NO_BASELINE`: the candidate stays in staging, and promoting it is left to a human. The procedures are in [docs/runbooks/validation_gated_retrain.md](docs/runbooks/validation_gated_retrain.md).

## Inference, backtesting and reporting

**Inference: `predict.py` and `predict_gui.py`.**

- They discover active bundles and skip `staging/` and `archive/`. Model IDs have the form `<model_type>:<bucket>:<stem>`.
- They infer the window and horizon from the model's shapes.
- They validate:
  - the scaler's feature count and names
  - the model's input and output shapes
  - finite, positive predictions
- They handle both target types and write a JSON result.
- See [docs/runbooks/inference_only.md](docs/runbooks/inference_only.md).

**Backtest: `backtest.py`.**

- It validates the dataset, scaler and model shapes.
- It predicts on the last 10% of sequences.
- It simulates a long-only trader:
  - trades are based on the forecast edge over the current close (`--signal-threshold-pct`)
  - commission and slippage are applied
  - equity is marked to market for drawdown
- It writes a trade plot and a JSON report that includes the git commit, inputs, schema, metrics and trade history.
- It rejects log-return models explicitly instead of producing meaningless trades.

**Reports: `generate_report.py` and `generate_report_V_2.py`.** They scan the repository for `*_log_*.txt` training logs and prediction/attention plots and build `SUMMARY_REPORT.md`. The V2 script adds MSE, R² and explained variance columns.

## Volatility forecasting

Price direction on 15-minute bars proved unpredictable (see [docs/runbooks/return_target_training.md](docs/runbooks/return_target_training.md)). Volatility, by contrast, clusters: the autocorrelation of |r| stays positive over lags of 1 to 192 bars. `utils/volatility.py` and `controller/volatility_baseline.py` implement the following:

- **Target:** log realized volatility over the next `horizon` bars, `RV = sqrt(Σ r²)` with `r = ln(close[i] / close[i-1])`.
- **Naive benchmark:** "the same as the previous `horizon` bars".
- **HAR model:** past RV at three scales (`horizon`, `2·horizon` and `4·horizon` bars), fitted by linear regression. The model is four numbers, stored as JSON.
- **Band evaluation:** a p10–p90 price band from the forecast, `entry · exp(±z · RV)` with z = 1.2816, together with its actual coverage and mean width.

The results are in [docs/runbooks/volatility_forecast.md](docs/runbooks/volatility_forecast.md). HAR beats the naive benchmark in all tested SOL configurations.

## Deployment

```text
[VPS]  ←→  [Telegram]
  ├── Telegram bot 24/7 (systemd, pipeline/.venv)
  ├── data collection (CCXT pipeline)
  └── data/ with accumulated CSV files

        ↕ rsync (data down, models up)

[GPU host, on demand]
  └── deploy/Dockerfile.ml → LSTM/GRU/hybrid training
```

- **`deploy/nexgen-bot.service`** is the systemd unit for the bot. It uses the absolute path of the pipeline interpreter, `EnvironmentFile=.env`, restarts on failure, logs to journald and is hardened with `NoNewPrivileges` and `PrivateTmp`.
- **`deploy/vps_setup.sh`** sets up an Ubuntu 22.04/24.04 server. It creates a service user, clones the repository, builds `pipeline/.venv` and installs the unit. Secrets are copied separately.
- **`deploy/Dockerfile.ml`** builds a training image based on `tensorflow/tensorflow:2.21.0-gpu`. Data, models and logs are mounted as volumes, and no secrets are baked in.
- **Forecast cycle:** runs from cron on the producer machine (see the forecast journal runbook).

Details: [deploy/README.md](deploy/README.md).

## Testing

The project uses the stdlib `unittest` with 114 tests in 9 modules. Run it from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

**Covered:**

- the forecast producer, cycle runner and journal publisher (some tests use temporary Git repositories)
- inference windows in `predict.py`
- the validation gate and model staging
- train-only scaling
- target conversions
- volatility utilities

**Not covered:**

- model training
- data collection
- the backtest trading loop
- the Telegram bot

## Known risks and technical debt

### 1. Runtime artifacts next to source code

Datasets, models, plots, logs, sessions and archives live inside the project tree. They are git-ignored, but they still make navigation, backup and moving the project harder. The target structure below separates them.

### 2. Config keys that the training scripts do not apply

- **`lr`.** The optimizer is created as `Adam()` without a learning rate, so the effective rate is Adam's default of 0.001, whatever the config says. The configured value is still written to the metadata sidecar. `configs/config_gpu_sol*.yaml` sets `lr: 0.001` to match the rate actually used.
- **`dropout_0`.** `configs/config.yaml` defines `dropout_0`, but the scripts read `dropout_rate` (default 0.3).
- **`num_layers`.** The architecture has a fixed recurrent block per model type, so `num_layers` is accepted but not used.
- **`validation_split`.** It is ignored in favor of explicit chronological splits.

### 3. Two indicator implementations

The collector (`add_indicators` in `pipeline/crypto_pipeline_v4.3.py`) and `utils/features.py` compute indicators independently, with different column names and parameters. The training scripts use the collector's columns. Any future pipeline that calls `make_features()` would produce a different feature set, and saved scalers and models would no longer match it.

### 4. Level-only backtest

`backtest.py` supports price-level models only. `validation_gate.py` and `predict.py` handle both target types.

### 5. Collector `.env` lookup

When `alerts.telegram_enabled` is true, the collector expects `.env` one directory above the repository root. A missing file stops the run. Telegram-free configs, such as `pipeline/test_run/test_config.yaml` and the `configs/*_pipeline.yaml` files, are not affected.

### 6. Legacy and experimental scripts

The following scripts are kept for reference and are not part of the supported workflow:

- `models/LSTM2v4.py`
- `models/train_gru11.py`
- `models/ktuner_visualize.py`
- `betatest/`

They are not maintained and have not been updated for the newer prediction targets or the validation gate.

### 7. Training CSV layout

The training scripts drop non-numeric columns. A merged dataset that contains several symbols or exchanges would therefore be flattened into one series without any warning. Training data should come from single-symbol, single-exchange datasets, such as those produced by the `configs/sol_*_pipeline.yaml` configs.

## Target structure (future direction)

The project could eventually adopt the following layout. This section describes a direction, not the current state.

```text
.
├── src/
│   └── crypto_ml/
│       ├── pipeline/
│       ├── bot/
│       ├── models/
│       ├── features/
│       ├── backtesting/
│       ├── reporting/
│       └── common/
├── configs/
│   ├── pipeline.yaml
│   └── training.yaml
├── scripts/
│   ├── collect_data.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   ├── train_hybrid.py
│   ├── backtest.py
│   └── run_bot.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── artifacts/
│   ├── models/
│   ├── scalers/
│   ├── plots/
│   └── reports/
├── runtime/
│   ├── sessions/
│   ├── logs/
│   └── archives/
├── tests/
├── docs/
├── requirements.txt
├── .env.example
└── README.md
```

In this layout, each directory has one job:

- **`src/crypto_ml/`:** importable code only.
- **`scripts/`:** thin CLI wrappers.
- **`configs/`:** portable configs with relative paths only.
- **`data/`:** datasets split into raw, processed and external.
- **`artifacts/`:** models, scalers, plots and reports.
- **`runtime/`:** logs, sessions and temporary archives.
- **`docs/`:** architecture, runbooks, and user and admin docs.
- **`tests/`:** fast unit and smoke checks for preprocessing, config parsing and command wiring.

## Stabilization roadmap

Completed:

1. `.gitignore` covers ML and runtime artifacts, secrets and access files.
2. `.env.example` provides a template without secrets.
3. `requirements-ml.txt` and `requirements-pipeline.txt` are pinned from the real environments.
4. Config and report paths are relative. The report scripts resolve the project root with `Path(__file__).resolve()`.
5. Preprocessing has a single source: `utils/data_utils.py` is a facade over `features` and `prep_utils`.
6. The scaler is fitted on training rows only.
7. Retraining is validation-gated, with reversible artifact staging.

Next:

1. Apply the `lr`, `dropout` and `num_layers` config keys in the training scripts.
2. Unify the indicator implementations.
3. Add log-return support to `backtest.py`.
4. Only then move the code into the package structure above.

## Decision-support prototype (QCI)

`qci/qci_v_0.01.py` is a standalone, rule-based decision-support prototype. It scores the hypotheses `BUY`, `SELL`, `WAIT` and `RISK_OFF` from input signals and converts the scores to probabilities with a softmax. It falls back to a safe action when its confidence is low.

No runtime component imports it. The pipeline, the bot, the training scripts and the backtest do not depend on it.

A more elaborate decision engine (QCI v0.2) was prototyped once and intentionally withdrawn. It was not meant to forecast or execute trades on its own. Its goal was explainable decision support: combining ML forecasts, indicators, market regime and the history of past decisions into `BUY` / `SELL` / `HOLD` / `WAIT` / `RISK_OFF` recommendations with a risk assessment and reasons. It is a historical direction, not current behavior, and nothing in the runtime depends on it.

Any future decision layer should be added as a separate, explicitly integrated and tested component. If it uses a persistent queue, that queue needs crash-safe atomic rewrites, exclusive cross-process locking and recovery of stale temporary files. Otherwise records could be processed twice after a crash.

## Summary

The project has working subsystems for data collection, Telegram control, ML training, model lifecycle management, inference, backtesting, reporting and a published forecast journal. Its main strength is evaluation discipline: every model is measured against persistence, and retraining is validation-gated. Its main architectural debt is that code, configs and runtime artifacts live side by side, along with the config and indicator inconsistencies listed above.
