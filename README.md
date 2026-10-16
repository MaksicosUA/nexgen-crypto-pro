# Nexgen Crypto Pro

An end-to-end research and operations toolkit for short-horizon crypto forecasting. It covers asynchronous multi-exchange data collection, feature engineering, recurrent neural networks with attention, a validation-gated model lifecycle, inference, backtesting and remote control through Telegram.

> The models only produce forecast signals. Nothing in this repository places orders.

---

## Highlights

- **Async market-data pipeline.** It pulls OHLCV data from 8 exchanges (Binance, Bybit, KuCoin, OKX, Gate.io, MEXC, Coinbase, Bitget) through CCXT `async_support`, with a concurrency limit, retries and an adaptive request delay. It checks for gaps and outliers, patches gaps from an alternate exchange and computes technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, CCI, Williams %R and momentum.
- **Telegram control plane.** An aiogram 3 bot has role-based access (`admin`, `operator`, `model`, `viewer`) and gives each user an isolated session. It runs the collector as a subprocess and handles start, graceful stop, CPU/RAM status, logs, plots and ZIP export.
- **Neural forecasters.** There are three model types: a bidirectional LSTM, a bidirectional GRU and a hybrid BiLSTM → BiGRU. Each one ends in multi-head self-attention. Training uses mixed precision (FP16), fits the scaler on training rows only and splits data strictly in time order. The target is either the price level or the log return.
- **Honest evaluation.** Every training run and every validation compares the model with a naive persistence forecast. Directional accuracy and skill against persistence are reported next to MAPE, RMSE and R².
- **Validation-gated retraining.** A retrained model goes to staging first. It is compared with the current model on the same holdout and is promoted only if it is not worse. Artifact moves are planned up front, journaled and reversible. They never overwrite or delete a file.
- **Inference, backtesting and a forecast journal.**
  - An inference-only CLI and a Tk GUI check the data and model shapes strictly before predicting.
  - A long-only backtest accounts for commission, slippage and drawdown.
  - An append-only, file-locked JSONL forecast journal is published to a separate Git repository with fast-forward-only pushes, so forecasts can be skill-tested out of sample.
- **Volatility forecasting.** A HAR baseline predicts realized volatility, and its prediction bands are checked against actual price coverage.
- **114 unit tests** use the stdlib `unittest`. They do not train models or call the network.

---

## Results and findings

The project is a research log as much as a codebase. The main findings are summarized below and documented in detail in the runbooks. All numbers come from runs on Binance SOL 15-minute data.

### Price-level forecasting does not beat a naive baseline

Source: [docs/runbooks/return_target_training.md](docs/runbooks/return_target_training.md)

| Data | Bars | Model MAPE | Persistence MAPE | Directional accuracy |
| --- | --- | --- | --- | --- |
| SOL/USDT, 6 years | 210,447 | 12.58% | 0.25% | 48.93% |
| SOL/USDC, 2.6 years | 91,976 | 1.97% | 0.20% | 48.27% |

On 15-minute bars, the "price will not change" forecast already reaches an R² of about 0.999. A low MAPE or a high "accuracy" therefore proves nothing on its own. This is why skill against persistence is reported everywhere in the project.

### Return targets reveal no directional signal in OHLCV-derived features

With log-return targets, the GRU converged to a constant prediction. That is the MSE-optimal answer when there is no signal. A linear-regression check over horizons from 30 minutes to 48 hours gave a negative out-of-sample R² and 48–50% directional accuracy at every horizon.

### Volatility is predictable

Source: [docs/runbooks/volatility_forecast.md](docs/runbooks/volatility_forecast.md)

A three-feature HAR model of log realized volatility beat the naive forecast in all six tested combinations. The naive forecast assumes volatility stays the same as in the last window.

| Dataset | Horizon | R² naive | R² HAR | p10–p90 band coverage |
| --- | --- | --- | --- | --- |
| SOL/USDC, 2.6 years | 2 h | 0.0136 | 0.2257 | 82.2% |
| SOL/USDC, 2.6 years | 12 h | 0.0430 | 0.2172 | 85.1% |
| SOL/USDC, 2.6 years | 24 h | 0.1602 | 0.2283 | 85.1% |
| SOL/USDT, 6 years | 2 h | 0.1457 | 0.3209 | 81.3% |
| SOL/USDT, 6 years | 12 h | 0.1397 | 0.3222 | 82.5% |
| SOL/USDT, 6 years | 24 h | 0.1674 | 0.3240 | 83.1% |

In a separate check, adding all 20 of the project's technical indicators to HAR made it worse (R² 0.348 → 0.316). Absolute R² depends on which slice is used for testing, so only the naive and HAR values from the same run should be compared. These results apply only to SOL on 15-minute bars. At 24-hour horizons there are only 152–349 independent windows, so the confidence intervals are wide.

---

## Architecture

```text
Exchanges ── CCXT async ──► pipeline/crypto_pipeline_v4.3.py ◄── pipeline/bot_pipeline_control.py
                              ├─ fetch OHLCV → base_data/*.csv         (Telegram bot, per-user sessions)
                              ├─ fix gaps, patch from alternate exchange, quality checks
                              ├─ technical indicators
                              └─ full_data_base/merged.csv + plots + Telegram summary
                                        │
                                        ▼
                   models/train_lstm.py · train_gru.py · train_both.py
                              ├─ numeric features, close first; MinMaxScaler fitted on train rows only
                              ├─ sliding windows (window_size bars) → steps_ahead targets
                              ├─ chronological train/val/test split, no shuffling
                              ├─ bidirectional RNN + multi-head attention, FP16
                              └─ models/saved_models/<type>_saved_model/<bucket>/<stem>_<type>.keras
                                 (+ _scaler.pkl and _target.json metadata)
                                        │
        ┌───────────────────────┬───────┴──────────────┬──────────────────────────┐
        ▼                       ▼                      ▼                          ▼
 validation_gate.py +    predict.py /           backtest.py               run_forecast_cycle.py →
 model_staging.py        predict_gui.py         (virtual trader,          forecast_producer.py →
 (staging → PASS/REJECT  (inference only)       JSON report)              publish_forecast_journal.py
  → active / archive)
```

[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) describes the subsystems, data flow, model design, storage layout and deployment in detail.

### Model design

| Model | Layers |
| --- | --- |
| LSTM (`models/train_lstm.py`) | Bidirectional LSTM → Dropout → LayerNorm → MultiHeadAttention (2 heads) → Dropout → Dense(ReLU) → GlobalAveragePooling → Dense(`steps_ahead`) |
| GRU (`models/train_gru.py`) | Same as the LSTM model, with a bidirectional GRU |
| Hybrid (`models/train_both.py`) | Bidirectional LSTM → Dropout → Bidirectional GRU → Dropout → the same attention head |

- **Training setup:** L2 (1e-3) kernel regularization on the recurrent layers, MSE loss, Adam inside a loss-scale optimizer and a global `mixed_float16` policy.
- **Callbacks:** early stopping with best-weight restore and `ReduceLROnPlateau`.
- **Data handling:** no shuffling, plus Gaussian-noise augmentation of the training windows.
- **Defaults from `configs/config.yaml`:** 64 units, window 30, horizon 2 steps, batch 128, up to 100 epochs with early-stopping patience 20, 16% validation and 16% test.

---

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.12 |
| ML | TensorFlow 2.21, Keras 3.14, scikit-learn 1.8, NumPy, pandas 3 |
| Data collection | CCXT 4.5 (async), aiohttp |
| Telegram bot | aiogram 3.22, psutil |
| Visualization | Matplotlib, Tkinter (inference GUI) |
| Configuration | YAML/JSON, python-dotenv |
| Operations | systemd, Docker (TensorFlow GPU image), cron |
| Testing | `unittest` |

---

## Repository layout

```text
.
├── pipeline/
│   ├── crypto_pipeline_v4.3.py      # async OHLCV collector: QA, indicators, merge, plots, Telegram summary
│   ├── bot_pipeline_control.py      # Telegram bot: roles, per-user sessions, collector subprocess control
│   ├── pipeline_config.yaml         # default collector config
│   ├── access.json.example          # template for bot roles (the real access.json is git-ignored)
│   └── test_run/test_config.yaml    # safe collector config with Telegram disabled
├── models/
│   ├── train_lstm.py                # BiLSTM + attention
│   ├── train_gru.py                 # BiGRU + attention
│   ├── train_both.py                # hybrid BiLSTM → BiGRU + attention
│   ├── ktuner_visualize.py          # Keras Tuner result plots (experimental)
│   ├── train_gru11.py               # archived Keras Tuner experiment (not maintained)
│   └── LSTM2v4.py                   # archived early monolithic script (not maintained)
├── controller/
│   ├── run_models.py                # sequential training runner
│   ├── auto_retrain.py              # retrain on the newest dataset, then run the validation gate
│   ├── validation_gate.py           # baseline vs candidate comparison on one holdout
│   ├── model_staging.py             # journaled active / staging / archive artifact moves
│   ├── forecast_producer.py         # inference → one contract-v1 JSONL record
│   ├── run_forecast_cycle.py        # collect → infer → publish (optional), under a file lock
│   ├── publish_forecast_journal.py  # validated fast-forward publication of the journal
│   └── volatility_baseline.py       # HAR realized-volatility baseline
├── utils/
│   ├── features.py                  # indicator and lag-feature helpers
│   ├── prep_utils.py                # train-only scaling, windowing, chronological splits, augmentation
│   ├── data_utils.py                # data I/O and a facade over features/prep_utils
│   ├── targets.py                   # level vs log-return targets, metadata sidecars
│   ├── volatility.py                # realized volatility, HAR features/fit, price bands
│   ├── metrics.py                   # accuracy (100 − MAPE), MAPE, MSE, R², explained variance
│   ├── plots.py                     # loss, attention and prediction plots
│   ├── logger.py                    # console/file/Telegram event logger
│   └── telegram_utils.py            # synchronous Telegram message helper
├── configs/
│   ├── config.yaml                  # default training config (JSON body, read with yaml.safe_load)
│   ├── config_gpu_sol.yaml          # GPU training config for SOL datasets (YAML)
│   ├── config_gpu_sol_returns.yaml  # same config with a log-return target
│   ├── sol_*_15m_*_pipeline.yaml    # collector configs for the SOL 15m research datasets
│   └── forecast_sol_15m_pipeline.yaml  # collector config for the forecast cycle
├── tests/                           # 114 unittest cases in 9 modules
├── docs/runbooks/                   # operational runbooks
├── deploy/                          # systemd unit, VPS setup script, GPU training Dockerfile
├── scripts/with_gpu.sh              # makes pip-installed CUDA libraries visible to TensorFlow
├── betatest/                        # early backtest and GUI launcher (experimental)
├── qci/qci_v_0.01.py                # standalone decision-support prototype (not wired in)
├── backtest.py                      # backtest of a saved price-level model
├── predict.py                       # inference-only CLI
├── predict_gui.py                   # inference-only Tk GUI
├── generate_report.py               # Markdown summary of training logs
├── generate_report_V_2.py           # same, with an extended metrics table
├── requirements-ml.txt              # pinned ML environment (.venv)
├── requirements-pipeline.txt        # pinned pipeline/bot environment (pipeline/.venv)
├── requirements.txt                 # combined reference list (not a lock file)
├── .env.example                     # environment variable template (no secrets)
└── PROJECT_ARCHITECTURE.md
```

The runtime directories are git-ignored and created on first use: `data/`, `logs/`, `state/`, `sessions/`, `models/saved_models/`, and `pipeline/{base_data,full_data_base,plot,archive,logs}/`.

---

## Getting started

### Prerequisites

- Python 3.12
- Linux (tested on Ubuntu/Debian; the forecast journal relies on `fcntl` file locks)
- An NVIDIA GPU with CUDA is optional. Everything also runs on the CPU.

### Install

```bash
git clone https://github.com/MaksicosUA/nexgen-crypto-pro.git nexgen-crypto-pro
cd nexgen-crypto-pro

# ML environment: training, inference, backtest, reports, tests
python3 -m venv .venv
.venv/bin/pip install -r requirements-ml.txt
# Optional, for hyperparameter search:
# .venv/bin/pip install "keras-tuner>=1.4.0"

# Pipeline environment: data collection and the Telegram bot
python3 -m venv pipeline/.venv
pipeline/.venv/bin/pip install -r requirements-pipeline.txt
```

The two environments are separate on purpose. The long-running bot stays lightweight, and the TensorFlow stack is only installed where ML jobs run. Always call the interpreter explicitly (`.venv/bin/python` or `pipeline/.venv/bin/python`).

### Configure secrets

```bash
cp .env.example .env
cp pipeline/access.json.example pipeline/access.json
```

| Variable | Used by | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | bot | Bot token from @BotFather |
| `TELEGRAM_ADMIN_ID` | bot | Telegram user ID of the primary admin |
| `TELEGRAM_TOKEN`, `CHAT_ID` | collector, training scripts | Notification channel |
| `TELEGRAM_EXTRA_CHAT_ID` | collector | Extra recipient (the bot sets it per session) |
| `DOTENV_PATH`, `PIPELINE_CONFIG`, `PIPELINE_PYTHON`, `SESSION_ROOT`, `PIPELINE_SEND_DATASET` | bot, collector | Optional runtime overrides |

`.env` and `pipeline/access.json` are git-ignored. The user in `TELEGRAM_ADMIN_ID` is always an admin. Other users get roles through the bot (`/allow`, `/role`), and the roles are saved to `pipeline/access.json`.

### Smoke checks

```bash
.venv/bin/python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
pipeline/.venv/bin/python -c "import aiogram, ccxt, psutil; print('pipeline OK')"
.venv/bin/python -c "from utils.features import add_technical_indicators; from utils.prep_utils import create_sequences; print('utils OK')"
```

---

## Usage

### 1. Collect market data

```bash
# Safe first run: Telegram disabled, one pair, one day of data
PIPELINE_CONFIG=pipeline/test_run/test_config.yaml pipeline/.venv/bin/python pipeline/crypto_pipeline_v4.3.py

# Default config (pipeline/pipeline_config.yaml)
pipeline/.venv/bin/python pipeline/crypto_pipeline_v4.3.py

# A research dataset, e.g. SOL/USDC 15m after relisting
PIPELINE_CONFIG=configs/sol_usdc_15m_clean_pipeline.yaml pipeline/.venv/bin/python pipeline/crypto_pipeline_v4.3.py
```

Main collector options:

- `symbols` and `exchanges`
- `timeframe`
- `collection_mode`:
  - `auto`: the last `history_days` days (7 by default)
  - `full`: the entire listing history
  - `days` / `hours`: `delta_days` / `delta_hours`
  - `custom`: `custom_start` to `custom_end`
- `async` limits
- `alerts.telegram_enabled`
- `output_format` (`csv`, `json` or `both`)
- `use_cache`

Relative paths in a config resolve against the config file's directory. The collector writes raw per-exchange CSV files to `base_data/` and the merged dataset with indicators to `full_data_base/merged.csv`.

### 2. Control the collector from Telegram

```bash
pipeline/.venv/bin/python pipeline/bot_pipeline_control.py
```

| Command | Role | Action |
| --- | --- | --- |
| `/start`, `/panel` | any granted role | Control panel (inline keyboard; buttons a role cannot use are locked) |
| `/menu` | operator, model | Session settings: pairs, exchanges, timeframe, history depth |
| `/my_new`, `/my_config` | operator, model | Start a fresh personal session / show its config |
| `/my_run` | operator, model | Launch the collector for the session as a subprocess |
| `/my_status` | operator, model | PID, uptime, CPU and RAM |
| `/my_stop` | operator, model | Stop the session process (SIGINT, then SIGTERM) |
| `/my_log`, `/my_report` | operator, model | Session log / final report |
| `/my_dataset` | model | Download the session's merged dataset |
| `/my_reset` | operator, model | Reset a stuck session process |
| `/download` | operator, model | ZIP of session data, plots and summary |
| `/whoami`, `/id` | anyone | Show your Telegram ID and access level |
| `/admin`, `/allow`, `/deny`, `/role`, `/ops` | admin | Access management |
| `/status`, `/stop`, `/lastlog`, `/plots` | admin | Global process status, stop, log tail, plots |

Admins can use every command. Each session runs in `sessions/<telegram_id>/<timestamp>/` with its own config, data, plots and logs.

### 3. Train models

```bash
.venv/bin/python models/train_gru.py  --config configs/config.yaml --file path/to/dataset.csv
.venv/bin/python models/train_lstm.py --config configs/config.yaml --file path/to/dataset.csv
.venv/bin/python models/train_both.py --config configs/config.yaml --file path/to/dataset.csv

# On a GPU host with pip-installed CUDA libraries
scripts/with_gpu.sh .venv/bin/python models/train_gru.py \
  --config configs/config_gpu_sol_returns.yaml \
  --file data/sol_usdc_15m_clean/full_data_base/merged_sol_usdc_15m_clean.csv
```

- **Input data:**
  - `--file` takes a CSV file or a folder. Without it, every CSV in `data/data_base/` is trained, and files that were already trained are moved to the model type's `ignored/` directory.
  - Every numeric column becomes a feature, with `close` moved to the first position.
  - Non-numeric columns (`timestamp`, `symbol`, `exchange`) are dropped, so use a dataset with a single symbol and a single exchange.
- **Target:** set `global_config.target` to `level` (the default) or `log_return_pct`.
- **Output:**
  - Each run prints a comparison with persistence.
  - It saves `<stem>_<type>.keras`, `<stem>_<type>_scaler.pkl` and a `<stem>_<type>_target.json` metadata sidecar to `models/saved_models/<type>_saved_model/<bucket>/`.
  - The bucket comes from accuracy = 100 − MAPE: `below_0%`, `0-25%`, `25-50%`, `50-75%`, `75-95%` or `95-100%`.
- **Sequential runs:** `controller/run_models.py` runs training scripts one after another. Only the LSTM script is enabled in `MODELS_TO_RUN`.

### 4. Retrain behind a validation gate

```bash
.venv/bin/python controller/auto_retrain.py --models gru,lstm,hybrid
.venv/bin/python controller/auto_retrain.py --models gru --dry-run   # print commands only
```

1. `auto_retrain.py` retrains on the newest CSV in `data/data_base/`.
2. Each new model is moved to `staging/`, and the previous model is restored from the archive.
3. `controller/validation_gate.py` compares the two models on the same holdout. On **PASS** the candidate becomes active. On **REJECT** it is archived.
4. The verdict is sent to Telegram and written to `logs/auto_retrain.log`.

The script runs once per call. Schedule it externally, for example with cron. The manual workflow, tolerances and exit codes are in [docs/runbooks/validation_gated_retrain.md](docs/runbooks/validation_gated_retrain.md).

### 5. Run inference only

```bash
.venv/bin/python predict.py --list-models
.venv/bin/python predict.py --model-id 'gru:95-100%:<stem>' --data path/to/dataset.csv --output prediction.json
.venv/bin/python predict.py --interactive --data path/to/dataset.csv
.venv/bin/python predict_gui.py
```

Model IDs have the form `<model_type>:<bucket>:<stem>`. Models in `staging/` and `archive/` are never offered for inference. See [docs/runbooks/inference_only.md](docs/runbooks/inference_only.md).

### 6. Backtest a saved model

```bash
.venv/bin/python backtest.py \
  --data path/to/dataset.csv \
  --model  "models/saved_models/lstm_saved_model/<bucket>/<stem>_lstm.keras" \
  --scaler "models/saved_models/lstm_saved_model/<bucket>/<stem>_lstm_scaler.pkl" \
  --window 30 --steps 2 \
  --commission 0.001 --slippage-pct 0.05 --signal-threshold-pct 0.2
```

The backtest runs on the last 10% of sequences. It simulates a long-only trader that enters when the forecast edge over the current close exceeds the threshold. It writes a trade plot (`--output`, default `backtest_result.png`) and a JSON report (`--report`, which defaults to the plot path with a `.json` suffix).

`--window` and `--steps` must match the model; the CLI defaults are 60 and 1. Only price-level models are supported.

### 7. Volatility baseline

```bash
.venv/bin/python controller/volatility_baseline.py \
  --data data/sol_usdc_15m_clean/full_data_base/merged_sol_usdc_15m_clean.csv \
  --horizon 96
```

The horizon is given in bars: on 15-minute data, `8` is 2 hours, `48` is 12 hours and `96` (the default) is 24 hours. The script exits with code `0` if HAR beats the naive forecast, `2` if it does not and `1` on error. The fitted model and report are saved as JSON under `models/saved_models/volatility_har/`.

### 8. Forecast journal

```bash
.venv/bin/python controller/run_forecast_cycle.py --journal-model sol_gru_15m_v1
```

The cycle does three things:

1. It collects fresh SOL/USDT 15m data with the pipeline environment.
2. It runs the producer with the ML environment and appends one contract-v1 record to `state/forecasts.jsonl`.
3. If `--publish-repo` is set, it publishes the journal to a separate Git repository.

The deployment details are in [docs/runbooks/forecast_journal_producer.md](docs/runbooks/forecast_journal_producer.md).

### 9. Reports

```bash
.venv/bin/python generate_report.py       # or generate_report_V_2.py for the extended table
```

The report script scans per-run training logs and prediction plots. It writes a summary to `SUMMARY_REPORT.md` in the repository root.

---

## Tests

```bash
# Run from the repository root
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

# A single module
.venv/bin/python -m unittest tests.test_validation_gate -v
```

The suite uses the stdlib `unittest`; pytest is not used. Run it from the repository root, because `tests/` is not a package and imports such as `from controller...` only resolve through the current directory.

The 114 tests take about 2 seconds. They do not train models or access the network. Some tests create temporary Git repositories in system temp directories and leave nothing behind in the project.

| Covered | Not covered |
| --- | --- |
| Forecast producer, cycle runner and journal publisher | Model training |
| Inference windows in `predict.py` | Data collection |
| Validation gate and model staging | The backtest trading loop |
| Train-only scaling | The Telegram bot |
| Return targets and volatility utilities | |

---

## Configuration

| File | Format | Purpose |
| --- | --- | --- |
| `configs/config.yaml` | JSON body, `.yaml` extension, read with `yaml.safe_load` | Training hyperparameters, enabled exchanges/symbols, log paths |
| `configs/config_gpu_sol*.yaml` | YAML | GPU training configs for SOL datasets (level / log-return target) |
| `pipeline/pipeline_config.yaml` and `configs/*_pipeline.yaml` | YAML | Collector settings: symbols, exchanges, timeframe, period, async limits, alerts |
| `.env` | dotenv | Telegram tokens and IDs, runtime overrides |
| `pipeline/access.json` | JSON | Bot roles, participants and access requests |

---

## Deployment

- **Telegram bot:** runs 24/7 on a small VPS as a systemd service (`deploy/nexgen-bot.service`) using `pipeline/.venv`. `deploy/vps_setup.sh` handles installation.
- **Training:** runs on demand on a GPU host from `deploy/Dockerfile.ml`, which is based on the TensorFlow 2.21 GPU image. Data and models are mounted as volumes, and trained models are synced back with `rsync`.

See [deploy/README.md](deploy/README.md).

---

## Documentation

- [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md): subsystems, data flow, model design, storage layout, deployment, known risks
- [docs/runbooks/inference_only.md](docs/runbooks/inference_only.md): running saved models without retraining
- [docs/runbooks/validation_gated_retrain.md](docs/runbooks/validation_gated_retrain.md): staging, gate criteria and rollback
- [docs/runbooks/return_target_training.md](docs/runbooks/return_target_training.md): log-return targets and the negative price-forecasting result
- [docs/runbooks/volatility_forecast.md](docs/runbooks/volatility_forecast.md): the HAR volatility baseline and band coverage
- [docs/runbooks/forecast_journal_producer.md](docs/runbooks/forecast_journal_producer.md): the forecast journal contract and its publication
- [docs/runbooks/bot_live_test.md](docs/runbooks/bot_live_test.md): a step-by-step live test of the Telegram bot
- [deploy/README.md](deploy/README.md): VPS and GPU deployment

---

## Known limitations

- **No demonstrated price edge.** See [Results and findings](#results-and-findings). The forecasters are research artifacts, not a trading system.
- **Accuracy buckets measure price-level error, not skill.** A persistence forecast would land in `95-100%`.
- **Some config keys are not used by the training scripts:**
  - `lr`: the optimizer uses Adam's default learning rate (0.001). The configured value is still recorded in the model metadata.
  - `dropout_0`: the scripts read `dropout_rate`, which defaults to 0.3.
  - `num_layers`: each model type has a fixed recurrent block.
- **The backtest is limited to level models.** `backtest.py` supports price-level models only. `validation_gate.py` and `predict.py` handle both target types.
- **Indicators are implemented twice.** The collector uses `add_indicators` in the pipeline, and `utils/features.py` has its own helpers.
- **The collector's `.env` lookup is off by one directory.** When run on its own with `alerts.telegram_enabled: true`, the collector looks for `.env` one directory above the repository root. Use `telegram_enabled: false` or the test config for local runs.
- **Legacy scripts are kept for reference only.** `models/LSTM2v4.py`, `models/train_gru11.py`, `models/ktuner_visualize.py` and `betatest/` are not maintained and have not been updated for the newer targets and the validation gate.

### Troubleshooting

| Symptom | What to do |
| --- | --- |
| GPU not detected | CPU mode works. If CUDA libraries were installed through pip, wrap the command in `scripts/with_gpu.sh`. Check `nvidia-smi`. |
| NaN predictions under FP16 | The training scripts set `mixed_float16` unconditionally. Switch the global policy to `float32` for that run. |
| Out of memory during training | Reduce `batch_size` or `window_size` in the training config. |
| `ImportError: No module named 'utils'` | Run scripts from the repository root. Entry points add the root to `sys.path`. |
| The model stops improving too early | Increase `patience_early_stop` (20–50). |
| CCXT timeouts | Increase `async.request_timeout` or `async.base_delay`, or reduce the number of exchanges. |
| The bot cannot find its token | Run it from the repository root or set `DOTENV_PATH`. |

---

## Roadmap

- **Richer inputs.** Every current feature is derived from the same price series. Order-book, volume-profile, funding-rate, cross-asset and on-chain data would add information that OHLCV does not contain.
- **Other model objectives.** Try classification objectives and uncertainty estimates (p10/p90 bands) instead of point regression.
- **Wider volatility validation.** Test the HAR baseline on other assets and timeframes, and estimate the cost of band breaches, not just their frequency.
- **A stronger backtest.** Add log-return model support to `backtest.py` and improve its reliability, reporting and risk accounting.
- **Package layout.** Move importable code into a package structure (see [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)).
- **Newer forecasting models.** Evaluate TimesFM, Chronos, PatchTST, N-BEATS/N-HiTS, TFT, XGBoost, LightGBM and CatBoost against the same persistence and HAR baselines.
- **A decision layer.** Design a layer that combines forecasts with risk, uncertainty, market regime and error history.

---

## Contributing

1. Open an issue or describe the intended change before a large refactor.
2. Keep changes focused. Do not mix code, documentation, runtime artifacts and model files in one commit.
3. Do not commit secrets, `.env`, Telegram IDs, private logs or local runtime outputs.
4. Run checks with the matching environment:

   ```bash
   .venv/bin/python -m py_compile backtest.py predict.py
   pipeline/.venv/bin/python -m py_compile pipeline/bot_pipeline_control.py
   .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
   ```

5. Update the documentation when behavior, commands or workflows change.

---

## Security

- Secrets live only in `.env`, which is git-ignored. Use `.env.example` as the template.
- `pipeline/access.json` holds Telegram user IDs and is git-ignored. Use `pipeline/access.json.example` as the template.
- Runtime data, logs, sessions and trained artifacts (`*.keras`, `*.pkl`) are git-ignored.
