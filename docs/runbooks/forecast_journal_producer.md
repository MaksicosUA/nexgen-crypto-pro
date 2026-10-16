# Forecast journal producer — contract v1

The producer runs a saved NEXGEN model and, after a successful inference, appends one JSONL line.

Current scope:

- one confirmed series, SOL/USDT 15m;
- model `gru:95-100%:SOLUSDT_SOLUSDC_15m_30d`;
- use case: paper-first skill history for timing, not for choosing the LP range width;
- default local journal: `state/forecasts.jsonl`.

## Line contract

```json
{"v":1,"ts":"2026-07-11T09:00:00Z","model":"sol_gru_15m_v1","asset":"SOL","quote":"USDT","timeframeMin":15,"windowBars":30,"executionPrice":83.12,"steps":[{"h":1,"price":83.4},{"h":2,"price":83.55}]}
```

`ts` is taken from the timestamp of the last fully closed input bar. The process start time is not used.

The writer refuses to append a line if:

- inference failed;
- the CSV does not contain the confirmed asset/quote;
- the CSV mixes exchanges and `--exchange` is not specified;
- there is no fully closed bar;
- more than 10 minutes have passed since the last bar closed; the deadline is checked both before and after inference;
- the prediction contains NaN/inf or a non-positive price;
- an identical or newer record already exists for the same model series;
- the existing JSONL contains a corrupted line.

The append runs under an exclusive file lock with `flush` + `fsync`. The producer never edits or overwrites the journal.

## P1: a single manual run

First collect a fresh single-symbol/single-exchange CSV. Then:

```bash
.venv/bin/python controller/forecast_producer.py \
  --data state/forecast_sol_15m/full_data_base/merged.csv \
  --model-id 'gru:95-100%:SOLUSDT_SOLUSDC_15m_30d' \
  --journal-model sol_gru_15m_v1 \
  --asset SOL \
  --quote USDT \
  --timeframe-min 15 \
  --exchange binance \
  --journal state/forecasts.jsonl
```

To check without writing:

```bash
.venv/bin/python controller/forecast_producer.py \
  --data state/forecast_sol_15m/full_data_base/merged.csv \
  --model-id 'gru:95-100%:SOLUSDT_SOLUSDC_15m_30d' \
  --journal-model sol_gru_15m_v1 \
  --asset SOL --quote USDT --timeframe-min 15 --exchange binance \
  --dry-run
```

After any retrain, the `--journal-model` value must be changed, for example from `sol_gru_15m_v1` to `sol_gru_15m_v2`.

## P2: regular cycle

The dedicated config `configs/forecast_sol_15m_pipeline.yaml`:

- collects only `SOL/USDT`;
- uses only Binance;
- uses the 15m timeframe;
- has Telegram disabled;
- stores runtime files in the ignored directory `state/forecast_sol_15m/`.

One full manual cycle:

```bash
.venv/bin/python controller/run_forecast_cycle.py \
  --journal-model sol_gru_15m_v1
```

The runner uses `pipeline/.venv` for the CCXT pipeline and `.venv` for TensorFlow inference. If the pipeline fails or does not update the merged CSV, the producer is not started.

## Local cron without cross-server delivery

Run a full cycle manually first. Once it succeeds, add the following via `crontab -e`:

```cron
2,17,32,47 * * * * cd /path/to/NEXGEN_CRYPTO_PROJECT && .venv/bin/python controller/run_forecast_cycle.py --journal-model sol_gru_15m_v1 >> logs/forecast_cycle.log 2>&1
```

That is 96 potential runs per day. The two-minute delay gives the exchange time to close the candle. The file lock prevents overlapping cycles.

This option is only suitable for a local smoke test. For the official audit journal, use a separate repository and the P3 command below.

After a retrain, the cron command must be updated with the new `--journal-model`.

## P3: separate private Git repository

NEXGEN and the consumer run on different machines. A separate private repository, `forecast-journal`, was chosen:

- NEXGEN is the only writer;
- the consumer is read-only;
- the runtime journal branch is `journal`, not `main`;
- the NEXGEN and consumer code repositories do not store the runtime JSONL;
- force-push, rebase, squash and deleting history of the `journal` branch are forbidden.

Automated direct commits are allowed only for append-only data in the separate `journal` branch. Changes to NEXGEN code and documentation still go through pull requests, without exception.

### 1. One-time initialization of the writer clone

First, manually create an empty private repository `forecast-journal`. Do not add tokens, SSH keys, `.env`, models or datasets to it. Then, on the NEXGEN machine:

```bash
git clone git@github.com:<owner>/forecast-journal.git state/forecast-journal
git -C state/forecast-journal switch --orphan journal
touch state/forecast-journal/forecasts.jsonl
git -C state/forecast-journal add forecasts.jsonl
git -C state/forecast-journal commit -m "chore: initialize append-only forecast journal"
git -C state/forecast-journal push -u origin journal
```

For cron, use a dedicated SSH deploy key with write access to `forecast-journal` only. Cron must not use an interactive password or token.

Bind the deploy key to the writer clone only, via the local Git config, so that cron does not depend on the user's `ssh-agent` and cannot use keys for other repositories:

```bash
git -C state/forecast-journal config --local core.sshCommand \
  "/usr/bin/ssh -i /home/<nexgen-user>/.ssh/nexgen_forecast_journal -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes"
```

In the remote repository settings, set up an actually enforced branch rule/ruleset for `journal`: no force-push, no branch deletion, write access only for the NEXGEN writer credential. After saving it, separately verify that the rule is actually enforced for a private repository on the current GitHub plan. Without remote enforcement, the Git history is still a useful log and the publisher still rejects any detected rewind/divergence, but it is not yet an independent, protected audit trail.

If a local `state/forecasts.jsonl` already exists before the cutover, do not backfill it into the new audit journal. Keep the old file as a pre-audit artifact without deleting it and start the official journal with the first new inference after the genesis commit.

### 2. Producer and publishing after every inference

For production, explicitly point the producer at the clone of the separate repository and enable the publisher:

```bash
.venv/bin/python controller/run_forecast_cycle.py \
  --journal-model sol_gru_15m_v1 \
  --journal state/forecast-journal/forecasts.jsonl \
  --publish-repo state/forecast-journal \
  --publish-branch journal
```

Before committing, `controller/publish_forecast_journal.py` checks that:

- the entire JSONL conforms to contract v1;
- timestamps of each model series are strictly increasing;
- the previously committed byte prefix has not been changed or truncated;
- the remote tip is an unmodified ancestor of the local branch, and every not-yet-published commit is a linear append to `forecasts.jsonl` only;
- the file is a regular UTF-8/LF blob with mode `100644`, not a symlink or an executable;
- the working tree and index have no changes outside `forecasts.jsonl`;
- the publisher and the producer do not read/write the journal at the same time.

The publisher creates a commit only when new lines appear, and performs a normal fast-forward push without `--force`. If the push fails temporarily, the local commit and lines are kept; the next run retries the push.

Production cron:

```cron
2,17,32,47 * * * * cd /path/to/NEXGEN_CRYPTO_PROJECT && .venv/bin/python controller/run_forecast_cycle.py --journal-model sol_gru_15m_v1 --journal state/forecast-journal/forecasts.jsonl --publish-repo state/forecast-journal --publish-branch journal >> logs/forecast_cycle.log 2>&1
```

Publishing happens immediately after a successful inference, not once an hour. For the h1=15m horizon an hourly push would be too late: the outcome could become known before the line appears on the remote. The publisher uses a single shared 120-second budget for waiting on the lock, `ls-remote`, `fetch`, commit and push; individual operations do not get a fresh full timeout.

To retry after a brief network failure, a separate publisher-only cron that runs before h1 matures is recommended:

```cron
8,23,38,53 * * * * cd /path/to/NEXGEN_CRYPTO_PROJECT && .venv/bin/python controller/publish_forecast_journal.py --repo state/forecast-journal --branch journal >> logs/forecast_journal_publish.log 2>&1
```

It does not create empty commits: if there are no new lines, it only validates the journal and retries a normal push of the current `HEAD`.

A successful remote response to the push must arrive before h1 matures. The publisher's JSON result contains `commit`, `publisherObservedAt`, `pendingRecords` and `lateRecords`; a normal run requires `lateRecords=0`. This is a local operational indicator, not an independent server timestamp. If both publish attempts failed, or the timely appearance of the SHA on the remote cannot be confirmed, the corresponding line must be excluded from an honest skill test; a later push does not turn it into an ex-ante forecast.

After a retrain, change `--journal-model` to the new version at the same time, for example `sol_gru_15m_v2`.

### 3. Read-only clone on the consumer

On the consumer server, use a separate read-only deploy key:

```bash
git clone --branch journal --single-branch \
  git@github.com:<owner>/forecast-journal.git \
  /opt/consumer/state/forecast-journal
cd /opt/consumer/state
test ! -e forecasts.jsonl
ln -s forecast-journal/forecasts.jsonl forecasts.jsonl
```

If `state/forecasts.jsonl` already exists, do not replace it automatically: stop first and decide on the migration separately. The consumer must not commit or push.

An hourly pull is not enough to independently record when a SHA first appears. For h1=15m, even a five-minute interval does not leave a strict worst-case margin after the allowed 10 minutes for inference and the two-minute publish timeout. The audit observer on the consumer must run a fast-forward-only pull every minute:

```cron
* * * * * git -C /opt/consumer/state/forecast-journal pull --ff-only origin journal >> /opt/consumer/logs/forecast_journal_pull.log 2>&1
```

After a successful pull, the receiving side runs its validator and records, append-only, the pair `observedAt` (consumer UTC time) + `commit SHA`. The first such record must precede the maturity of the corresponding h1; otherwise the forecast is considered late. The receipt log is implemented on the consumer side and remains a mandatory deployment gate. Any branch divergence, contract error or history rewrite is a reason to stop the import, not to run reset/rebase.

Git history provides a useful remote audit trail only when force-push is forbidden and there is a single write credential. Local author/committer timestamps can be altered, so they prove nothing on their own; the first timely observation of the SHA on the consumer machine (or an equivalent server-side receipt timestamp) is treated as independent operational evidence.

## Validation on the receiving side

The NEXGEN repository does not contain `forecast/CONTRACT.md` or `forecast/validate.mjs`, so Node validation is not available locally. After a pull on the consumer:

```bash
cd /opt/consumer
node forecast/validate.mjs
```

Until the validator reports green and `n >= 500` records have accumulated, the journal is used only for paper-first analysis.
