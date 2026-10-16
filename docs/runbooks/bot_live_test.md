# Runbook: Telegram bot live test

> This document describes how to safely verify that the Telegram bot works correctly
> after all the stabilization changes. Follow it step by step.

---

## Prerequisites

- [ ] `.env` in the project root is filled in (TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_ID, CHAT_ID)
- [ ] `pipeline/access.json` exists and contains your admin Telegram ID
- [ ] `pipeline/.venv` is installed (`pipeline/.venv/bin/python -c "import aiogram"`)
- [ ] Telegram is open on your phone (you will need to interact with the bot)

Check the environment:
```bash
pipeline/.venv/bin/python -c "import aiogram, ccxt, psutil; print('OK')"
```

---

## Step 1. Make sure access.json is in place

```bash
cat pipeline/access.json
```

If the file does not exist (it is in .gitignore):
```bash
cp pipeline/access.json.example pipeline/access.json
```

Then edit `pipeline/access.json` and add your Telegram ID to the admins list.
The file structure is shown in `pipeline/access.json.example`.

---

## Step 2. Start the bot

```bash
cd /path/to/crypto_ml_project
pipeline/.venv/bin/python pipeline/bot_pipeline_control.py
```

Expected output on startup:
```
[DEBUG] .env loaded from: /path/.env
[DEBUG] using config: /path/pipeline/pipeline_config.yaml
```

If you get the error `TELEGRAM_BOT_TOKEN is not set`, check `.env`.

---

## Step 3. Basic command check in Telegram

Open the bot in Telegram and run:

| Command | Expected result |
|---------|-----------------|
| `/start` | Welcome message |
| `/panel` | Inline keyboard with control buttons |
| `/my_status` | Process status (pipeline not running) |

---

## Step 4. Test launching the pipeline from the bot

> ⚠️ Before this step, make sure `telegram_enabled: false` is set in `pipeline/pipeline_config.yaml`
> to avoid unnecessary notifications during the test.

1. Send `/menu` → configure: 1 pair (LTC/USDT), binance, 1 day
2. Send `/my_run` → a subprocess should start
3. Send `/my_status` → should show the PID and uptime
4. Wait for it to finish (1-2 minutes)
5. Send `/lastlog` → the last lines of the log should appear
6. Send `/plots` → PNG charts should arrive (if they were generated)

---

## Step 5. Test stopping the pipeline

```
/my_run     ← start
/my_stop    ← stop
/my_status  ← make sure the process has exited
```

---

## Step 6. Test admin commands

```
/admin      ← the admin panel should open
```

If access is denied, check that your Telegram ID matches `TELEGRAM_ADMIN_ID` in `.env`.

---

## Expected logs

While the bot is running, logs are written to:
- `pipeline/logs/crypto_log.txt` — main pipeline log
- `pipeline/logs/user_activity/<telegram_id>.log` — command history

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `TELEGRAM_BOT_TOKEN is not set` | `.env` not found | Run from the project root or set `DOTENV_PATH=/path/.env` |
| `Access denied` | Your ID is not in access.json | Add your ID to `roles.admins` in `pipeline/access.json` |
| Pipeline subprocess crashes | Wrong interpreter | Check that `pipeline/.venv/bin/python` exists |
| `FileNotFoundError: pipeline_config.yaml` | Not started from the project root | Run from the project root |
| CCXT timeout during collection | No internet connection or the exchange is unavailable | Check the connection, shorten the exchange list in pipeline_config.yaml |

---

## Stopping the bot

In the terminal: `Ctrl+C`

---

## After a successful test

If all steps pass, you can:
1. Lift the `telegram_enabled: false` restriction for production use
2. Run the pipeline to collect real data and train a model
3. Move on to new features (BUY/SELL signals, `/start_live` commands)
