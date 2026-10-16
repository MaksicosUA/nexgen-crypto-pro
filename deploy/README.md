# Deploy

Files for deploying the project to production.

## Contents

| File | Purpose |
|------|---------|
| `nexgen-bot.service` | systemd service for the Telegram bot on a VPS |
| `Dockerfile.ml` | Docker image for GPU-based ML training (RunPod / Vast.ai) |
| `vps_setup.sh` | Automated setup script for Ubuntu 22.04/24.04 |

---

## Deployment layout

```
[VPS ~€4/month]  ←→  [Telegram]
  ├── Telegram bot 24/7 (systemd)
  ├── Data collection (CCXT pipeline)
  └── data/ — accumulated CSV files

        ↕ rsync (data ↓, models ↑)

[RunPod / Vast.ai — on demand]
  └── Dockerfile.ml → LSTM/GRU/Hybrid training
```

---

## VPS: deploying the bot

### 1. Quick setup (Ubuntu 22.04+)

```bash
# On the server, as root:
curl -sSL https://raw.githubusercontent.com/MaksicosUA/nexgen-crypto-pro/main/deploy/vps_setup.sh | bash
```

Or manually:
```bash
git clone git@github.com:MaksicosUA/nexgen-crypto-pro.git /opt/nexgen_crypto_pro
cd /opt/nexgen_crypto_pro
bash deploy/vps_setup.sh
```

### 2. Upload secrets (from your local machine)

```bash
scp .env user@YOUR_SERVER:/opt/nexgen_crypto_pro/.env
scp pipeline/access.json user@YOUR_SERVER:/opt/nexgen_crypto_pro/pipeline/access.json
```

### 3. Start the bot

```bash
systemctl start nexgen-bot
systemctl status nexgen-bot
journalctl -u nexgen-bot -f   # live logs
```

---

## GPU: training with Docker (RunPod)

### Build the image

```bash
docker build -f deploy/Dockerfile.ml -t nexgen-ml .
```

### Run training (with mounted data)

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  --env-file .env \
  nexgen-ml \
  python models/train_lstm.py --config configs/config.yaml
```

### On RunPod

1. Create a Pod from the **RunPod PyTorch** or **TensorFlow** template
2. Get the code into the Pod: `git clone`
3. Install dependencies: `pip install -r requirements-ml.txt`
4. Mount a Persistent Volume for `data/` and `models/`
5. Run training; when it finishes, download `models/saved_models/`

### Syncing models back to the VPS

```bash
rsync -avz models/saved_models/ user@YOUR_SERVER:/opt/nexgen_crypto_pro/models/saved_models/
```

---

## Important

- **Never commit `.env`** — copy it to the server manually with `scp`
- **`pipeline/access.json`** — same rule
- To update the code on the VPS: `git pull` + `systemctl restart nexgen-bot`
