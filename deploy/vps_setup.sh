#!/bin/bash
# =============================================================
# VPS Setup Script — NexGen Crypto Pro
# Run as root on a clean Ubuntu 22.04 / 24.04
# =============================================================
set -e

PROJECT_DIR="/opt/nexgen_crypto_pro"
SERVICE_USER="nexgen"
REPO_URL="git@github.com:MaksicosUA/nexgen-crypto-pro.git"

echo ">>> [1/6] Creating system user ${SERVICE_USER}..."
id "$SERVICE_USER" &>/dev/null || useradd -r -m -s /bin/bash "$SERVICE_USER"

echo ">>> [2/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl build-essential

echo ">>> [3/6] Cloning the repository into ${PROJECT_DIR}..."
if [ -d "$PROJECT_DIR" ]; then
    echo "    Directory already exists, running git pull..."
    cd "$PROJECT_DIR" && sudo -u "$SERVICE_USER" git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    chown -R "$SERVICE_USER":"$SERVICE_USER" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

echo ">>> [4/6] Creating the pipeline virtualenv..."
sudo -u "$SERVICE_USER" python3.12 -m venv pipeline/.venv
sudo -u "$SERVICE_USER" pipeline/.venv/bin/pip install --upgrade pip -q
sudo -u "$SERVICE_USER" pipeline/.venv/bin/pip install -r requirements-pipeline.txt -q
echo "    pipeline/.venv is ready"

echo ">>> [5/6] Copying .env and access.json (manual step)..."
echo ""
echo "    IMPORTANT: copy the following files manually:"
echo "    1. scp .env ${SERVICE_USER}@SERVER:${PROJECT_DIR}/.env"
echo "    2. scp pipeline/access.json ${SERVICE_USER}@SERVER:${PROJECT_DIR}/pipeline/access.json"
echo ""

echo ">>> [6/6] Installing the systemd service..."
cp deploy/nexgen-bot.service /etc/systemd/system/nexgen-bot.service

# Substitute the actual path if needed
sed -i "s|/opt/nexgen_crypto_pro|${PROJECT_DIR}|g" /etc/systemd/system/nexgen-bot.service
sed -i "s|User=nexgen|User=${SERVICE_USER}|g" /etc/systemd/system/nexgen-bot.service

systemctl daemon-reload
systemctl enable nexgen-bot
echo ""
echo "============================================"
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Copy .env and pipeline/access.json to the server"
echo "  2. Start the bot: systemctl start nexgen-bot"
echo "  3. Check the status: systemctl status nexgen-bot"
echo "  4. Follow the logs: journalctl -u nexgen-bot -f"
echo "============================================"
