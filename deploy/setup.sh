#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/adi}"
SERVICE_USER="${SERVICE_USER:-ubuntu}"

if [ ! -f "$APP_DIR/run.py" ]; then
  echo "run.py not found in $APP_DIR"
  echo "copy the project there first. see deploy/README.md"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

cd "$APP_DIR"
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

sudo cp deploy/adi@.service /etc/systemd/system/adi@.service
sudo systemctl daemon-reload
sudo systemctl enable --now adi@1
sudo systemctl enable --now adi@2

echo
echo "installed."
echo "  systemctl status adi@1 adi@2"
echo "  journalctl -u adi@1 -f"
echo "  ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 $SERVICE_USER@<public-ip>"
