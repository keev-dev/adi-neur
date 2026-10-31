#!/usr/bin/env bash
# Safe update: pause, checkpoint, pull, install, optionally restart and resume.
# Usage: bash deploy/pull.sh            # code only, no restart
#        bash deploy/pull.sh --restart  # full safe update
set -euo pipefail
cd /opt/adi

say() { echo "[adi] $*"; }

for port in 8000 8001; do
  if curl -sf -m 5 "http://127.0.0.1:$port/state" >/dev/null 2>&1; then
    say "checkpoint $port"
    curl -sf -m 10 -X POST "http://127.0.0.1:$port/cmd" \
      -H 'Content-Type: application/json' -d '{"cmd":"set_paused","value":true}' >/dev/null || true
    curl -sf -m 60 -X POST "http://127.0.0.1:$port/cmd" \
      -H 'Content-Type: application/json' -d '{"cmd":"checkpoint"}' >/dev/null || true
  fi
done

say "git pull"
git pull --ff-only

say "pip install"
./venv/bin/pip install -q -r requirements.txt

say "systemd unit"
sudo cp deploy/adi@.service /etc/systemd/system/
sudo systemctl daemon-reload

if [ "${1:-}" = "--restart" ]; then
  say "restart"
  sudo systemctl restart adi@1 adi@2
  sleep 3
  for port in 8000 8001; do
    curl -sf -m 10 -X POST "http://127.0.0.1:$port/cmd" \
      -H 'Content-Type: application/json' -d '{"cmd":"resume"}' >/dev/null || true
  done
fi

say "done"
