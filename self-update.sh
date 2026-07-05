#!/usr/bin/env bash
# Pull the latest code + deps. Runs on EVERY service start (systemd ExecStartPre)
# so a stopped instance boots straight onto the newest main. Never blocks the
# worker from starting: failures are logged, the existing code keeps running.
set -u
cd /opt/training-audit || exit 0
echo "[self-update] fetching origin/main..."
if git fetch origin main 2>&1; then
  git reset --hard origin/main
  ./.venv/bin/pip install -q -r requirements.txt || echo "[self-update] pip install failed; keeping current deps"
else
  echo "[self-update] git fetch failed (offline?); starting with current code"
fi
exit 0
