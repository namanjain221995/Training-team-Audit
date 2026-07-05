#!/usr/bin/env bash
# Used by CI (SSM RunCommand) and by hand: update code then restart the worker.
set -euo pipefail
bash /opt/training-audit/deploy/self-update.sh
systemctl restart training-worker
systemctl --no-pager --lines=5 status training-worker || true
echo "deploy complete"
