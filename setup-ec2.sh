#!/usr/bin/env bash
# ONE-TIME bootstrap of a fresh Ubuntu 22.04/24.04 EC2 instance as the analysis
# worker. Run as root:  sudo bash setup-ec2.sh https://github.com/<you>/<repo>.git
set -euo pipefail
REPO_URL="${1:?usage: setup-ec2.sh <git-repo-url>}"

apt-get update
apt-get install -y --no-install-recommends \
  git python3-venv python3-pip python3-dev build-essential \
  ffmpeg libglib2.0-0 libgl1 libgomp1 snapd
# SSM agent (usually preinstalled on AWS Ubuntu images; make sure)
snap install amazon-ssm-agent --classic 2>/dev/null || true
systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true

# code
if [ ! -d /opt/training-audit/.git ]; then
  git clone "$REPO_URL" /opt/training-audit
fi
cd /opt/training-audit

# venv + deps (numpy<2 + cython first: insightface builds a C extension)
python3 -m venv .venv
./.venv/bin/pip install -U pip wheel
./.venv/bin/pip install "numpy>=1.24,<2.0" cython
./.venv/bin/pip install -r requirements.txt

# env file (edit it!)
if [ ! -f .env ]; then
  cat > .env << 'ENVEOF'
AWS_REGION=us-east-1
ANALYSIS_QUEUE_URL=REPLACE_ME   # https://sqs.us-east-1.amazonaws.com/985100584614/training-analysis-jobs
IDLE_MINUTES=10
STOP_WHEN_IDLE=true
MERGE_STRATEGY=stitch
# trainer photos ship with the repo (trainer/<Trainer_Name>.png); override only if moved:
# TRAINER_PHOTOS_DIR=/opt/training-audit/trainer
OPENAI_SECRET_NAME=training-analysis/openai
OPENAI_MODEL=gpt-4o
MODEL_CACHE_DIR=/opt/training-audit/.cache
ENABLE_VIDEO_ANALYSIS=true
SAVE_ANALYSIS_VIDEO=true
ENVEOF
  echo ">>> EDIT /opt/training-audit/.env (queue URL at minimum) <<<"
fi

# service
cp deploy/training-worker.service /etc/systemd/system/training-worker.service
chmod +x deploy/*.sh
systemctl daemon-reload
systemctl enable --now training-worker

echo "
Bootstrap done. Remaining AWS setup (once):
  1. Attach an instance role with deploy/iam-worker-policy.json
     + the AWS managed policy AmazonSSMManagedInstanceCore.
  2. Tag this instance:  Role=training-audit-worker  (the stop-self permission is tag-scoped).
  3. Put the OpenAI key in Secrets Manager as 'training-analysis/openai'
     (plain string or {\"OPENAI_API_KEY\": \"...\"}).
  4. Add WORKER_INSTANCE_ID env + ec2:StartInstances to the Lambda (deploy/lambda-start-worker.md).
  5. Optional: give the SQS queue a dead-letter queue (maxReceiveCount 3).
"
