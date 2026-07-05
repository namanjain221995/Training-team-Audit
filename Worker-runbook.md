# 🛠️ Training-Audit Worker — Runbook

| | |
|---|---|
| **Instance** | `i-0523c58e6a83b7731` (Amazon Linux 2023, us-east-1) |
| **Service** | `training-worker` (systemd) |
| **Code + env** | `/opt/training-audit` |
| **Queue** | `training-analysis-jobs` (SQS) |

> [!NOTE]
> The worker **powers itself off** after `IDLE_MINUTES` with an empty queue — that's normal.
> The Lambda (new job) or the GitHub Action (deploy) wakes it up again.

---

## 0. Connect

**EC2 console → instance `i-0523c58e6a83b7731` → Connect → Session Manager.**

If the instance is **stopped**: select it → *Instance state* → *Start instance*, wait for
"Running", then Connect. (Lambda / GitHub Actions also start it automatically.)

---

## 1. Update the `.env` (key, model, idle time, merge mode…)

> [!WARNING]
> **Stop the worker first** — otherwise the box can idle-power-off in the middle of
> your edit (this happened once already).

```bash
# STEP 1 — pause the worker
sudo systemctl stop training-worker

# STEP 2 — edit (ALWAYS with sudo — the file is root-only chmod 600;
#          plain `nano .env` shows an empty "Permission denied" buffer)
sudo nano /opt/training-audit/.env
#   save = Ctrl+O then Enter,  exit = Ctrl+X

# STEP 3 — verify what is really in the file
sudo cat /opt/training-audit/.env

# STEP 4 — start again (service re-reads .env on every start)
sudo systemctl start training-worker
```

**Keys the worker reads:**
`AWS_REGION` · `ANALYSIS_QUEUE_URL` · `IDLE_MINUTES` · `STOP_WHEN_IDLE` ·
`MERGE_STRATEGY` (`stitch`|`longest`) · `OPENAI_API_KEY` · `OPENAI_MODEL` ·
`OPENAI_REASONING_EFFORT` · `MODEL_CACHE_DIR` · `ENABLE_VIDEO_ANALYSIS` ·
`SAVE_ANALYSIS_VIDEO` · `TRAINER_PHOTOS_DIR` (optional override)

<details>
<summary><b>Overwrite the whole file in one paste</b> (alternative to nano)</summary>

```bash
sudo tee /opt/training-audit/.env > /dev/null << 'EOF'
AWS_REGION=us-east-1
ANALYSIS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/985100584614/training-analysis-jobs
IDLE_MINUTES=10
STOP_WHEN_IDLE=true
MERGE_STRATEGY=stitch
OPENAI_API_KEY=sk-PASTE-KEY-HERE
OPENAI_MODEL=gpt-5.5
OPENAI_REASONING_EFFORT=high
MODEL_CACHE_DIR=/opt/training-audit/.cache
ENABLE_VIDEO_ANALYSIS=true
SAVE_ANALYSIS_VIDEO=true
EOF
sudo chmod 600 /opt/training-audit/.env
```

</details>

---

## 2. Start / stop / restart / status

```bash
sudo systemctl start   training-worker     # start
sudo systemctl stop    training-worker     # stop (e.g. before long edits)
sudo systemctl restart training-worker     # apply a new .env / new code
systemctl status training-worker           # running? last lines, PID, uptime
```

A **healthy start** looks like this in the logs:

```
Worker up. queue=...training-analysis-jobs idle_stop=True(10.0m) merge=stitch region=us-east-1
```

---

## 3. See the real logs

```bash
# LIVE — follow everything as it happens
# (Ctrl+C exits the VIEW only, the service keeps running)
journalctl -u training-worker -f

# Last 100 lines
journalctl -u training-worker -n 100 --no-pager

# Everything since the last boot (e.g. after Lambda woke the box)
journalctl -u training-worker -b --no-pager

# Everything from the last 2 hours
journalctl -u training-worker --since "2 hours ago" --no-pager

# Only one meeting's lines
journalctl -u training-worker --no-pager | grep 93488261018
```

**What a normal job looks like, in order:**

```
=== JOB (attempt 1): meeting 934... day 8 candidate Abhinaya... ===
got s3://zoom-automation-bucket/Training/.../MP4/...
[1/5] Transcript track...   ... [5/5] Writing proof folder + report...
RESULT -> .../result.json   (+ coverage % and integrity /100 lines)
Uploaded N files to s3://zoom-automation-bucket/Training/.../
Session merge (stitch, 1 chunk(s)) -> .../session-result-<id>.json
=== JOB DONE (message deleted) ===
...10 idle minutes later:
Idle for 10.0 min — stopping this instance.
```

---

## 4. Run one meeting by hand (test / reprocess)

```bash
sudo /opt/training-audit/.venv/bin/python -m worker.worker --once \
  --bucket zoom-automation-bucket \
  --prefix "Training/Divya_Prajapati/2026/July/Abhinaya_Sree_Talluri/2026-07-01/Time-2-26-AM-IST/93488261018/"
```

- Prints the full pipeline to the terminal directly.
- **First ever run** downloads models (~300 MB) → slow once, cached after.
- Outputs land in S3 **at that same prefix**: `result.json`, `report.html`,
  `analysis-video.mp4`, `proof/` + date-level `session-result-<meeting>.json`.

---

## 5. Update code on the instance

**Normal way:** `git push` to `main` → the GitHub Action deploys automatically.

**By hand on the instance:**

```bash
sudo bash /opt/training-audit/deploy/deploy.sh     # git pull + pip + restart
```

> [!TIP]
> Every service start also **self-updates first** (`deploy/self-update.sh`),
> so even a plain wake-up boots the newest `main`.

---

## 6. Quick fixes

| Symptom | Cause / fix |
|---|---|
| "Permission denied" reading/writing `.env` | You forgot `sudo` |
| nano saved to a weird name (`.envma` etc.) | At the *"File Name to Write"* prompt just press **Enter** |
| Box powered off while I was working | Idle-stop did it. Start the instance again; next time run `sudo systemctl stop training-worker` first |
| Worker running but nothing happens | Queue is empty — check the Lambda logs wrote `Enqueued analysis job` |
| Job failed 3 times | It is **dropped** (poison-message guard). Fix the cause, rerun with `--once` (section 4) |
| Want the box to NEVER self-stop (long debugging) | Set `STOP_WHEN_IDLE=false` in `.env` + restart. **Set back to `true` after!** |
