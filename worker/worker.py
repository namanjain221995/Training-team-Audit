"""EC2 worker — drains the training-analysis-jobs SQS queue.

Per job (one job = one recording chunk, enqueued by the Lambda when the
transcript webhook lands and training-temp.json is written):

  1. Download from the meeting's S3 prefix:  MP4/, TRANSCRIPT/ (.vtt) and
     training-temp.json. The trainer's reference photo is NOT downloaded — it
     comes from the repo's trainer/ folder (ships with git pull; see
     TRAINER_PHOTOS_DIR).
  2. Run the full analyzer (analyzer.api) in a per-job temp dir.
  3. Upload result.json, report.html, analysis-video.mp4 and the whole proof/
     folder BACK TO THE SAME MEETING PREFIX. Nothing is kept on the instance —
     the temp dir is deleted after every job (only the model cache persists).
  4. Re-merge the session file: gather every sibling chunk's result.json for
     this meeting_id under the same date and write
     {date}/session-result-{meeting_id}.json  (strategy: MERGE_STRATEGY).

Lifecycle: if the queue stays empty for IDLE_MINUTES, the worker stops its own
EC2 instance (the Lambda starts it again when the next job is enqueued). While a
job is running, new jobs simply wait in the queue and are taken next.

Manual reprocess (no SQS):
  python -m worker.worker --once --bucket zoom-automation-bucket \
      --prefix "Training/T/2026/July/C/2026-07-01/Time-7-07-PM-IST/97783959572/"
"""
import argparse
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

import boto3

from analyzer import api
from worker import merge as session_merge

REGION       = os.environ.get("AWS_REGION", "us-east-1").strip() or "us-east-1"
QUEUE_URL    = os.environ.get("ANALYSIS_QUEUE_URL", "").strip()
IDLE_MINUTES = float(os.environ.get("IDLE_MINUTES", "10") or 10)
POLL_WAIT    = int(os.environ.get("POLL_WAIT_SECONDS", "20") or 20)
STOP_WHEN_IDLE = (os.environ.get("STOP_WHEN_IDLE", "true").strip().lower()
                  in ("1", "true", "yes", "on"))
MERGE_STRATEGY = os.environ.get("MERGE_STRATEGY", "stitch").strip().lower() or "stitch"
# trainer reference photos live IN THE REPO (trainer/<Trainer_Name>.png) and ship
# with git pull — no S3 round-trip. Overridable for non-standard layouts.
TRAINER_PHOTOS_DIR = os.path.abspath(
    os.environ.get("TRAINER_PHOTOS_DIR", "").strip()
    or os.path.join(os.path.dirname(__file__), "..", "trainer"))
OPENAI_SECRET_NAME = os.environ.get("OPENAI_SECRET_NAME", "training-analysis/openai").strip()
WORK_ROOT = os.environ.get("WORK_ROOT", tempfile.gettempdir())
MAX_RECEIVES = int(os.environ.get("MAX_RECEIVES", "3") or 3)

_shutdown = False


def main() -> int:
    ap = argparse.ArgumentParser(description="Training-analysis EC2 worker")
    ap.add_argument("--once", action="store_true", help="process one prefix and exit (no SQS)")
    ap.add_argument("--bucket", default="", help="--once: S3 bucket")
    ap.add_argument("--prefix", default="", help="--once: full meeting prefix ending with /{meeting_id}/")
    args = ap.parse_args()

    _resolve_openai_key()

    if args.once:
        if not (args.bucket and args.prefix):
            ap.error("--once needs --bucket and --prefix")
        job = _job_from_prefix(args.bucket, args.prefix)
        process_job(job)
        return 0

    if not QUEUE_URL:
        print("ERROR: ANALYSIS_QUEUE_URL is not set"); return 2
    return loop()


# ── the loop ─────────────────────────────────────────────────────────────────
def loop() -> int:
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    sqs = boto3.client("sqs", region_name=REGION)
    last_active = time.time()
    print(f"Worker up. queue={QUEUE_URL} idle_stop={STOP_WHEN_IDLE}({IDLE_MINUTES}m) "
          f"merge={MERGE_STRATEGY} region={REGION}")

    while not _shutdown:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=POLL_WAIT,
            AttributeNames=["ApproximateReceiveCount"],
        )
        msgs = resp.get("Messages") or []
        if not msgs:
            idle_min = (time.time() - last_active) / 60
            if STOP_WHEN_IDLE and idle_min >= IDLE_MINUTES:
                print(f"Idle for {idle_min:.1f} min — stopping this instance. "
                      f"(The Lambda starts it again on the next job.)")
                _stop_self()
                last_active = time.time()   # if the stop failed, don't spam
            continue

        m = msgs[0]
        receives = int((m.get("Attributes") or {}).get("ApproximateReceiveCount", "1"))
        try:
            job = json.loads(m["Body"])
            print(f"\n=== JOB (attempt {receives}): meeting {job.get('meeting_id')} "
                  f"day {job.get('day')} candidate {job.get('candidate')} ===")
            process_job(job)
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
            print("=== JOB DONE (message deleted) ===")
        except Exception as exc:
            print(f"JOB FAILED: {exc!r}")
            if receives >= MAX_RECEIVES:
                print(f"Dropping message after {receives} attempts (poison-message guard). "
                      f"Reprocess manually with --once when fixed.")
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
            # else: message reappears after the visibility timeout for a retry
        last_active = time.time()

    print("Worker exiting (signal).")
    return 0


def _on_term(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"Signal {signum} received — finishing current job then exiting.")


# ── one job ──────────────────────────────────────────────────────────────────
def process_job(job: dict) -> None:
    bucket = job["bucket"]
    prefix = job["prefix"].rstrip("/") + "/"
    meeting_id = str(job.get("meeting_id") or "").strip()
    s3 = boto3.client("s3", region_name=REGION)

    work = tempfile.mkdtemp(prefix=f"job-{meeting_id or 'meeting'}-", dir=WORK_ROOT)
    try:
        # ── 1. download inputs ──────────────────────────────────────────────
        video = _download_first(s3, bucket, prefix + "MP4/", (".mp4",), work) \
                or _download_first(s3, bucket, prefix, (".mp4",), work)
        if not video:
            raise RuntimeError(f"no MP4 under s3://{bucket}/{prefix}")
        vtt = _download_first(s3, bucket, prefix + "TRANSCRIPT/", (".vtt",), work)
        temp_json = _download_key(s3, bucket, prefix + "training-temp.json", work)
        trainer_img = _find_trainer_photo(job.get("trainer") or "")

        # ── 2. analyze ──────────────────────────────────────────────────────
        out_dir = os.path.join(work, "output")
        cfg = api.build_config(
            video_path=video, output_dir=out_dir,
            transcript_path=vtt, trainer_image_path=trainer_img,
            training_temp_path=temp_json,
            day_number=_int_or_none(job.get("day")),
            trainer_name=str(job.get("trainer") or ""),
            candidate_name=str(job.get("candidate") or ""),
            meeting_id=meeting_id or None,
            day_step_name=job.get("day_step_name"),
            s3_prefix=prefix, s3_bucket=bucket,
            frames_work_dir=os.path.join(work, "frames_work"),
            save_frames=False,   # proof/ carries the evidence copies; skip bulk frames
        )
        rc = api.run(cfg)
        if rc != 0:
            raise RuntimeError(f"analyzer returned {rc}")

        # ── 3. upload outputs to the SAME meeting prefix ────────────────────
        uploaded = _upload_dir(s3, bucket, prefix, out_dir)
        print(f"Uploaded {uploaded} files to s3://{bucket}/{prefix}")

        # ── 4. session merge (all chunks of this meeting_id on this date) ───
        if meeting_id:
            _merge_session(s3, bucket, prefix, meeting_id, cfg.day_plan)
    finally:
        shutil.rmtree(work, ignore_errors=True)   # nothing persists on the instance


def _merge_session(s3, bucket: str, prefix: str, meeting_id: str, day_plan: dict) -> None:
    date_prefix = _date_prefix(prefix)
    if not date_prefix:
        print("Could not derive date prefix; skipping session merge")
        return
    suffix = f"/{meeting_id}/result.json"
    keys = [k for k in _list_keys(s3, bucket, date_prefix) if k.endswith(suffix)]
    if not keys:
        print("No chunk results found for merge (unexpected)"); return
    chunks = []
    for k in sorted(keys):
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
        chunks.append((k, json.loads(body)))
    session = session_merge.merge(chunks, MERGE_STRATEGY, day_plan)
    out_key = f"{date_prefix}session-result-{meeting_id}.json"
    s3.put_object(Bucket=bucket, Key=out_key,
                  Body=json.dumps(session, indent=2, ensure_ascii=False).encode(),
                  ContentType="application/json")
    sc = session.get("scoring", {})
    print(f"Session merge ({session['merge']['strategy']}, {len(chunks)} chunk(s)) -> "
          f"s3://{bucket}/{out_key}  integrity={sc.get('session_integrity_score')} "
          f"coverage={sc.get('trainer_coverage_score')}%")


# ── S3 helpers ───────────────────────────────────────────────────────────────
def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents") or []]
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")


def _download_first(s3, bucket, prefix, exts, work) -> str | None:
    """Largest object under prefix with one of the extensions (skip deeper folders for root scans)."""
    cands = [(k, sz) for k, sz in _list_with_size(s3, bucket, prefix)
             if k.lower().endswith(exts)]
    if not cands:
        return None
    key, _ = max(cands, key=lambda x: x[1])
    return _download_key(s3, bucket, key, work)


def _list_with_size(s3, bucket, prefix):
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents") or []:
            yield o["Key"], o.get("Size") or 0
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")


def _download_key(s3, bucket, key, work) -> str | None:
    local = os.path.join(work, "in", os.path.basename(key))
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        s3.download_file(bucket, key, local)
        print(f"  got s3://{bucket}/{key}")
        return local
    except Exception:
        return None


def _find_trainer_photo(trainer_raw: str) -> str | None:
    """Reference photo from the repo's trainer/ folder. Filenames are the
    folder-style trainer names exactly as in the S3 prefix / SQS job
    (e.g. Naghma_Akhtar.png). Exact match first, then case-insensitive on the
    filename stem. The analyzer degrades gracefully when this returns None."""
    if not trainer_raw:
        return None
    exts = (".png", ".jpg", ".jpeg")
    for ext in exts:
        p = os.path.join(TRAINER_PHOTOS_DIR, trainer_raw + ext)
        if os.path.isfile(p):
            print(f"  trainer photo: {p}")
            return os.path.abspath(p)
    try:
        want = trainer_raw.lower()
        for name in sorted(os.listdir(TRAINER_PHOTOS_DIR)):
            stem, ext = os.path.splitext(name)
            if stem.lower() == want and ext.lower() in exts:
                p = os.path.abspath(os.path.join(TRAINER_PHOTOS_DIR, name))
                print(f"  trainer photo (case-insensitive match): {p}")
                return p
    except OSError:
        pass
    print(f"  no trainer photo for {trainer_raw} in trainer/")
    return None


def _upload_dir(s3, bucket: str, prefix: str, out_dir: str) -> int:
    n = 0
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, out_dir).replace(os.sep, "/")
            ctype = mimetypes.guess_type(f)[0] or "application/octet-stream"
            s3.upload_file(local, bucket, prefix + rel, ExtraArgs={"ContentType": ctype})
            n += 1
    return n


def _date_prefix(prefix: str) -> str | None:
    """Training/T/Y/M/C/DATE/TIME/MEETING/ -> Training/T/Y/M/C/DATE/"""
    parts = prefix.strip("/").split("/")
    if len(parts) < 3:
        return None
    return "/".join(parts[:-2]) + "/"


def _job_from_prefix(bucket: str, prefix: str) -> dict:
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3", region_name=REGION)
    try:
        body = s3.get_object(Bucket=bucket, Key=prefix + "training-temp.json")["Body"].read()
        t = json.loads(body)
    except Exception:
        t = {}
    parts = prefix.strip("/").split("/")
    return {
        "bucket": bucket, "prefix": prefix,
        "meeting_id": t.get("meeting_id") or (parts[-1] if parts else ""),
        "day": t.get("day"), "day_step_name": t.get("day_step_name"),
        "trainer": t.get("trainer") or (parts[1] if len(parts) > 1 else ""),
        "candidate": t.get("candidate") or (parts[4] if len(parts) > 4 else ""),
    }


# ── OpenAI key from Secrets Manager (so no key sits in a repo/.env) ──────────
def _resolve_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    if os.environ.get("MOCK_OPENAI", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    if not OPENAI_SECRET_NAME:
        return
    try:
        sm = boto3.client("secretsmanager", region_name=REGION)
        val = sm.get_secret_value(SecretId=OPENAI_SECRET_NAME).get("SecretString") or ""
        try:
            val = json.loads(val).get("OPENAI_API_KEY", val)
        except Exception:
            pass
        if val:
            os.environ["OPENAI_API_KEY"] = val.strip()
            print(f"OPENAI_API_KEY loaded from Secrets Manager ({OPENAI_SECRET_NAME})")
    except Exception as exc:
        print(f"Could not read secret {OPENAI_SECRET_NAME} ({exc}); "
              f"set OPENAI_API_KEY or MOCK_OPENAI=true")


# ── stop this instance ───────────────────────────────────────────────────────
def _stop_self() -> None:
    try:
        iid, region = _instance_identity()
        boto3.client("ec2", region_name=region or REGION).stop_instances(InstanceIds=[iid])
        print(f"stop_instances({iid}) requested — waiting for power-off...")
        time.sleep(300)
    except Exception as exc:
        print(f"EC2 API stop failed ({exc}); falling back to OS shutdown "
              f"(instance-initiated-shutdown-behavior must be 'stop').")
        try:
            subprocess.run(["shutdown", "-h", "now"], check=False)
            time.sleep(300)
        except Exception as exc2:
            print(f"shutdown failed too ({exc2}); staying up.")


def _instance_identity() -> tuple[str, str]:
    tok_req = urllib.request.Request(
        "http://169.254.169.254/latest/api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    token = urllib.request.urlopen(tok_req, timeout=2).read().decode()
    def md(path):
        req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=2).read().decode()
    iid = md("instance-id")
    az = md("placement/availability-zone")
    return iid, az[:-1] if az else REGION


def _int_or_none(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
