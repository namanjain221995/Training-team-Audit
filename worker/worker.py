"""EC2 worker — drains the training-analysis-jobs SQS queue.

Per job (one job = one recording chunk, enqueued by the Lambda when the
transcript webhook lands and training-temp.json is written):

  1. Download from the meeting's S3 prefix:  MP4/, TRANSCRIPT/ (.vtt) and
     training-temp.json. The trainer's reference photo is NOT downloaded — it
     comes from the repo's trainer/ folder (ships with git pull; see
     TRAINER_PHOTOS_DIR).
  2. Run the full analyzer (analyzer.api) in a per-job temp dir.
  3. Upload result.json, report.html, analysis-video.mp4 and the whole proof/
     folder BACK TO THE SAME MEETING PREFIX — plus, when enabled, the
     transcripts (transcript-whisper.json, transcript-combined.vtt/.json). Nothing is kept on the instance —
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
import threading
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
TECHSARA_SECRET_NAME = os.environ.get("TECHSARA_SECRET_NAME", "training-analysis/techsara").strip()
WORK_ROOT = os.environ.get("WORK_ROOT", tempfile.gettempdir())
MAX_RECEIVES = int(os.environ.get("MAX_RECEIVES", "3") or 3)
# heartbeat: a job (Whisper + several models + video) can outlast the queue's
# 30-min visibility timeout; keep pushing it out so the job is never re-delivered
VISIBILITY_EXTEND_EVERY_SEC = 300
VISIBILITY_EXTEND_TO_SEC = 1800

_shutdown = False


def main() -> int:
    ap = argparse.ArgumentParser(description="Training-analysis EC2 worker")
    ap.add_argument("--once", action="store_true", help="process one meeting prefix and exit (no SQS)")
    ap.add_argument("--weekly", action="store_true",
                    help="build the weekly report only (no reprocess); --prefix = any one day's meeting prefix")
    ap.add_argument("--backfill", action="store_true",
                    help="reprocess stale daily reports under --prefix root + ensure every candidate's weekly")
    ap.add_argument("--bucket", default="", help="S3 bucket")
    ap.add_argument("--prefix", default="", help="meeting prefix (--once/--weekly) or root prefix (--backfill)")
    ap.add_argument("--since", default="", help="--backfill: only sessions on/after YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true", help="--backfill: list what would run, change nothing")
    ap.add_argument("--force", action="store_true",
                    help="--backfill: reprocess/rebuild even reports already at the current schema")
    args = ap.parse_args()

    _resolve_openai_key()

    if args.once:
        if not (args.bucket and args.prefix):
            ap.error("--once needs --bucket and --prefix")
        process_job(_job_from_prefix(args.bucket, args.prefix))
        return 0

    if args.weekly:
        if not (args.bucket and args.prefix):
            ap.error("--weekly needs --bucket and --prefix (any one day's meeting prefix of the week)")
        s3 = boto3.client("s3", region_name=REGION)
        _rebuild_weekly(s3, args.bucket, args.prefix.rstrip("/") + "/", _weekly_cfg())
        return 0

    if args.backfill:
        if not args.bucket:
            ap.error("--backfill needs --bucket (and optional --prefix root, default Training/Resume-Based/)")
        root = (args.prefix or "Training/Resume-Based/").rstrip("/") + "/"
        backfill(args.bucket, root, since=args.since.strip() or None,
                 dry_run=args.dry_run, force=args.force)
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
        stop_beat = threading.Event()
        beat = threading.Thread(target=_keep_invisible, daemon=True,
                                args=(sqs, m["ReceiptHandle"], stop_beat))
        beat.start()
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
        finally:
            stop_beat.set()
        last_active = time.time()

    print("Worker exiting (signal).")
    return 0


def _keep_invisible(sqs, receipt: str, stop: threading.Event) -> None:
    """Extend the in-flight message's visibility until the job finishes."""
    while not stop.wait(VISIBILITY_EXTEND_EVERY_SEC):
        try:
            sqs.change_message_visibility(QueueUrl=QUEUE_URL, ReceiptHandle=receipt,
                                          VisibilityTimeout=VISIBILITY_EXTEND_TO_SEC)
        except Exception as exc:
            print(f"(heartbeat) could not extend message visibility: {exc!r}")


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
        #   A meeting's MP4/ can hold MORE THAN ONE mp4 when the session dropped
        #   and rejoined the SAME meeting_id (Zoom writes the session in parts).
        #   Analyse ALL the parts (concatenated into one), not just the biggest
        #   file — otherwise half the session is missing from the report.
        vtt = _download_first(s3, bucket, prefix + "TRANSCRIPT/", (".vtt",), work)
        video = _prepare_video(s3, bucket, prefix, vtt, work) \
                or _download_first(s3, bucket, prefix, (".mp4",), work)
        if not video:
            raise RuntimeError(f"no MP4 under s3://{bucket}/{prefix}")
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
        #     Non-fatal: the per-chunk result.json + report were already uploaded
        #     above, so a merge glitch must not fail (and re-run) the whole job.
        if meeting_id:
            try:
                _merge_session(s3, bucket, prefix, meeting_id, cfg)
            except Exception as exc:
                print(f"Session merge failed (non-fatal): {exc!r}")

        # ── 5. weekly rebuild (Option B: after EVERY day, rebuild the
        #        candidate's week report from whatever days exist so far) ─────
        try:
            _rebuild_weekly(s3, bucket, prefix, cfg)
        except Exception as exc:
            print(f"Weekly rebuild failed (non-fatal): {exc!r}")
    finally:
        shutil.rmtree(work, ignore_errors=True)   # nothing persists on the instance


def _merge_session(s3, bucket: str, prefix: str, meeting_id: str, cfg) -> None:
    """Build the ONE day-level result for this meeting_id+date from all its
    recording chunks. One recording -> fast copy (no LLM). Two or more -> stitch
    the transcripts and re-run the LLM once over the whole day (see merge.py),
    then render ONE date-level report.html so reviewers never open a single part
    by mistake. Every breakable step logs where + why it failed."""
    date_prefix = _date_prefix(prefix)
    if not date_prefix:
        print("  [merge] could not derive date prefix; skipping session merge")
        return

    # Load every recording's result.json under this date, then group by the DAY
    # (topic): all recordings that share this date + Day-number are ONE day, even
    # across different meeting_ids (host restarted into a new meeting, or a
    # re-taught session) — owner rule: combine + re-analyse as one day. When the
    # Day number is unknown (Salesforce lookup failed) we can't safely combine
    # across meetings, so we fall back to grouping by meeting_id.
    loaded = []
    for k in sorted(k for k in _list_keys(s3, bucket, date_prefix) if k.endswith("/result.json")):
        try:
            body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
            loaded.append((k, json.loads(body)))
        except Exception as exc:
            print(f"  [merge:load-chunk] FAILED key={k} : {exc!r}")
    if not loaded:
        print("  [merge] no chunk results found for merge (unexpected)"); return

    day = _int_or_none(getattr(cfg, "day_number", None))
    if day is not None:
        # this day's recordings = those tagged with this Day, PLUS any whose Day
        # is unknown (Salesforce miss) on the same date — a null-Day recording on
        # the date is almost certainly part of this day, so fold it in rather than
        # strand it as its own partial result.
        chunks = [(k, r) for k, r in loaded
                  if _int_or_none((r.get("meeting") or {}).get("day")) in (day, None)]
        group_id = f"day{day}"
    else:
        chunks = []
    if not chunks:   # unknown day, or nothing matched -> this meeting only
        suffix = f"/{meeting_id}/result.json"
        chunks = [(k, r) for k, r in loaded if k.endswith(suffix)]
        group_id = meeting_id
    if not chunks:
        print("  [merge] every chunk result failed to load; skipping merge"); return

    n = len(chunks)
    mwork = tempfile.mkdtemp(prefix=f"merge-{meeting_id}-", dir=WORK_ROOT)
    try:
        chunk_transcripts: dict = {}
        oa_list: list = []
        if n > 1:
            # a multi-recording day (same meeting restarted, or two meetings of
            # the same day): re-stitch the transcripts + re-run the LLM
            print(f"  [merge] {n} recordings for {group_id} on this date "
                  f"— stitching + re-analysing the whole day")
            for i, (k, _res) in enumerate(chunks):
                try:
                    chunk_transcripts[k] = _load_chunk_segments(
                        s3, bucket, k, os.path.join(mwork, f"c{i}"))
                except Exception as exc:
                    print(f"  [merge:load-transcript] FAILED key={k} : {exc!r}")
                    chunk_transcripts[k] = []
            oa_list = _merge_clients(cfg)

        session = session_merge.merge(chunks, MERGE_STRATEGY, cfg,
                                      oa_list=oa_list, chunk_transcripts=chunk_transcripts)

        out_key = f"{date_prefix}session-result-{group_id}.json"
        s3.put_object(Bucket=bucket, Key=out_key,
                      Body=json.dumps(session, indent=2, ensure_ascii=False).encode(),
                      ContentType="application/json")

        merge_block = session.get("merge") or {}
        if (merge_block.get("chunk_count") or 0) > 1:
            # only a genuinely merged day gets a date-level report (a single
            # recording already has its own report.html with its video)
            _render_session_report(s3, bucket, date_prefix, group_id, session)

        sc = session.get("scoring") or {}
        print(f"  [merge] {merge_block.get('strategy')} x{merge_block.get('chunk_count')} "
              f"(of {merge_block.get('recordings_total')} rec) -> s3://{bucket}/{out_key}  "
              f"status={sc.get('analysis_status')} "
              f"integrity={sc.get('session_integrity_score')} "
              f"coverage={sc.get('trainer_coverage_score')}%")
    finally:
        shutil.rmtree(mwork, ignore_errors=True)


def _load_chunk_segments(s3, bucket: str, result_key: str, dest: str) -> list:
    """Transcript segments for ONE recording chunk. Prefer the persisted
    transcript-combined.json (speaker-labelled segments); else the raw Zoom VTT.
    Returns [] if neither is present (logged)."""
    os.makedirs(dest, exist_ok=True)
    chunk_prefix = result_key[: -len("result.json")]
    cj = _download_key(s3, bucket, chunk_prefix + "transcript-combined.json", dest)
    if cj:
        try:
            with open(cj, encoding="utf-8") as fh:
                obj = json.load(fh)
            segs = obj.get("segments") if isinstance(obj, dict) else obj
            if segs:
                return segs
        except Exception as exc:
            print(f"  [merge:combined-json] parse failed key={result_key} : {exc!r}")
    vtt = _download_first(s3, bucket, chunk_prefix + "TRANSCRIPT/", (".vtt",), dest)
    if vtt:
        try:
            from analyzer import transcript as T
            return T.parse_vtt(vtt)
        except Exception as exc:
            print(f"  [merge:vtt] parse failed key={result_key} : {exc!r}")
    print(f"  [merge] no transcript found for chunk {chunk_prefix}")
    return []


def _merge_clients(cfg) -> list:
    try:
        from analyzer.openai_client import make_clients
        return make_clients(cfg)
    except Exception as exc:
        print(f"  [merge] no LLM client ({exc!r}); a multi-recording day cannot be "
              f"re-analysed — it will be marked analysis incomplete")
        return []


def _render_session_report(s3, bucket: str, date_prefix: str, group_id: str, session: dict) -> None:
    """Render the ONE day-level report from the merged result and upload it to the
    date folder as report-{group_id}.html (group_id = day{N}, or a meeting_id when
    the Day number is unknown — keyed so two different days on one date never
    overwrite each other). Best-effort — a render failure never aborts the merge."""
    try:
        from analyzer import report as R
        merge_block = session.get("merge") or {}
        # relative links to each recording's own report (video + evidence frames)
        for c in merge_block.get("chunks") or []:
            k = c.get("key") or ""
            if k.startswith(date_prefix) and k.endswith("result.json"):
                c["report_href"] = k[len(date_prefix): -len("result.json")] + "report.html"
        html = R.render(session, base_dir=None)
        tmp = tempfile.mkdtemp(prefix="dayreport-", dir=WORK_ROOT)
        try:
            p = os.path.join(tmp, "report.html")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(html)
            out = f"{date_prefix}report-{group_id}.html"
            s3.upload_file(p, bucket, out, ExtraArgs={"ContentType": "text/html"})
            print(f"  [merge] day report -> s3://{bucket}/{out}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:
        print(f"  [merge:render-report] FAILED date={date_prefix} group={group_id} : {exc!r}")


# ── weekly report (Option B: rebuilt after each day) ─────────────────────────
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
WEEKLY_WINDOW_DAYS = int(os.environ.get("WEEKLY_WINDOW_DAYS", "9") or 9)


def _prev_month(year: str, month: str):
    try:
        i = _MONTHS.index(month)
    except ValueError:
        return None
    return (str(int(year) - 1), "December") if i == 0 else (year, _MONTHS[i - 1])


def _candidate_prefixes(prefix: str) -> list[str]:
    """Candidate folders to scan for this week: the session's own month AND the
    PREVIOUS month (so a week straddling a month boundary — e.g. 29-31 Oct +
    1-2 Nov — is found as one week). Only earlier days can already exist, so the
    next month is never needed here."""
    parts = prefix.strip("/").split("/")
    if len(parts) < 9 or parts[0] != "Training":
        return []
    prog, trn, year, month, cand = parts[1], parts[2], parts[3], parts[4], parts[5]
    out = ["/".join(["Training", prog, trn, year, month, cand]) + "/"]
    pm = _prev_month(year, month)
    if pm:
        out.append("/".join(["Training", prog, trn, pm[0], pm[1], cand]) + "/")
    return out


def _relpath(key: str, base_prefix: str) -> str:
    """Relative path from base_prefix (candidate folder) to an S3 key.
    OS-independent (pure key string math)."""
    if key.startswith(base_prefix):
        return key[len(base_prefix):]
    b = base_prefix.rstrip("/").split("/")
    k = key.split("/")
    i = 0
    while i < len(b) and i < len(k) and b[i] == k[i]:
        i += 1
    return "../" * (len(b) - i) + "/".join(k[i:])


def _rel_href(day_key: str, base_prefix: str) -> str:
    """Relative link from the week report to a per-recording report.html
    (derived from that recording's result.json key)."""
    rp = (day_key[: -len("result.json")] + "report.html"
          if day_key.endswith("result.json") else day_key)
    return _relpath(rp, base_prefix)


def _exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _day_report_href(s3, bucket: str, key: str, base_prefix: str, session: dict | None = None) -> str:
    """The canonical day report link for the weekly table. A merged (>1
    recording) day has a date-level report.html; a single-recording day links its
    one recording's report.html; a legacy per-recording result.json links itself."""
    bn = key.rsplit("/", 1)[-1]
    if bn.startswith("session-result-") and bn.endswith(".json"):
        date_dir = key[: key.rfind("/") + 1]
        mid = bn[len("session-result-"):-len(".json")]
        merged = date_dir + f"report-{mid}.html"   # only a >1-recording day has this
        if _exists(s3, bucket, merged):
            target = merged
        else:
            chunks = ((session or {}).get("merge") or {}).get("chunks") or []
            k0 = chunks[0].get("key") if chunks else None
            target = (k0[: -len("result.json")] + "report.html"
                      if k0 and k0.endswith("result.json") else merged)
        return _relpath(target, base_prefix)
    return _rel_href(key, base_prefix)


def _file_day(s3, bucket: str, key: str):
    """The Day-number a result file belongs to. Parsed straight from a
    session-result-day{N}.json name (no fetch); otherwise read from the file's
    meeting.day."""
    bn = key.rsplit("/", 1)[-1]
    if bn.startswith("session-result-day") and bn.endswith(".json"):
        n = _int_or_none(bn[len("session-result-day"):-len(".json")])
        if n is not None:
            return n
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return _int_or_none((json.loads(body).get("meeting") or {}).get("day"))
    except Exception:
        return None


def _pick_week_sources(s3, bucket: str, cand_prefixes: list, cur_date) -> list:
    """One result file PER (candidate, date, Day-number) in the week window.
    Prefer the merged session-result-*.json (which already combines all the
    recordings of a day's topic, even across meeting_ids); fall back to ONE
    per-recording result.json for a day that has no session file yet. Keyed by
    (cand_prefix, date, day) so a merged same-topic day appears once, two DIFFERENT
    days on one date are both kept, and recording chunks never double-count.
    Returns [(key, cand_prefix, provisional)] — provisional=True marks an un-merged
    multi-recording day (shown but excluded from the week math)."""
    session_by: dict = {}
    legacy_by: dict = {}
    legacy_count: dict = {}
    for cp in cand_prefixes:
        for k in _list_keys(s3, bucket, cp):
            bn = k.rsplit("/", 1)[-1]
            is_session = bn.startswith("session-result-") and bn.endswith(".json")
            is_legacy = k.endswith("/result.json")
            if not (is_session or is_legacy):
                continue
            kd = _date_from_key(k, cp)
            if cur_date is not None and (kd is None or abs((kd - cur_date).days) > WEEKLY_WINDOW_DAYS):
                continue
            ds = kd.strftime("%Y-%m-%d") if kd else "?"
            dkey = (cp, ds, _file_day(s3, bucket, k))
            if is_session:
                session_by[dkey] = (k, cp)
            else:
                legacy_count[dkey] = legacy_count.get(dkey, 0) + 1
                legacy_by.setdefault(dkey, (k, cp))           # first recording of that day
    out = [(k, cp, False) for (k, cp) in session_by.values()]
    for dkey, (k, cp) in legacy_by.items():
        if dkey not in session_by:
            # un-merged: a single recording is a real day; >1 recording means the
            # merge hasn't run, so this one chunk is NOT the whole day -> provisional
            out.append((k, cp, legacy_count.get(dkey, 1) > 1))
    return out


def _date_from_key(key: str, cand_prefix: str):
    from datetime import datetime
    rest = key[len(cand_prefix):] if key.startswith(cand_prefix) else key
    seg = rest.split("/", 1)[0]
    try:
        return datetime.strptime(seg, "%Y-%m-%d")
    except Exception:
        return None


def _rebuild_weekly(s3, bucket: str, prefix: str, cfg) -> None:
    """Rebuild the candidate's cumulative week report after each day and upload
    week-report.html + week-result.json. Idempotent (overwrites).

    Robust to month boundaries: scans this month + the previous month's candidate
    folder, and keeps only the days whose DATE is within WEEKLY_WINDOW_DAYS of the
    session that just ran — so a week that spans two months is grouped as ONE
    week, and a DIFFERENT week the candidate did in an adjacent month is not mixed
    in. The report is written to every candidate folder that contributed a day, so
    it is findable under either month."""
    from datetime import timedelta
    cand_prefixes = _candidate_prefixes(prefix)
    if not cand_prefixes:
        print("Weekly: could not derive candidate prefix; skipping")
        return
    cur_date = _date_from_key(prefix, cand_prefixes[0])   # the Date folder of this session

    picked = _pick_week_sources(s3, bucket, cand_prefixes, cur_date)
    if not picked:
        print("Weekly: no daily/session result in this week window yet")
        return
    contributing = {cp for _k, cp, _p in picked}

    base = cand_prefixes[0]   # the week report is written here; links are relative to it
    wk_dir = tempfile.mkdtemp(prefix="weekly-", dir=WORK_ROOT)
    try:
        paths = []
        for i, (k, _cp, provisional) in enumerate(sorted(picked)):
            local = os.path.join(wk_dir, f"d{i}", "result.json")
            os.makedirs(os.path.dirname(local), exist_ok=True)
            s3.download_file(bucket, k, local)
            try:   # stamp the relative link to this day's canonical report.html
                with open(local, encoding="utf-8") as fh:
                    d = json.load(fh)
                d["report_href"] = _day_report_href(s3, bucket, k, base, d)
                if provisional:
                    # an un-merged multi-recording day: don't let one chunk's
                    # numbers stand in for the day — show it, exclude from the math
                    d.setdefault("meeting", {})["analysis_status"] = "incomplete"
                    d["meeting"]["analysis_reason"] = "merge_pending"
                    d.setdefault("scoring", {})["analysis_status"] = "incomplete"
                with open(local, "w", encoding="utf-8") as fh:
                    json.dump(d, fh)
            except Exception as exc:
                print(f"  [weekly:stamp-href] key={k} : {exc!r}")
            paths.append(local)

        from analyzer import weekly as weekly_mod
        oa = None
        try:
            from analyzer.openai_client import make_clients
            oa = make_clients(cfg)[0]
        except Exception as exc:
            print(f"Weekly: no LLM client ({exc}); building without the AI narrative")

        out = os.path.join(wk_dir, "out")
        agg = weekly_mod.build(paths, out, oa=oa)
        if not agg:
            return
        for cp in sorted(contributing):
            for fn, ctype in (("week-report.html", "text/html"),
                              ("week-result.json", "application/json")):
                p = os.path.join(out, fn)
                if os.path.isfile(p):
                    s3.upload_file(p, bucket, cp + fn, ExtraArgs={"ContentType": ctype})
        print(f"Weekly report rebuilt ({len(agg['meta']['days_found'])} day(s): "
              f"{agg['meta']['days_found']}) -> {sorted(contributing)}")
    finally:
        shutil.rmtree(wk_dir, ignore_errors=True)


# ── backfill: reprocess stale daily reports + ensure weeklies ────────────────
def _weekly_cfg():
    """A minimal Config carrying only the model settings, for make_clients() in
    --weekly / weekly-ensure (no video/job needed)."""
    from analyzer.config import Config, model_settings
    return Config(**model_settings())


def _result_schema_version(s3, bucket: str, meeting_prefix: str):
    """schema_version of an existing result.json, or None if there is no result.json
    (never analyzed). 0 if present but unstamped (analyzed by old code)."""
    try:
        body = s3.get_object(Bucket=bucket, Key=meeting_prefix + "result.json")["Body"].read()
        v = (json.loads(body).get("meeting") or {}).get("schema_version")
        return int(v) if isinstance(v, (int, float)) else 0
    except Exception:
        return None


def _weekly_exists(s3, bucket: str, cand_prefix: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=cand_prefix + "week-report.html")
        return True
    except Exception:
        return False


def _list_meeting_prefixes(s3, bucket: str, root: str) -> list[str]:
    """Every meeting folder under root — identified by a training-temp.json or a
    result.json (so both analyzed and not-yet-analyzed sessions are included)."""
    out = set()
    for k in _list_keys(s3, bucket, root):
        if k.endswith("/training-temp.json"):
            out.add(k[: -len("training-temp.json")])
        elif k.endswith("/result.json"):
            out.add(k[: -len("result.json")])
    return sorted(out)


def _since_ok(meeting_prefix: str, since: str | None) -> bool:
    if not since:
        return True
    from datetime import datetime
    parts = meeting_prefix.strip("/").split("/")
    if len(parts) < 7:
        return True
    try:
        return datetime.strptime(parts[6], "%Y-%m-%d") >= datetime.strptime(since, "%Y-%m-%d")
    except Exception:
        return True


def backfill(bucket: str, root: str, since: str | None = None,
             dry_run: bool = False, force: bool = False) -> None:
    """Reprocess every stale daily report under `root` with the current code, then
    ensure every candidate has a weekly. Idempotent: skips reports already at the
    current schema_version unless --force. Reprocessing re-runs the full analyzer
    (video + LLM) per day, so scope with --prefix / --since and preview with
    --dry-run first."""
    from analyzer.api import SCHEMA_VERSION
    s3 = boto3.client("s3", region_name=REGION)
    prefixes = _list_meeting_prefixes(s3, bucket, root)
    print(f"Backfill scan under s3://{bucket}/{root}: {len(prefixes)} meeting folder(s) "
          f"(target schema v{SCHEMA_VERSION}, since={since or 'any'}, force={force})")

    to_do, skip = [], 0
    for mp in prefixes:
        if not _since_ok(mp, since):
            continue
        ver = _result_schema_version(s3, bucket, mp)
        if force or ver is None or ver < SCHEMA_VERSION:
            to_do.append(mp)
        else:
            skip += 1
    print(f"  to reprocess: {len(to_do)}   already current (skip): {skip}")

    if dry_run:
        for mp in to_do[:60]:
            print(f"  WOULD reprocess: {mp}")
        if len(to_do) > 60:
            print(f"  ... +{len(to_do) - 60} more")
        print("Dry run — nothing changed. Re-run without --dry-run to apply.")
        return

    done = failed = 0
    for mp in to_do:
        try:
            print(f"\n--- backfill reprocess: {mp} ---")
            process_job(_job_from_prefix(bucket, mp))   # reprocess + upload + merge + weekly rebuild
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  backfill FAILED for {mp}: {exc!r}")

    # ensure a weekly for EVERY candidate (covers candidates whose days were all
    # already current, so no day reprocessed and no weekly was triggered)
    all_cands: dict = {}
    for mp in prefixes:
        cps = _candidate_prefixes(mp)
        if cps:
            all_cands.setdefault(cps[0], mp)   # one meeting prefix per candidate
    cfg = _weekly_cfg()
    built = 0
    for cand, mp in all_cands.items():
        if force or not _weekly_exists(s3, bucket, cand):
            try:
                _rebuild_weekly(s3, bucket, mp, cfg)
                built += 1
            except Exception as exc:
                print(f"  weekly build FAILED for {cand}: {exc!r}")

    print(f"\nBackfill done: reprocessed {done}, failed {failed}, skipped {skip}; "
          f"weeklies ensured {built} (of {len(all_cands)} candidate(s)).")


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


# ── multi-MP4 (one meeting recorded in several parts) ────────────────────────
def _list_mp4s(s3, bucket: str, prefix: str) -> list:
    """(.mp4 key, size, LastModified) directly under prefix, ordered by
    LastModified ascending — Zoom delivers/uploads the parts of one session in
    chronological order, so this is their recording order."""
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents") or []:
            if o["Key"].lower().endswith(".mp4"):
                out.append((o["Key"], o.get("Size") or 0, o.get("LastModified")))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    out.sort(key=lambda x: (x[2] is None, x[2]))   # LastModified asc, None last
    return out


def _ffmpeg_duration(path: str):
    """Seconds of an MP4 via `ffmpeg -i` (parsed from stderr — no ffprobe needed,
    which the box may lack). None on failure."""
    import re
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=180)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception as exc:
        print(f"  [mp4:duration] {path}: {exc!r}")
    return None


def _vtt_span(vtt_path: str | None):
    """The real session length = the last VTT cue end (seconds). None if no/bad VTT."""
    if not vtt_path:
        return None
    try:
        from analyzer import transcript as T
        ends = [s.get("end") or 0 for s in T.parse_vtt(vtt_path)]
        return max(ends) if ends else None
    except Exception:
        return None


def _concat_mp4s(paths: list, out_path: str) -> str | None:
    """Concatenate MP4 parts into ONE continuous video (ffmpeg concat demuxer).
    Stream-copy first (fast; Zoom parts of one session share codec params), then
    fall back to re-encode. Returns out_path or None."""
    listf = out_path + ".txt"
    try:
        with open(listf, "w", encoding="utf-8") as fh:
            for p in paths:
                fh.write("file '%s'\n" % p.replace("'", "'\\''"))
    except OSError as exc:
        print(f"  [mp4:concat] could not write list: {exc!r}")
        return None
    for extra in (["-c", "copy"], []):   # copy, then re-encode
        try:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                "-f", "concat", "-safe", "0", "-i", listf, *extra, out_path],
                               capture_output=True, text=True, timeout=3600)
            if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            print(f"  [mp4:concat] ffmpeg {'copy' if extra else 're-encode'} rc={r.returncode}: "
                  f"{(r.stderr or '')[-300:]}")
        except Exception as exc:
            print(f"  [mp4:concat] {exc!r}")
    return None


def _prepare_video(s3, bucket: str, prefix: str, vtt_path: str | None, work: str) -> str | None:
    """Return a single local MP4 to analyse. If the meeting's MP4/ holds several
    parts (same meeting_id dropped + rejoined), download them all in recording
    order and CONCATENATE into one continuous session. If the several files are
    instead Zoom VIEWS (same time range, different layout — summed duration far
    exceeds the real session length), keep only the largest. One file → as-is.
    Returns None if MP4/ is empty (caller falls back)."""
    listing = _list_mp4s(s3, bucket, prefix + "MP4/")
    if not listing:
        return None
    if len(listing) == 1:
        return _download_key(s3, bucket, listing[0][0], work)

    parts = []   # (local_path, size)
    for i, (k, sz, _lm) in enumerate(listing):
        lp = _download_key(s3, bucket, k, os.path.join(work, f"mp4{i}"))
        if lp:
            parts.append((lp, sz))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0][0]

    paths = [p for p, _ in parts]
    total = sum(d for d in (_ffmpeg_duration(p) for p in paths) if d) or 0
    span = _vtt_span(vtt_path)
    if span and total and total > span * 1.5:
        print(f"  [mp4] {len(paths)} MP4s total ~{round(total)}s vs session ~{round(span)}s "
              f"-> looks like VIEWS; analysing the largest only")
        return max(parts, key=lambda x: x[1])[0]

    print(f"  [mp4] {len(paths)} MP4 parts of one meeting (drop/rejoin) total ~{round(total)}s "
          f"-> concatenating into ONE session")
    out = os.path.join(work, "in", "session-concat.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged = _concat_mp4s(paths, out)
    if merged:
        return merged
    print("  [mp4:concat] failed; using the largest single part")
    return max(parts, key=lambda x: x[1])[0]


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
    # S3 layout: Training/{Program}/{Trainer}/{Year}/{Month}/{Candidate}/{Date}/{Time}/{MeetingID}/
    # training-temp.json (when present) is authoritative; these path segments are
    # only the fallback for older meetings that have no temp file.
    parts = prefix.strip("/").split("/")
    return {
        "bucket": bucket, "prefix": prefix,
        "meeting_id": t.get("meeting_id") or (parts[-1] if parts else ""),
        "day": t.get("day"), "day_step_name": t.get("day_step_name"),
        "trainer": t.get("trainer") or (parts[2] if len(parts) > 2 else ""),
        "candidate": t.get("candidate") or (parts[5] if len(parts) > 5 else ""),
    }


# ── model API keys from Secrets Manager (so no key has to sit in .env) ───────
def _resolve_openai_key() -> None:
    """Fill OPENAI_API_KEY / TECHSARA_API_KEY from Secrets Manager when they are
    needed (per LLM_PROVIDERS / TECHSARA_WHISPER) and not already set in .env."""
    if os.environ.get("MOCK_OPENAI", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    providers = {p.strip().lower() for p in
                 (os.environ.get("LLM_PROVIDERS", "") or "techsara").split(",") if p.strip()}
    whisper = os.environ.get("TECHSARA_WHISPER", "").strip().lower() in ("1", "true", "yes", "on")
    if "openai" in providers:
        _secret_to_env("OPENAI_API_KEY", OPENAI_SECRET_NAME)
    if "techsara" in providers or whisper:
        _secret_to_env("TECHSARA_API_KEY", TECHSARA_SECRET_NAME)


def _secret_to_env(var: str, secret_name: str) -> None:
    if os.environ.get(var, "").strip() or not secret_name:
        return
    try:
        sm = boto3.client("secretsmanager", region_name=REGION)
        val = sm.get_secret_value(SecretId=secret_name).get("SecretString") or ""
        try:
            val = json.loads(val).get(var, val)    # plain string or {"VAR": "..."}
        except Exception:
            pass
        if val:
            os.environ[var] = val.strip()
            print(f"{var} loaded from Secrets Manager ({secret_name})")
    except Exception as exc:
        print(f"Could not read secret {secret_name} ({exc}); set {var} in .env or MOCK_OPENAI=true")


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
