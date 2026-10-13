"""Rebuild the HTML reports of meetings ALREADY analyzed, with the current report.py.

No re-analysis, no model calls: for every meeting folder under the prefix it
downloads each result*.json plus only the proof files it references, re-renders
the matching report*.html (evidence images, proof texts and a compact copy of
the analysis video embedded — see analyzer/report.py) and uploads it over the old
one. A meeting that fails is reported and skipped.

RESUMABLE: every report this script (or the worker) uploads is tagged with S3
metadata report-version = analyzer.report.REPORT_VERSION. Reports already at the
current version are skipped, so an interrupted run just continues; bumping
REPORT_VERSION makes the next run redo every older report. --force redoes all.

  python -m worker.rebuild_reports --dry-run              # list what would change
  python -m worker.rebuild_reports                        # every meeting under Training/
  python -m worker.rebuild_reports --prefix "Training/Resume-Based/Divya_Prajapati/"

Runs on the EC2 worker (its IAM role can list/get/put Training/*). Run it with
nohup so a closed SSM session does not kill it (see CLAUDE.md).
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile

from analyzer import report as R

BUCKET = os.environ.get("S3_BUCKET", "zoom-automation-bucket").strip() or "zoom-automation-bucket"
REGION = os.environ.get("AWS_REGION", "us-east-1").strip() or "us-east-1"
_RESULT = re.compile(r"^result(-[^/]+)?\.json$")       # result.json, result-<model>.json
_EVIDENCE_EXT = R._IMG_EXT + (".txt",)
_VERSION_META = "report-version"


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild existing report*.html files in S3")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default="Training/", help="only meetings under this key prefix")
    ap.add_argument("--dry-run", action="store_true", help="list the reports, change nothing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N meetings (0 = all)")
    ap.add_argument("--force", action="store_true", help="also redo reports already rebuilt")
    args = ap.parse_args()
    # line-buffered: under `nohup ... > log` progress shows in the log immediately
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    other = _other_run_pid()
    if other:
        print(f"Another rebuild is already running (pid {other}). Watch it with "
              f"`tail -f /tmp/rebuild.log`, or stop it first: sudo kill {other}")
        return 2

    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    _hold_instance()
    try:
        totals = rebuild_all(s3, args.bucket, args.prefix, dry_run=args.dry_run, limit=args.limit,
                             force=args.force)
    finally:
        _release_instance()
    print(f"\nDone: {totals['meetings']} meeting(s), {totals['reports']} report(s) "
          f"{'would be ' if args.dry_run else ''}rebuilt, {totals['skipped']} already current "
          f"(skipped), {totals['failed']} failed.")
    return 1 if totals["failed"] else 0


def rebuild_all(s3, bucket: str, prefix: str, dry_run: bool = False, limit: int = 0,
                force: bool = False) -> dict:
    print(f"Listing s3://{bucket}/{prefix} ...")
    keys = set(_list_keys(s3, bucket, prefix))
    n_reports = sum(1 for k in keys if k.rsplit("/", 1)[-1].startswith("report")
                    and k.endswith(".html"))
    print(f"{len(keys)} objects, {n_reports} report(s); checking which are already at "
          f"version {R.REPORT_VERSION}{' (skipped: --force)' if force else ''} ...")
    # meeting folder -> [(result json name, report html name)] for reports still to rebuild
    meetings: dict[str, list[tuple[str, str]]] = {}
    skipped = 0
    for k in keys:
        folder, name = k.rsplit("/", 1) if "/" in k else ("", k)
        m = _RESULT.match(name)
        if not m:
            continue
        report = f"report{m.group(1) or ''}.html"
        rkey = f"{folder}/{report}"
        if rkey not in keys:
            continue
        if not force and _is_current(s3, bucket, rkey):
            skipped += 1
            continue
        meetings.setdefault(folder, []).append((name, report))
    if skipped:
        print(f"{skipped} report(s) already at version {R.REPORT_VERSION} - skipped (--force to redo)")

    totals = {"meetings": 0, "reports": 0, "failed": 0, "skipped": skipped}
    for folder in sorted(meetings):
        if limit and totals["meetings"] >= limit:
            break
        totals["meetings"] += 1
        pairs = sorted(meetings[folder])
        print(f"[{totals['meetings']}/{len(meetings)}] {folder}/  ->  {', '.join(r for _, r in pairs)}")
        if dry_run:
            totals["reports"] += len(pairs)
            continue
        try:
            totals["reports"] += _rebuild_meeting(s3, bucket, folder, pairs, keys)
        except Exception as exc:          # one bad meeting must not stop the backfill
            totals["failed"] += 1
            print(f"    FAILED: {exc!r}")
    return totals


def _rebuild_meeting(s3, bucket: str, folder: str, pairs, keys: set) -> int:
    work = tempfile.mkdtemp(prefix="rebuild-report-")
    try:
        done = 0
        for result_name, report_name in pairs:
            local_json = os.path.join(work, result_name)
            s3.download_file(bucket, f"{folder}/{result_name}", local_json)
            with open(local_json, encoding="utf-8") as fh:
                result = json.load(fh)
            got = 0
            for rel in _evidence_paths(result) | _video_path(result):
                key = f"{folder}/{rel}"
                dst = os.path.normpath(os.path.join(work, rel))
                if key not in keys or not dst.startswith(os.path.normpath(work) + os.sep):
                    continue
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    s3.download_file(bucket, key, dst)
                got += 1
            out = R.write(result, os.path.join(work, report_name))
            s3.upload_file(out, bucket, f"{folder}/{report_name}",
                           ExtraArgs={"ContentType": "text/html; charset=utf-8",
                                      "Metadata": {_VERSION_META: R.REPORT_VERSION}})
            print(f"    {report_name}: rebuilt with {got} evidence file(s), "
                  f"{os.path.getsize(out) / 2**20:.1f} MB")
            done += 1
        return done
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _evidence_paths(result: dict) -> set:
    """Proof files the report shows: images + proof texts of every deduction/flag,
    and the coverage table."""
    paths = set()
    items = list((result.get("scoring") or {}).get("deductions") or []) + list(result.get("flags") or [])
    for it in items:
        if isinstance(it, dict):
            paths.update(p for p in it.get("proof") or [] if isinstance(p, str))
    cov = ((result.get("transcript") or {}).get("coverage_analysis") or {}).get("proof")
    if isinstance(cov, str):
        paths.add(cov)
    return {p for p in paths if p.lower().endswith(_EVIDENCE_EXT) and not os.path.isabs(p)}


# Same path as worker.worker.MAINTENANCE_LOCK (not imported: that module pulls in the
# whole analyzer). While this file names our live pid, the worker's idle timer will
# not stop the instance — so a deploy/restart of the worker can't cut the run short.
_LOCK = (os.environ.get("MAINTENANCE_LOCK", "").strip()
         or os.path.join(tempfile.gettempdir(), "training-audit-maintenance.lock"))


def _other_run_pid() -> int | None:
    """pid of another live rebuild holding the lock (two runs would redo the same
    meetings and interleave one log), else None. Stale locks are ignored."""
    try:
        with open(_LOCK) as fh:
            pid = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid() or os.name == "nt":
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return pid
    except OSError:
        return None
    try:                                  # pid reuse: only count it if it is a rebuild
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return pid if b"rebuild_reports" in fh.read() else None
    except OSError:
        return pid


def _hold_instance() -> None:
    try:
        with open(_LOCK, "w") as fh:
            fh.write(str(os.getpid()))
        print(f"(holding the instance awake while this runs: {_LOCK})")
    except OSError as exc:
        print(f"(could not write {_LOCK}: {exc}; stop the worker service so it can't idle-stop)")


def _release_instance() -> None:
    try:
        with open(_LOCK) as fh:
            mine = fh.read().strip() == str(os.getpid())
        if mine:
            os.remove(_LOCK)
    except OSError:
        pass


def _is_current(s3, bucket: str, key: str) -> bool:
    """The report in S3 was rendered by the current report.py."""
    try:
        meta = s3.head_object(Bucket=bucket, Key=key).get("Metadata") or {}
    except Exception:
        return False
    return meta.get(_VERSION_META) == R.REPORT_VERSION


def _video_path(result: dict) -> set:
    """The analysis video, so the report can embed a compact copy of it."""
    vid = (result.get("video") or {}).get("analysis_video")
    return {vid} if isinstance(vid, str) and vid.lower().endswith(".mp4") and not os.path.isabs(vid) else set()


def _list_keys(s3, bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents") or []:
            yield o["Key"]


if __name__ == "__main__":
    sys.exit(main())
