"""Rebuild the HTML reports of meetings ALREADY analyzed, with the current report.py.

No re-analysis, no model calls: for every meeting folder under the prefix it
downloads each result*.json plus only the proof files it references, re-renders
the matching report*.html (evidence images + proof texts embedded, video
fallback — see analyzer/report.py) and uploads it over the old one. Idempotent:
safe to run again; a meeting that fails is reported and skipped.

  python -m worker.rebuild_reports --dry-run              # list what would change
  python -m worker.rebuild_reports                        # every meeting under Training/
  python -m worker.rebuild_reports --prefix "Training/Resume-Based/Divya_Prajapati/"

Runs on the EC2 worker (its IAM role can list/get/put Training/*).
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild existing report*.html files in S3")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default="Training/", help="only meetings under this key prefix")
    ap.add_argument("--dry-run", action="store_true", help="list the reports, change nothing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N meetings (0 = all)")
    args = ap.parse_args()

    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    totals = rebuild_all(s3, args.bucket, args.prefix, dry_run=args.dry_run, limit=args.limit)
    print(f"\nDone: {totals['meetings']} meeting(s), {totals['reports']} report(s) "
          f"{'would be ' if args.dry_run else ''}rebuilt, {totals['failed']} failed.")
    return 1 if totals["failed"] else 0


def rebuild_all(s3, bucket: str, prefix: str, dry_run: bool = False, limit: int = 0) -> dict:
    keys = set(_list_keys(s3, bucket, prefix))
    # meeting folder -> [(result json name, report html name)] for reports that exist
    meetings: dict[str, list[tuple[str, str]]] = {}
    for k in keys:
        folder, name = k.rsplit("/", 1) if "/" in k else ("", k)
        m = _RESULT.match(name)
        if not m:
            continue
        report = f"report{m.group(1) or ''}.html"
        if f"{folder}/{report}" in keys:
            meetings.setdefault(folder, []).append((name, report))

    totals = {"meetings": 0, "reports": 0, "failed": 0}
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
            for rel in _evidence_paths(result):
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
                           ExtraArgs={"ContentType": "text/html; charset=utf-8"})
            print(f"    {report_name}: rebuilt with {got} evidence file(s)")
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


def _list_keys(s3, bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents") or []:
            yield o["Key"]


if __name__ == "__main__":
    sys.exit(main())
