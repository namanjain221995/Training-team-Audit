"""Proof folder — every deduction and flag gets its evidence on disk.

Layout (under /data/output/proof, i.e. ./output/proof on the host):

  proof/
    README.txt                       what this folder is / how to read it
    <deduction_reason>/              one folder per applied deduction
      proof.txt                      reason, points, quote, timestamp, numbers
      frame_*.jpg                    evidence frames (from the video, if any)
    section_coverage/
      coverage_report.txt            per-section: status, planned vs spent minutes, quote

Each result.json deduction gains a "proof" list of paths (relative to output/)
pointing at its receipts. Runs before the optional frames-dir cleanup so the
copies survive SAVE_EVIDENCE_FRAMES=false.
"""
import json
import os
import shutil


def build(result: dict, cfg, frames: list[dict]) -> None:
    root = cfg.proof_dir
    shutil.rmtree(root, ignore_errors=True)   # idempotent re-runs
    os.makedirs(root, exist_ok=True)
    _write(os.path.join(root, "README.txt"), _readme(result))

    meeting = result.get("meeting", {})
    # frames live in the container-local work dir during the run; resolve
    # detector-named evidence frames through this map, not the output dir
    frame_paths = {os.path.basename(f["path"]): f["path"]
                   for f in frames if isinstance(f, dict) and f.get("path")}
    for d in result["scoring"]["deductions"]:
        d["proof"] = _proof_for_deduction(d, meeting, root, frame_paths, frames)

    # give flags the same pointers as their matching deduction
    by_reason = {d["reason"]: d.get("proof") for d in result["scoring"]["deductions"]}
    for f in result.get("flags", []):
        p = by_reason.get(f.get("type"))
        if p:
            f["proof"] = p

    cov = result.get("transcript", {}).get("coverage_analysis", {})
    if cov.get("coverage"):
        path = _coverage_report(cov, meeting, root)
        cov["proof"] = path

    result["proof_dir"] = "proof/"


# ── per-deduction ────────────────────────────────────────────────────────────
def _proof_for_deduction(d: dict, meeting: dict, root: str,
                         frame_paths: dict, frames: list[dict]) -> list[str]:
    sub = os.path.join(root, d["reason"])
    os.makedirs(sub, exist_ok=True)
    rel = lambda name: f"proof/{d['reason']}/{name}"
    paths = []

    lines = [
        f"RED FLAG : {d['reason']}",
        f"POINTS   : {d['points']}   (severity: {d.get('severity')})",
        f"SESSION  : meeting {meeting.get('meeting_id')} | Day {meeting.get('day')} | "
        f"trainer {meeting.get('trainer_name')} | candidate {meeting.get('candidate_name')}",
        f"EVIDENCE : {d.get('evidence')}",
    ]
    if d.get("approx_time"):
        lines.append(f"WHERE    : ~{d['approx_time']} into the recording")
    _write(os.path.join(sub, "proof.txt"), "\n".join(lines) + "\n")
    paths.append(rel("proof.txt"))

    # evidence frames named by the detector (gaze / identity)
    for name in (d.get("evidence_frames") or []):
        src = frame_paths.get(name)
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(sub, name))
            paths.append(rel(name))

    # transcript-sourced flags: attach the frames around the quoted moment
    if not d.get("evidence_frames") and d.get("approx_time") and frames:
        t = _ts_to_sec(d["approx_time"])
        for dt in (-5, 0, 5):
            fr = _nearest_frame(frames, t + dt)
            if fr and os.path.exists(fr["path"]):
                name = os.path.basename(fr["path"])
                dst = os.path.join(sub, name)
                if not os.path.exists(dst):
                    shutil.copy2(fr["path"], dst)
                    paths.append(rel(name))
    return paths


# ── coverage report ──────────────────────────────────────────────────────────
def _coverage_report(cov: dict, meeting: dict, root: str) -> str:
    sub = os.path.join(root, "section_coverage")
    os.makedirs(sub, exist_ok=True)
    lines = [
        f"SECTION COVERAGE — Day {meeting.get('day')}: {meeting.get('day_title')}",
        f"session label (Salesforce): {meeting.get('day_step_name')}",
        f"trainer coverage score: {cov.get('coverage_pct')}%",
        "",
        f"{'SECTION':<38} {'STATUS':<13} {'PLAN':>6} {'SPENT':>6}  EVIDENCE",
        "-" * 110,
    ]
    for c in cov.get("coverage") or []:
        if not isinstance(c, dict):
            continue
        plan = c.get("expected_minutes")
        spent = c.get("approx_minutes_spent")
        ev = c.get("evidence") or "-"
        at = f" @{c['approx_time']}" if c.get("approx_time") else ""
        lines.append(
            f"{(c.get('section') or '?'):<38} {(c.get('status') or '?'):<13} "
            f"{_fmt_min(plan):>6} {_fmt_min(spent):>6}  \"{ev}\"{at}"
        )
    if cov.get("under_covered_key_sections"):
        lines += ["", "KEY SECTIONS RUSHED/SKIPPED (>=25 planned min): "
                  + ", ".join(cov["under_covered_key_sections"])]
    _write(os.path.join(sub, "coverage_report.txt"), "\n".join(lines) + "\n")
    return "proof/section_coverage/coverage_report.txt"



# ── helpers ──────────────────────────────────────────────────────────────────
def _readme(result: dict) -> str:
    m = result.get("meeting", {})
    s = result.get("scoring", {})
    return (
        "PROOF FOLDER\n"
        "============\n"
        f"Meeting {m.get('meeting_id')} | Day {m.get('day')} ({m.get('day_title')})\n"
        f"Trainer: {m.get('trainer_name')} | Candidate: {m.get('candidate_name')}\n"
        f"Scores: coverage {s.get('trainer_coverage_score')}% | "
        f"integrity {s.get('session_integrity_score')}/100 ({s.get('tier')})\n\n"
        "One folder per red flag, each holding proof.txt (the quote/numbers that\n"
        "triggered it) and any evidence frames from the video.\n"
        "section_coverage/ holds the per-section coverage vs the day plan.\n"
    )


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _fmt_min(v) -> str:
    if v is None:
        return "?"
    try:
        return f"{round(float(v))}m"
    except (TypeError, ValueError):
        return "?"


def _nearest_frame(frames: list[dict], t: float):
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f["time_sec"] - t))


def _ts_to_sec(ts: str) -> float:
    try:
        parts = [float(p) for p in str(ts).split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[-3] * 3600 + parts[-2] * 60 + parts[-1]
    except (ValueError, AttributeError):
        return 0.0
