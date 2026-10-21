"""Weekly cumulative report.

Aggregates the five daily result.json files for ONE (trainer, candidate) pair
(Day 1-5) into a single week overview — week-result.json + week-report.html.

  - Numbers (scores, averaged effectiveness/candidate dimensions, flag tallies,
    cumulative coverage) are computed deterministically here. No AI.
  - A short week NARRATIVE (executive summary + trainer/candidate progression) is
    written by ONE optional GPT call (synthesize()).
  - Missing days are tolerated: the report is built from whatever days exist and
    the gaps are noted.

CLI:  python -m analyzer.weekly <dir-with-result.json | file1 file2 ...> [out_dir]
"""
import glob
import json
import os
import re
from statistics import mean

# dimension keys -> display label, in display order
TRAINER_DIMS = [
    ("clarity_1_10", "Explanation clarity"),
    ("examples_1_10", "Use of examples"),
    ("teaching_effectiveness_1_10", "Teaching effectiveness"),
    ("revision_summary_1_10", "Revision & summarization"),
    ("professionalism_1_10", "Professionalism"),
    ("engagement_1_10", "Kept candidate engaged"),
]
CANDIDATE_DIMS = [
    ("attentiveness_1_10", "Attentiveness / engagement"),
    ("participation_1_10", "Participation"),
    ("communication_1_10", "Communication"),
    ("conduct_1_10", "Conduct / behaviour"),
    ("understanding_1_10", "Understanding shown"),
]


# ── load ─────────────────────────────────────────────────────────────────────
def load_daily(src) -> list[dict]:
    """src = a directory (glob **/result.json), or a list of result.json paths.
    Returns the daily result dicts that have a day number, sorted by day."""
    paths = []
    if isinstance(src, (list, tuple)):
        paths = list(src)
    elif os.path.isdir(src):
        paths = sorted(glob.glob(os.path.join(src, "**", "result.json"), recursive=True))
    elif os.path.isfile(src):
        paths = [src]
    out = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and (d.get("meeting") or {}).get("day") is not None:
                d["_path"] = p
                out.append(d)
        except Exception as exc:
            print(f"weekly: could not load {p}: {exc}")
    # keep one per day (latest file wins), then sort by day
    by_day = {}
    for d in out:
        by_day[int((d["meeting"] or {}).get("day"))] = d
    return [by_day[k] for k in sorted(by_day)]


# ── helpers ──────────────────────────────────────────────────────────────────
def _num(x):
    return x if isinstance(x, (int, float)) else None


def _date_of(d) -> str | None:
    m = d.get("meeting") or {}
    for v in (m.get("scheduled_start"), m.get("analyzed_at")):
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    mt = re.search(r"(\d{4}-\d{2}-\d{2})", m.get("s3_prefix") or "")
    return mt.group(1) if mt else None


def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(mean(vals), 1) if vals else None


def _scored(d):
    return [x for x in ((d.get("scoring") or {}).get("deductions") or [])
            if isinstance(x, dict) and (x.get("points") or 0) != 0]


# ── aggregate ────────────────────────────────────────────────────────────────
def aggregate(dailies: list[dict]) -> dict:
    first = dailies[0] if dailies else {}
    m0 = first.get("meeting") or {}
    days_found = [int((d["meeting"] or {}).get("day")) for d in dailies]
    days_missing = [n for n in range(1, 6) if n not in days_found]

    per_day = []
    integ_vals, cov_vals = [], []
    cum_weighted, cum_total = 0.0, 0
    flag_tally: dict = {}
    trainer_rows, cand_rows = [], []
    daily_summaries = []

    for d in dailies:
        m = d.get("meeting") or {}
        sc = d.get("scoring") or {}
        day = int(m.get("day"))
        integ = _num(sc.get("session_integrity_score"))
        cov = _num(sc.get("trainer_coverage_score"))
        if integ is not None:
            integ_vals.append(integ)
        if cov is not None:
            cov_vals.append(cov)

        ca = (d.get("transcript") or {}).get("coverage_analysis") or {}
        rows = [c for c in (ca.get("coverage") or []) if isinstance(c, dict)]
        n_cov = sum(1 for c in rows if c.get("status") == "covered")
        n_par = sum(1 for c in rows if c.get("status") == "partial")
        n_miss = sum(1 for c in rows if c.get("status") == "not_covered")
        total = ca.get("expected_section_count") or len(rows)
        cum_weighted += n_cov + 0.5 * n_par
        cum_total += total or 0
        missed_sections = [c.get("section") for c in rows if c.get("status") == "not_covered"]

        flags = []
        flags_detail = []
        for x in _scored(d):
            r = x.get("reason")
            flags.append(r)
            flags_detail.append({"reason": r, "points": x.get("points")})
            flag_tally.setdefault(r, []).append(day)

        planned = _planned_minutes(d)
        actual = round((m.get("duration_sec") or 0) / 60.0, 1)
        length_pct = round(100 * actual / planned) if planned else None

        desc = d.get("meeting_description") or {}
        te = desc.get("trainer_effectiveness") or {}
        cp = desc.get("candidate_profile") or {}
        trainer_rows.append({"day": day, **te})
        cand_rows.append({"day": day, "camera_pct": ((d.get("video") or {}).get("camera") or {})
                          .get("candidate", {}).get("on_camera_pct_of_speaking"), **cp})

        topics = [{"section": c.get("section"), "status": c.get("status"),
                   "expected": c.get("expected_minutes"), "spent": c.get("approx_minutes_spent"),
                   "note": c.get("depth_note")} for c in rows]
        per_day.append({
            "day": day, "date": _date_of(d), "integrity": integ, "coverage": cov,
            "length_pct": length_pct, "tier": sc.get("tier"),
            "day_title": m.get("day_title"), "day_step_name": m.get("day_step_name"),
            "covered": n_cov, "partial": n_par, "missed": n_miss, "total": total,
            "missed_sections": missed_sections, "flags": flags, "flags_detail": flags_detail,
            "topics": topics,
            "exec_summary": desc.get("executive_summary"),
            "overview": desc.get("overview"),
            "trainer_overall": te.get("overall_1_10"), "trainer_summary": te.get("summary"),
            "candidate_overall": cp.get("overall_1_10"), "candidate_summary": cp.get("summary"),
            "report": "report.html", "report_dir": os.path.dirname(d.get("_path", "")),
        })
        if desc.get("executive_summary"):
            daily_summaries.append({"day": day, "summary": desc["executive_summary"],
                                    "trainer": te.get("summary"), "candidate": cp.get("summary")})

    week_coverage = round(100 * cum_weighted / cum_total) if cum_total else None

    # averaged dimensions + per-day values
    def _dim_avgs(rows, keys):
        out = []
        for k, label in keys:
            per = [{"day": r["day"], "v": _num(r.get(k))} for r in rows]
            out.append({"key": k, "label": label, "avg": _avg([p["v"] for p in per]), "per_day": per})
        return out

    trainer_overall = _avg([_num(r.get("overall_1_10")) for r in trainer_rows])
    cand_overall = _avg([_num(r.get("overall_1_10")) for r in cand_rows])

    # progression: first vs last present value for key candidate dims
    def _progress(rows, key):
        present = [(_num(r.get(key)), r["day"]) for r in rows if _num(r.get(key)) is not None]
        if len(present) < 2:
            return None
        return {"from": present[0][0], "from_day": present[0][1],
                "to": present[-1][0], "to_day": present[-1][1],
                "delta": round(present[-1][0] - present[0][0], 1)}

    return {
        "meta": {
            "candidate": m0.get("candidate_name"), "trainer": m0.get("trainer_name"),
            "program": m0.get("program") or m0.get("day_title"),
            "days_found": days_found, "days_missing": days_missing,
            "date_from": per_day[0]["date"] if per_day else None,
            "date_to": per_day[-1]["date"] if per_day else None,
        },
        "rolled": {
            "week_integrity": round(mean(integ_vals)) if integ_vals else None,
            "week_coverage": week_coverage,
            "avg_integrity": _avg(integ_vals), "avg_coverage": _avg(cov_vals),
        },
        "per_day": per_day,
        "trainer": {"overall_avg": trainer_overall, "dims": _dim_avgs(trainer_rows, TRAINER_DIMS),
                    "strengths": _recurring(trainer_rows, "strengths"),
                    "improvements": _recurring(trainer_rows, "improvements")},
        "candidate": {"overall_avg": cand_overall, "dims": _dim_avgs(cand_rows, CANDIDATE_DIMS),
                      "progression": {k: _progress(cand_rows, k)
                                      for k in ("understanding_1_10", "communication_1_10", "overall_1_10")},
                      "camera_per_day": [{"day": r["day"], "pct": _num(r.get("camera_pct"))} for r in cand_rows],
                      "strengths": _recurring(cand_rows, "strengths"),
                      "improvements": _recurring(cand_rows, "improvements")},
        "flag_tally": flag_tally,
        "daily_summaries": daily_summaries,
    }


def _planned_minutes(d):
    for f in d.get("flags") or []:
        if isinstance(f, dict) and f.get("type") == "session_too_short":
            return f.get("planned_minutes")
    m = re.search(r"of (\d+) planned", str((d.get("summary") or {}).get("duration", "")))
    return int(m.group(1)) if m else None


def _recurring(rows, key, cap=6):
    """Count how often each strength/improvement phrase appears across the days."""
    counts: dict = {}
    for r in rows:
        for x in (r.get(key) or []):
            if isinstance(x, str) and x.strip():
                k = x.strip()
                counts[k] = counts.get(k, 0) + 1
    return [{"text": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])][:cap]


# ── AI narrative (one call) ──────────────────────────────────────────────────
WEEK_SYSTEM = (
    "You are a quality-assurance lead writing a concise WEEKLY training summary for one trainer and one "
    "candidate, from five daily session summaries. Be factual and neutral. Return ONLY a JSON object."
)


def synthesize(agg: dict, oa) -> dict:
    if oa is None:
        return {}
    meta = agg["meta"]
    lines = []
    for s in agg.get("daily_summaries", []):
        lines.append(f"Day {s['day']}: {s.get('summary','')}")
        if s.get("trainer"):
            lines.append(f"  trainer: {s['trainer']}")
        if s.get("candidate"):
            lines.append(f"  candidate: {s['candidate']}")
    rolled = agg["rolled"]
    user = (
        f"TRAINER: {meta.get('trainer')}\nCANDIDATE: {meta.get('candidate')}\n"
        f"PROGRAM: {meta.get('program')}\nDAYS PRESENT: {meta.get('days_found')} "
        f"(missing: {meta.get('days_missing')})\n"
        f"WEEK AVERAGES: integrity {rolled.get('avg_integrity')}/100, coverage {rolled.get('avg_coverage')}%, "
        f"cumulative coverage {rolled.get('week_coverage')}%\n\n"
        "DAILY SUMMARIES:\n" + "\n".join(lines) + "\n\n"
        "Respond with JSON of exactly this shape:\n"
        "{\n"
        '  "week_executive_summary": "<4-6 sentences: how the week went overall, what the trainer '
        'focused on across the days, and the candidate\'s outcome and progression>",\n'
        '  "trainer_week_summary": "<3-5 sentences on the trainer across the week: consistency, '
        'strengths, and what to improve>",\n'
        '  "trainer_impact": "<2-3 sentences on the trainer\'s overall impact on the candidate across '
        'the week — did the candidate visibly improve/learn because of the trainer>",\n'
        '  "candidate_week_summary": "<3-5 sentences on the candidate across the week: progression from '
        'the first to the last day, strengths and remaining gaps>",\n'
        '  "candidate_acknowledgement": "<1-3 sentences: any explicit feedback or acknowledgement the '
        'candidate gave during the week (e.g. found the sessions helpful, confirmed understanding, raised '
        'doubts or concerns). If the candidate gave no explicit feedback, say so.>"\n'
        "}"
    )
    try:
        r = oa.chat_json(WEEK_SYSTEM, user)
    except Exception as exc:
        print(f"weekly: synthesis failed ({exc})")
        return {}
    if not isinstance(r, dict):
        return {}
    return {k: (r.get(k).strip() if isinstance(r.get(k), str) and r.get(k).strip() else None)
            for k in ("week_executive_summary", "trainer_week_summary", "trainer_impact",
                      "candidate_week_summary", "candidate_acknowledgement")}


# ── build ────────────────────────────────────────────────────────────────────
def build(src, out_dir: str, oa=None) -> dict | None:
    dailies = load_daily(src)
    if not dailies:
        print("weekly: no daily result.json with a day number found")
        return None
    agg = aggregate(dailies)
    agg["narrative"] = synthesize(agg, oa)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "week-result.json"), "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, ensure_ascii=False)
    from . import weekly_report as WR
    html_path = os.path.join(out_dir, "week-report.html")
    WR.write(agg, html_path)
    print(f"weekly: wrote {html_path} (days {agg['meta']['days_found']})")
    return agg


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("usage: python -m analyzer.weekly <dir | result.json ...> [out_dir]")
        raise SystemExit(2)
    out = args[-1] if (len(args) > 1 and not args[-1].endswith(".json")
                       and not os.path.isdir(args[-1])) else "."
    srcs = args[:-1] if out != "." and out == args[-1] else args
    src = srcs[0] if len(srcs) == 1 else srcs
    oa = None
    try:
        from .config import load_config
        from .openai_client import make_clients
        oa = make_clients(load_config())[0]
    except Exception as exc:
        print(f"weekly: no LLM client ({exc}); building without the AI narrative")
    build(src, out, oa)
