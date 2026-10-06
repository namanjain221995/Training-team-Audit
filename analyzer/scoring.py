"""Scoring — turns raw signals into two numbers, each with its receipts.

  trainer_coverage_score   : % of the day's planned sections actually covered
  session_integrity_score  : 100 = clean; deductions per detected red flag,
                             each carrying the evidence that triggered it.

The score answers "should a human review this session", NOT "how good is the
candidate". Weights are a starting point and meant to be tuned.
"""
import math

# reason -> (points, severity). Deductions are applied when the flag is present.
WEIGHTS = {
    "proxy_interview_coaching": (-10, "high"),    # repeated proxy coaching; single mention -> PROXY_SINGLE
    "fabricated_experience":    (-30, "critical"),
    "scripted_deception":       (-20, "high"),
    "person_change_midsession": (-40, "critical"),
    "person_appeared_midsession": (-25, "high"),   # group: unexpected face appears mid-session (past grace)
    "candidate_camera_off":     (-20, "high"),
    "gaze_fixed_offscreen":     (-15, "medium"),
    "non_english_heavy":        (-5,  "low"),
    "key_section_rushed":       (-10, "medium"),
    "session_too_short":        (-15, "high"),   # escalates to -25/critical below SESSION_SEVERE_RATIO
}

CAMERA_OFF_THRESHOLD = 0.5   # candidate on-camera < 50% of their speaking time
GAZE_THRESHOLD       = 0.6   # reading_likelihood >= 0.6 counts as a flag
NON_ENGLISH_THRESHOLD = 120  # seconds of non-English before the (all-speakers) flag
# Trainer teaching heavily in non-English: grace 10% of the trainer's talk, then each
# further 5% band costs 5 more points (>10% -> -5, >15% -> -10, >20% -> -15, ...).
TRAINER_NE_GRACE_PCT   = 10
TRAINER_NE_BRACKET_PCT = 5
TRAINER_NE_BRACKET_PTS = 5
# Coverage / depth (integrated — replaces key_section_rushed)
COVERAGE_VERY_LESS_PCT = 50   # < this % covered = "very less" -> deduct missed topics even at full length
MISSED_TOPIC_PTS       = -5   # per planned topic not covered (when session short OR coverage < 50%)
SHALLOW_TOPIC_PTS      = -4   # per topic covered but below the expected depth
SESSION_SHORT_GRACE_MIN   = 10  # minutes a session may run short with NO deduction
SESSION_SHORT_BRACKET_MIN = 5   # each further 5-min bracket short...
SESSION_SHORT_BRACKET_PTS = 5   # ...costs another 5 points (10-15 short=-15, 15-20=-20, ...)
PROXY_SINGLE = (-5, "medium")  # proxy mentioned exactly once in the whole session
# Trainer camera off while speaking. Only the LONGEST CONTINUOUS off-stretch scores,
# and screen-only (unassessable) time is excluded (never penalise a screen-share-only
# moment). Grace 5 min; a stretch of 5-8 min = -5, longer than 8 min = -10.
TRAINER_CAM_OFF_GRACE_SEC = 300
TRAINER_CAM_OFF_BAND_SEC  = 480
TRAINER_CAM_OFF_PTS_1     = -5
TRAINER_CAM_OFF_PTS_2     = -10
# Trainer reading from screen WHILE EXPLAINING (not during a shared document).
TRAINER_READING_THRESHOLD = 0.6
TRAINER_READING_PTS       = -4
# Trainer (host) joined later than the SCHEDULED start. Grace 5 min, then each
# further 5-min band costs 5 more points (5-10 late=-5, 10-15=-10, 15-20=-15...).
# trainer_late_min is computed by the Lambda (Zoom scheduled start vs host join)
# and read from training-temp.json; null/absent -> no deduction.
TRAINER_LATE_GRACE_MIN   = 5
TRAINER_LATE_BRACKET_MIN = 5
TRAINER_LATE_BRACKET_PTS = 5

# Integrity flag types that are DETECTED but intentionally NOT acted on — no deduction,
# and not surfaced as a flag/proof. Owner policy (2026-09-07): trainer coaching of
# fabricated experience OR scripted/deceptive answers must NOT reduce the session
# integrity score. Weights are kept in WEIGHTS above, so re-enabling scoring for any of
# these is just removing the type from this set.
IGNORED_INTEGRITY_FLAGS = {"fabricated_experience", "scripted_deception"}

# Of everything DETECTED, only these reasons actually change the integrity score. Every
# other detected issue is still recorded and shown as a 0-point review flag, so a human
# still sees it, but the score is unaffected. Add a reason here to make it actually
# score. Everything not listed stays a 0-point review flag (integrity unaffected).
# Currently scored:
#   - session_too_short           : ran short of the scheduled time (bracket rule below)
#   - person_change_midsession    : a DIFFERENT candidate on camera > the grace window
#                                   (>2 min) in a 1:1 session -> proxy swap (-40)
#   - trainer_non_english_heavy   : trainer taught heavily in non-English (>10% of talk,
#                                   graduated -5 per 5% band)
#   - topics_missed               : planned topics not covered when short OR coverage <50% (-5 each)
#   - topics_shallow              : topics covered but below the expected depth (-4 each)
#   - trainer_camera_off          : trainer camera off > 5 min continuously while
#                                   speaking (screen-only time excluded) -> -5 / -10
#   - trainer_reading_screen      : trainer reading from screen while explaining,
#                                   not during a shared document -> -4 each
#   - trainer_joined_late         : trainer (host) joined > 5 min after the
#                                   scheduled start -> -5 per 5-min band
SCORED_FLAGS = {"session_too_short", "person_change_midsession", "trainer_non_english_heavy",
                "topics_missed", "topics_shallow", "trainer_camera_off", "trainer_reading_screen",
                "trainer_joined_late"}
# NOTE (multi-recording): if a host stop/restarts, one logical session lands in
# several chunks and EACH chunk looks "short". The merge step (§9 CLAUDE.md)
# must recompute this from summed chunk durations — per-chunk raw numbers are
# stored in flags for exactly that reason.


def assemble(meta: dict, transcript: dict, video: dict, cfg) -> dict:
    deductions = []
    flags = []

    # ── session ran shorter than the scheduled time ─────────────────────────
    # Grace: up to SESSION_SHORT_GRACE_MIN minutes short costs nothing. Beyond that,
    # each further 5-min bracket short adds 5 points (10-15 min short = -15,
    # 15-20 = -20, 20-25 = -25, ...). Computed from raw seconds so rounding can't flip a
    # boundary and a 0-duration (broken) session still flags.
    from .config import day_sections
    session_ran_short = False
    planned_min = (day_sections(cfg) or {}).get("total_minutes") or 0
    dur_sec = meta.get("duration_sec") or 0
    actual_min = round(dur_sec / 60.0, 1)
    if planned_min:
        short_min = planned_min - (dur_sec / 60.0)
        session_ran_short = short_min > SESSION_SHORT_GRACE_MIN
        if short_min > SESSION_SHORT_GRACE_MIN:
            pts = -SESSION_SHORT_BRACKET_PTS * math.ceil(short_min / SESSION_SHORT_BRACKET_MIN)
            sev = "critical" if pts <= -30 else ("high" if pts <= -20 else "medium")
            deductions.append({
                "reason": "session_too_short", "points": pts, "severity": sev,
                "evidence": f"session ran {actual_min} of {planned_min} planned minutes "
                            f"({round(short_min)} min short)",
            })
            flags.append({"source": "transcript", "type": "session_too_short",
                          "planned_minutes": planned_min, "actual_minutes": actual_min,
                          "minutes_short": round(short_min, 1)})

    # ── trainer joined late (vs scheduled start; from training-temp.json) ────
    # Grace TRAINER_LATE_GRACE_MIN minutes; beyond that each further 5-min band
    # adds 5 points. trainer_late_min is null on local .env runs or when the
    # Lambda couldn't compute it (instant meeting / report not ready) -> skipped.
    late_min = getattr(cfg, "trainer_late_min", None)
    if isinstance(late_min, (int, float)) and late_min > TRAINER_LATE_GRACE_MIN:
        n = math.ceil((late_min - TRAINER_LATE_GRACE_MIN) / TRAINER_LATE_BRACKET_MIN)
        pts = -TRAINER_LATE_BRACKET_PTS * n
        sev = "critical" if pts <= -30 else ("high" if pts <= -20 else "medium")
        deductions.append({
            "reason": "trainer_joined_late", "points": pts, "severity": sev,
            "evidence": f"trainer joined ~{round(late_min)} min after the scheduled start",
            "approx_time": getattr(cfg, "trainer_join", None)})
        flags.append({"source": "zoom", "type": "trainer_joined_late",
                      "trainer_late_min": late_min,
                      "scheduled_start": getattr(cfg, "scheduled_start", None),
                      "trainer_join": getattr(cfg, "trainer_join", None)})

    # ── transcript integrity flags (from GPT-4o) ────────────────────────────
    cov = transcript.get("coverage_analysis", {})
    for f in cov.get("integrity_flags", []) or []:
        if not isinstance(f, dict):   # shape-drifted model output: keep it visible, don't crash
            flags.append({"source": "transcript", "type": "other", "raw": f})
            continue
        ftype = f.get("type", "other")
        if ftype in IGNORED_INTEGRITY_FLAGS:   # owner policy: detected but not acted on
            continue
        if ftype in WEIGHTS:
            pts, sev = WEIGHTS[ftype]
            deductions.append({"reason": ftype, "points": pts, "severity": sev,
                               "evidence": f.get("evidence"), "approx_time": f.get("approx_time")})
        flags.append({"source": "transcript", **f})

    # ── language ────────────────────────────────────────────────────────────
    lang = transcript.get("metrics", {}).get("language", {})
    if lang.get("available") and lang.get("non_english_sec", 0) >= NON_ENGLISH_THRESHOLD:
        pts, sev = WEIGHTS["non_english_heavy"]
        deductions.append({"reason": "non_english_heavy", "points": pts, "severity": sev,
                           "evidence": f"{lang['non_english_sec']}s non-English"})
        flags.append({"source": "transcript", "type": "non_english_heavy",
                      "non_english_sec": lang["non_english_sec"]})

    # ── trainer teaching heavily in non-English ─────────────────────────────
    # Primary: GPT's estimate of the % of the TRAINER's speech that was non-English.
    # Fallback: langdetect time-based % (trainer non-English seconds / trainer talk time).
    tr_pct = cov.get("trainer_non_english_pct")
    tr_lang = cov.get("trainer_non_english_language")
    src = "transcript analysis"
    if not isinstance(tr_pct, (int, float)):
        by_role = (lang.get("non_english_sec_by_role") or {}) if isinstance(lang, dict) else {}
        tr_ne = by_role.get("trainer", 0) or 0
        tr_sec = ((transcript.get("metrics", {}) or {}).get("talk_time", {}) or {}).get("trainer_sec") or 0
        tr_pct = (100.0 * tr_ne / tr_sec) if tr_sec else None
        src = "language detection over trainer talk time"
    if isinstance(tr_pct, (int, float)) and tr_pct > TRAINER_NE_GRACE_PCT:
        n = math.ceil((tr_pct - TRAINER_NE_GRACE_PCT) / TRAINER_NE_BRACKET_PCT)
        pts = -TRAINER_NE_BRACKET_PTS * n
        sev = "critical" if pts <= -30 else ("high" if pts <= -20 else "medium")
        lang_txt = f" ({tr_lang})" if tr_lang else ""
        deductions.append({"reason": "trainer_non_english_heavy", "points": pts, "severity": sev,
                           "evidence": f"trainer spoke ~{round(tr_pct)}% in non-English{lang_txt} ({src})"})
        flags.append({"source": "transcript", "type": "trainer_non_english_heavy",
                      "trainer_non_english_pct": round(tr_pct, 1), "language": tr_lang})

    # ── coverage: missed topics + shallow depth (integrated) ────────────────
    # Topics are matched by MEANING and order-independently in analyze_coverage — the
    # trainer need not name them or cover them in sequence. Missed topics are penalised
    # only when the session ran short OR overall coverage is very low (< 50%); if the
    # session ran full length with decent coverage, they are just flagged for review.
    rows = [c for c in (cov.get("coverage") or []) if isinstance(c, dict)]
    coverage_pct = cov.get("coverage_pct")
    missed = [c for c in rows if c.get("status") == "not_covered"]
    shallow = [c for c in rows if c.get("status") == "partial"
               or (c.get("status") == "covered" and c.get("depth_vs_allotted") == "under_covered")]

    def _topic_line(c):
        exp, spent = c.get("expected_minutes"), c.get("approx_minutes_spent")
        note = c.get("depth_note") or c.get("evidence")
        line = str(c.get("section", "?"))
        if exp is not None or spent is not None:
            line += f" (expected ~{exp}m, spent ~{spent}m)"
        if note:
            line += f" — {note}"
        return line

    if missed:
        very_less = coverage_pct is not None and coverage_pct < COVERAGE_VERY_LESS_PCT
        if session_ran_short or very_less:
            why = "session ran short" if session_ran_short else f"only {coverage_pct}% of topics covered"
            deductions.append({
                "reason": "topics_missed", "severity": "high",
                "points": MISSED_TOPIC_PTS * len(missed),
                "evidence": f"{len(missed)} planned topic(s) not covered ({why}): "
                            + "; ".join(_topic_line(c) for c in missed[:8])})
        else:
            deductions.append({
                "reason": "topics_left_flag", "points": 0, "severity": "info",
                "evidence": f"session ran full length ({actual_min} of {planned_min} min) but "
                            f"{len(missed)} topic(s) were left uncovered: "
                            + ", ".join(str(c.get('section', '?')) for c in missed[:8])
                            + " — review whether acceptable"})
        for c in missed:
            flags.append({"source": "transcript", "type": "topic_not_covered",
                          "section": c.get("section"), "expected_minutes": c.get("expected_minutes"),
                          "note": c.get("depth_note")})

    if shallow:
        deductions.append({
            "reason": "topics_shallow", "severity": "medium",
            "points": SHALLOW_TOPIC_PTS * len(shallow),
            "evidence": f"{len(shallow)} topic(s) covered but below the expected depth: "
                        + "; ".join(_topic_line(c) for c in shallow[:8])})
        for c in shallow:
            flags.append({"source": "transcript", "type": "topic_shallow",
                          "section": c.get("section"),
                          "expected_minutes": c.get("expected_minutes"),
                          "approx_minutes_spent": c.get("approx_minutes_spent"),
                          "note": c.get("depth_note")})

    # ── video flags ─────────────────────────────────────────────────────────
    if video.get("enabled"):
        cam = video.get("camera", {}).get("candidate", {})
        pct = cam.get("on_camera_pct_of_speaking")
        if pct is not None and pct < CAMERA_OFF_THRESHOLD and cam.get("speaking_sec", 0) > 30:
            pts, sev = WEIGHTS["candidate_camera_off"]
            deductions.append({"reason": "candidate_camera_off", "points": pts, "severity": sev,
                               "evidence": f"candidate on camera {round(pct*100)}% of speaking time"})
            flags.append({"source": "video", "type": "candidate_camera_off",
                          "on_camera_pct_of_speaking": pct})

        vis = video.get("vision", {})
        for g in vis.get("gaze", []) or []:
            if isinstance(g, dict) and (g.get("reading_likelihood") or 0) >= GAZE_THRESHOLD:
                pts, sev = WEIGHTS["gaze_fixed_offscreen"]
                deductions.append({"reason": "gaze_fixed_offscreen", "points": pts, "severity": sev,
                                   "evidence": f"gaze fixed off-screen while answering "
                                               f"(reading_likelihood {g.get('reading_likelihood')}, "
                                               f"stability {g.get('stability')})",
                                   "approx_time": g.get("answer_time"),
                                   "evidence_frames": g.get("evidence_frames")})
                flags.append({"source": "video", "type": "gaze_fixed_offscreen", **g})

        # ── trainer camera off while speaking (screen-only time excluded) ────
        tr_cam = (video.get("camera_state") or {}).get("trainer") or {}
        longest_off = tr_cam.get("longest_camera_off_sec") or 0
        if longest_off > TRAINER_CAM_OFF_GRACE_SEC:
            pts = TRAINER_CAM_OFF_PTS_1 if longest_off <= TRAINER_CAM_OFF_BAND_SEC else TRAINER_CAM_OFF_PTS_2
            at = tr_cam.get("longest_camera_off_start")
            deductions.append({
                "reason": "trainer_camera_off", "points": pts,
                "severity": "medium" if pts == TRAINER_CAM_OFF_PTS_1 else "high",
                "evidence": (f"trainer camera off ~{round(longest_off / 60)} min continuously while speaking"
                             + (f" (starting ~{at})" if at else "")
                             + "; screen-share-only moments were not counted"),
                "approx_time": at})
            flags.append({"source": "video", "type": "trainer_camera_off",
                          "longest_camera_off_sec": longest_off,
                          "camera_off_sec": tr_cam.get("camera_off_sec"),
                          "unassessable_sec": tr_cam.get("unassessable_sec")})

        # ── trainer reading from screen while explaining (not a shared doc) ───
        for r in vis.get("trainer_reading", []) or []:
            if not isinstance(r, dict):
                continue
            if (r.get("reading_likelihood") or 0) >= TRAINER_READING_THRESHOLD and not r.get("document_shared"):
                deductions.append({
                    "reason": "trainer_reading_screen", "points": TRAINER_READING_PTS, "severity": "medium",
                    "evidence": (f"trainer appears to read from screen while explaining "
                                 f"(reading_likelihood {r.get('reading_likelihood')}, "
                                 f"stability {r.get('stability')})"),
                    "approx_time": r.get("answer_time"),
                    "evidence_frames": r.get("evidence_frames")})
                flags.append({"source": "video", "type": "trainer_reading_screen", **r})

        cons = vis.get("consistency")
        if isinstance(cons, dict):
            _apply_person_change(cons, cfg, deductions, flags)

    # ── assemble scores ─────────────────────────────────────────────────────
    # de-duplicate the same reason (worst single occurrence counts once)
    best = {}
    for d in deductions:
        r = d["reason"]
        if r not in best or d["points"] < best[r]["points"]:
            best[r] = d

    # proxy coaching is graduated by how often it came up: a single passing
    # mention costs PROXY_SINGLE; repeated discussion costs the full weight
    if "proxy_interview_coaching" in best:
        n = sum(1 for f in cov.get("integrity_flags", []) or []
                if isinstance(f, dict) and f.get("type") == "proxy_interview_coaching")
        d = best["proxy_interview_coaching"]
        d["occurrences"] = max(1, n)
        if n <= 1:
            d["points"], d["severity"] = PROXY_SINGLE
            d["note"] = "proxy mentioned once in the session"
        else:
            d["note"] = f"proxy support discussed {n} times"

    applied = list(best.values())

    # SCORING GATE — only reasons in SCORED_FLAGS reduce the score; every other detected
    # issue is kept as a 0-point review flag (still visible to a human). With SCORED_FLAGS
    # empty this pins the integrity score at 100.
    for d in applied:
        if d["reason"] not in SCORED_FLAGS:
            d["points"] = 0
            d["scoring"] = "flag_only"

    integrity = max(0, 100 + sum(d["points"] for d in applied))
    coverage = cov.get("coverage_pct")
    applied = sorted(applied, key=lambda d: d["points"])

    # lift the meeting description to the top level — it's the "what happened"
    # record and should be easy to find next to summary
    description = transcript.pop("meeting_description", None) or {"available": False}

    report_metrics = _report_metrics(transcript, description, rows, coverage)

    return {
        "meeting": meta,
        "summary": _summary(meta, cov, coverage, integrity, applied,
                            planned_min, actual_min),
        "meeting_description": description,
        "report_metrics": report_metrics,
        "scoring": {
            "trainer_coverage_score": coverage,
            "session_integrity_score": integrity,
            "tier": _tier(integrity),
            "deductions": applied,
        },
        "flags": flags,
        "transcript": transcript,
        "video": video,
    }


def _summary(meta, cov, coverage, integrity, applied,
             planned_min, actual_min) -> dict:
    """Plain-language read of the whole result, for humans skimming result.json."""
    rows = [c for c in (cov.get("coverage") or []) if isinstance(c, dict)]
    total_planned = cov.get("expected_section_count") or len(rows)
    n_cov = sum(1 for c in rows if c.get("status") == "covered")
    n_par = sum(1 for c in rows if c.get("status") == "partial")
    n_miss = sum(1 for c in rows if c.get("status") == "not_covered")

    red_flags = []
    for d in applied:
        line = f"{d['reason']} ({d['points']} pts): {d.get('evidence')}"
        if d.get("approx_time"):
            line += f" — at ~{d['approx_time']}"
        red_flags.append(line)

    return {
        "session": f"Day {meta.get('day')} — {meta.get('day_title')} "
                   f"(label: {meta.get('day_step_name')})",
        "duration": (f"{actual_min} of {planned_min} planned minutes "
                     f"({round(100 * actual_min / planned_min)}%)" if planned_min and actual_min
                     else f"{actual_min} min"),
        "trainer_coverage": f"{coverage}% — {n_cov} covered, {n_par} partial, "
                            f"{n_miss} not covered of {total_planned} planned sections",
        "session_integrity": f"{integrity}/100 ({_tier(integrity)})",
        "red_flags": red_flags or ["none"],
        "proof_folder": "proof/ (one folder per red flag, with frames and quotes)",
    }


def _report_metrics(transcript: dict, description: dict, rows: list, coverage) -> dict:
    """Context-only numbers for the report. These NEVER change the scores.

      - talk-time split + candidate-vs-trainer ratio
      - trainer fluency 1-10 (from the GPT meeting description)
      - coverage-vs-expected ratio (minutes actually spent / minutes the plan allots)
    """
    tt = ((transcript.get("metrics") or {}).get("talk_time") or {})
    tr_sec = tt.get("trainer_sec") or 0
    ca_sec = tt.get("candidate_sec") or 0
    both = tr_sec + ca_sec
    exp_tot = sum((c.get("expected_minutes") or 0) for c in rows)
    spent_tot = sum((c.get("approx_minutes_spent") or 0) for c in rows)
    fluency = description.get("trainer_fluency") if isinstance(description, dict) else None
    return {
        "trainer_talk_pct": round(100 * tr_sec / both) if both else None,
        "candidate_talk_pct": round(100 * ca_sec / both) if both else None,
        "talk_time_candidate_vs_trainer": round(ca_sec / tr_sec, 2) if tr_sec else None,
        "trainer_fluency_1_10": fluency,
        "trainer_fluency_note": (description.get("trainer_fluency_note")
                                 if isinstance(description, dict) else None),
        "coverage_vs_expected_ratio": round(spent_tot / exp_tot, 2) if exp_tot else None,
        "coverage_pct": coverage,
        "note": "Report-only metrics — they do NOT affect the integrity score.",
    }


def _tier(score: int) -> str:
    if score >= 80:
        return "Clean"
    if score >= 50:
        return "Review"
    return "High-risk"


def _fmt_mmss(sec) -> str:
    s = int(sec or 0)
    return f"{s // 60}:{s % 60:02d}"


def _apply_person_change(cons: dict, cfg, deductions: list, flags: list) -> None:
    """Duration-gated proxy-swap policy.

    A second on-screen identity only costs points if it is present LONGER than the
    grace window. Session type (from Salesforce) changes what counts as suspicious:

      one_on_one : any secondary identity over the grace window       -> -40 (critical)
      group      : a face present from the START is an expected participant
                   (no deduction); a face that FIRST appears mid-session and
                   stays over the grace window                         -> -25 (high)

    Anything at/under the grace window is a 0-point note so a reviewer still sees
    it, but no marks are lost. Falls back to the old boolean behaviour when the
    detector did not report per-identity presence (e.g. no trainer reference photo).
    """
    grace = getattr(cfg, "person_change_grace_sec", 120) or 120
    start_window = getattr(cfg, "session_start_window_sec", 300) or 300
    session_type = (getattr(cfg, "session_type", "one_on_one") or "one_on_one").lower()

    presences = cons.get("identity_presence")
    if not presences:
        # no per-identity presence available -> original all-or-nothing behaviour
        if cons.get("same_person_throughout") is False:
            pts, sev = WEIGHTS["person_change_midsession"]
            deductions.append({"reason": "person_change_midsession", "points": pts, "severity": sev,
                               "evidence": cons.get("note"), "evidence_frames": cons.get("evidence_frames")})
            flags.append({"source": "video", "type": "person_change_midsession", **cons})
        return

    frames = cons.get("evidence_frames")
    secondaries = [p for p in presences if not p.get("primary")]
    for p in secondaries:
        pres = p.get("presence_sec", 0) or 0
        first = p.get("first_time_sec", 0) or 0
        ts = _fmt_mmss(first)
        base = {"source": "video", "label": p.get("label"),
                "presence_sec": pres, "first_time_sec": first}

        if pres <= grace:                       # brief appearance -> note only, no marks
            deductions.append({
                "reason": "person_change_brief", "points": 0, "severity": "info",
                "evidence": f"a different face appeared briefly (~{round(pres)}s on screen, first at {ts}); "
                            f"under the {round(grace / 60)}-min review threshold — no marks deducted",
                "evidence_frames": frames})
            flags.append({**base, "type": "person_change_brief"})
            continue

        if session_type == "group":
            if first <= start_window:           # present from the start -> expected participant
                flags.append({**base, "type": "expected_participant"})
                continue
            pts, sev = WEIGHTS["person_appeared_midsession"]
            reason = "person_appeared_midsession"
            evidence = (f"an additional person (not on camera at the start) appeared at {ts} and stayed "
                        f"~{round(pres / 60)} min — unexpected for this group session")
        else:
            pts, sev = WEIGHTS["person_change_midsession"]
            reason = "person_change_midsession"
            evidence = (f"a different on-screen person was present ~{round(pres / 60)} min (first at {ts}) — "
                        f"a 1:1 session should have a single candidate")

        deductions.append({"reason": reason, "points": pts, "severity": sev,
                           "evidence": evidence, "approx_time": ts, "evidence_frames": frames})
        flags.append({**base, "type": reason, "note": cons.get("note"),
                      "candidate_identities": cons.get("candidate_identities")})
