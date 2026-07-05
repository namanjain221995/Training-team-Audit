"""Scoring — turns raw signals into two numbers, each with its receipts.

  trainer_coverage_score   : % of the day's planned sections actually covered
  session_integrity_score  : 100 = clean; deductions per detected red flag,
                             each carrying the evidence that triggered it.

The score answers "should a human review this session", NOT "how good is the
candidate". Weights are a starting point and meant to be tuned.
"""

# reason -> (points, severity). Deductions are applied when the flag is present.
WEIGHTS = {
    "proxy_interview_coaching": (-10, "high"),    # repeated proxy coaching; single mention -> PROXY_SINGLE
    "fabricated_experience":    (-30, "critical"),
    "scripted_deception":       (-20, "high"),
    "person_change_midsession": (-40, "critical"),
    "candidate_camera_off":     (-20, "high"),
    "gaze_fixed_offscreen":     (-15, "medium"),
    "non_english_heavy":        (-5,  "low"),
    "key_section_rushed":       (-10, "medium"),
    "session_too_short":        (-15, "high"),   # escalates to -25/critical below SESSION_SEVERE_RATIO
}

CAMERA_OFF_THRESHOLD = 0.5   # candidate on-camera < 50% of their speaking time
GAZE_THRESHOLD       = 0.6   # reading_likelihood >= 0.6 counts as a flag
NON_ENGLISH_THRESHOLD = 120  # seconds of non-English before it's flagged
SESSION_SHORT_RATIO  = 0.5   # ran < 50% of the day plan's total minutes
SESSION_SEVERE_RATIO = 0.25  # ran < 25% -> harsher deduction
PROXY_SINGLE = (-5, "medium")  # proxy mentioned exactly once in the whole session
# NOTE (multi-recording): if a host stop/restarts, one logical session lands in
# several chunks and EACH chunk looks "short". The merge step (§9 CLAUDE.md)
# must recompute this from summed chunk durations — per-chunk raw numbers are
# stored in flags for exactly that reason.


def assemble(meta: dict, transcript: dict, video: dict, presentation: dict | None, cfg) -> dict:
    deductions = []
    flags = []

    # ── session ran far shorter than the day plan ───────────────────────────
    from .config import day_sections
    planned_min = (day_sections(cfg) or {}).get("total_minutes") or 0
    dur_sec = meta.get("duration_sec") or 0
    actual_min = round(dur_sec / 60.0, 1)
    if planned_min:
        # ratio from raw seconds — pre-rounding actual_min can flip the boundary,
        # and a 0-duration (empty/broken) session must still flag, not escape
        ratio = (dur_sec / 60.0) / planned_min
        if ratio < SESSION_SHORT_RATIO:
            severe = ratio < SESSION_SEVERE_RATIO
            pts, sev = (-25, "critical") if severe else WEIGHTS["session_too_short"]
            deductions.append({
                "reason": "session_too_short", "points": pts, "severity": sev,
                "evidence": f"session ran {actual_min} of {planned_min} planned minutes "
                            f"({round(ratio * 100)}% of the day plan)",
            })
            flags.append({"source": "transcript", "type": "session_too_short",
                          "planned_minutes": planned_min, "actual_minutes": actual_min,
                          "ratio": round(ratio, 3)})

    # ── transcript integrity flags (from GPT-4o) ────────────────────────────
    cov = transcript.get("coverage_analysis", {})
    for f in cov.get("integrity_flags", []) or []:
        if not isinstance(f, dict):   # shape-drifted model output: keep it visible, don't crash
            flags.append({"source": "transcript", "type": "other", "raw": f})
            continue
        ftype = f.get("type", "other")
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

    # ── rushed / skipped high-weight sections ───────────────────────────────
    rushed = cov.get("under_covered_key_sections", []) or []
    if rushed:
        pts, sev = WEIGHTS["key_section_rushed"]
        deductions.append({"reason": "key_section_rushed", "points": pts, "severity": sev,
                           "evidence": "under-covered vs allotted time: " + ", ".join(rushed[:4])})
        flags.append({"source": "transcript", "type": "key_section_rushed", "sections": rushed})

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

        cons = vis.get("consistency")
        if isinstance(cons, dict) and cons.get("same_person_throughout") is False:
            pts, sev = WEIGHTS["person_change_midsession"]
            deductions.append({"reason": "person_change_midsession", "points": pts, "severity": sev,
                               "evidence": cons.get("note"), "evidence_frames": cons.get("evidence_frames")})
            flags.append({"source": "video", "type": "person_change_midsession", **cons})

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

    integrity = max(0, 100 + sum(d["points"] for d in applied))
    coverage = cov.get("coverage_pct")
    applied = sorted(applied, key=lambda d: d["points"])

    # lift the meeting description to the top level — it's the "what happened"
    # record and should be easy to find next to summary
    description = transcript.pop("meeting_description", None) or {"available": False}

    return {
        "meeting": meta,
        "summary": _summary(meta, cov, coverage, integrity, applied,
                            presentation, planned_min, actual_min),
        "meeting_description": description,
        "scoring": {
            "trainer_coverage_score": coverage,
            "session_integrity_score": integrity,
            "tier": _tier(integrity),
            "deductions": applied,
        },
        "flags": flags,
        "candidate_presentation": presentation or {"enabled": False},
        "transcript": transcript,
        "video": video,
    }


def _summary(meta, cov, coverage, integrity, applied, presentation,
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

    pres_line = "not analyzed"
    if isinstance(presentation, dict) and presentation.get("enabled"):
        if presentation.get("person_visible") is False:
            pres_line = presentation.get("note") or "candidate not clearly visible on camera"
        else:
            pres_line = (presentation.get("overall_notes") or presentation.get("note")
                         or presentation.get("error") or "see candidate_presentation block")

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
        "candidate_presentation": pres_line,
        "proof_folder": "proof/ (one folder per red flag, with frames and quotes)",
    }


def _tier(score: int) -> str:
    if score >= 80:
        return "Clean"
    if score >= 50:
        return "Review"
    return "High-risk"
