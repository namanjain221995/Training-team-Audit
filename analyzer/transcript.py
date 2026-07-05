"""Transcript track.

1. load_transcript  -> speaker-tagged, timestamped segments (from VTT, or Whisper)
2. compute_metrics  -> duration, per-role talk-time (seconds), non-English flags  [no AI]
3. analyze_coverage -> GPT-4o: day-plan coverage + integrity read                 [1 AI call]
"""
import os
import re
from difflib import SequenceMatcher


# ── 1. Load ────────────────────────────────────────────────────────────────
def load_transcript(cfg) -> tuple[list[dict], str]:
    """Return (segments, source). segment = {speaker, start, end, text}."""
    if cfg.transcript_path and os.path.exists(cfg.transcript_path):
        return parse_vtt(cfg.transcript_path), "vtt"
    print(f"No transcript file; transcribing video with Whisper ({cfg.whisper_model})...")
    return transcribe_whisper(cfg.video_path, cfg.whisper_model), "whisper"


def parse_vtt(path: str) -> list[dict]:
    import webvtt
    segments = []
    for cap in webvtt.read(path):
        raw = cap.text.replace("\n", " ").strip()
        speaker, text = None, raw
        m = re.match(r"^\s*([^:]{1,60}?):\s*(.*)$", raw)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
        segments.append({
            "speaker": speaker,
            "start": _ts_to_sec(cap.start),
            "end":   _ts_to_sec(cap.end),
            "text":  text,
        })
    print(f"Parsed {len(segments)} VTT cues from {path}")
    return segments


def transcribe_whisper(video_path: str, model_name: str) -> list[dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    seg_iter, info = model.transcribe(video_path, vad_filter=True)
    segments = [{"speaker": None, "start": float(s.start), "end": float(s.end),
                 "text": s.text.strip()} for s in seg_iter]
    print(f"Whisper produced {len(segments)} segments (detected language: {info.language})")
    return segments


# ── 2. Metrics (no AI) ──────────────────────────────────────────────────────
def compute_metrics(segments: list[dict], cfg) -> dict:
    if not segments:
        return {"duration_sec": 0, "talk_time": {}, "language": {}, "speaker_roles": {}}

    duration = max(s["end"] for s in segments) - min(s["start"] for s in segments)
    roles = _assign_roles(segments, cfg)

    talk = {"trainer_sec": 0.0, "candidate_sec": 0.0, "other_sec": 0.0}
    for s in segments:
        role = roles.get(s["speaker"], "other")
        talk[f"{role}_sec"] = talk.get(f"{role}_sec", 0.0) + max(0.0, s["end"] - s["start"])

    spoken = talk["trainer_sec"] + talk["candidate_sec"] + talk["other_sec"]
    ratios = {
        "trainer_ratio":   round(talk["trainer_sec"] / spoken, 3) if spoken else None,
        "candidate_ratio": round(talk["candidate_sec"] / spoken, 3) if spoken else None,
    }

    lang = _language_flags(segments, roles)

    return {
        "duration_sec": round(duration, 1),
        "talk_time": {**{k: round(v, 1) for k, v in talk.items()}, **ratios},
        "language": lang,
        "speaker_roles": roles,
    }


def _assign_roles(segments: list[dict], cfg) -> dict:
    """Map each distinct speaker label to 'trainer' / 'candidate' / 'other'."""
    talk_by_speaker: dict = {}
    for s in segments:
        if s["speaker"]:
            talk_by_speaker[s["speaker"]] = talk_by_speaker.get(s["speaker"], 0.0) + (s["end"] - s["start"])

    if not talk_by_speaker:  # Whisper: no speaker labels
        return {}

    roles = {}
    ranked = sorted(talk_by_speaker, key=talk_by_speaker.get, reverse=True)

    if cfg.trainer_name:
        trainer = max(ranked, key=lambda sp: _sim(sp, cfg.trainer_name))
        roles[trainer] = "trainer"
    else:
        # Heuristic: in these sessions the trainer speaks the most.
        trainer = ranked[0]
        roles[trainer] = "trainer"

    for sp in ranked:
        if sp in roles:
            continue
        if cfg.candidate_name and _sim(sp, cfg.candidate_name) > 0.6:
            roles[sp] = "candidate"
        elif "candidate" not in roles.values():
            roles[sp] = "candidate"
        else:
            roles[sp] = "other"
    return roles


def _language_flags(segments: list[dict], roles: dict) -> dict:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
    except Exception:
        return {"available": False}

    non_en_sec = 0.0
    flagged = []
    for s in segments:
        text = s["text"]
        if len(text) < 20:
            continue
        try:
            lang = detect(text)
        except Exception:
            continue
        if lang != "en":
            dur = max(0.0, s["end"] - s["start"])
            non_en_sec += dur
            if len(flagged) < 25:
                flagged.append({
                    "start": _sec_to_ts(s["start"]),
                    "speaker_role": roles.get(s["speaker"], "unknown"),
                    "lang": lang,
                    "text": text[:120],
                })
    return {"available": True, "non_english_sec": round(non_en_sec, 1), "flagged_segments": flagged}


# ── 3. Coverage + integrity (GPT-4o) ────────────────────────────────────────
COVERAGE_SYSTEM = (
    "You are a quality-control auditor reviewing an internal training session recording "
    "for a staffing company. You are given the session's planned sections for the day and "
    "the meeting transcript. Assess, factually and neutrally, whether the trainer actually "
    "covered each planned section, and flag any integrity concerns (for example: coaching "
    "the candidate to use a live proxy during real interviews, to fabricate work experience, "
    "or to give scripted/deceptive answers to employers). "
    "Return ONLY a JSON object. Keep every quote under 15 words."
)


def analyze_coverage(segments: list[dict], sections_cfg: dict, oa, session_label: str | None = None,
                     topics: list | None = None) -> dict:
    sections = sections_cfg.get("sections", [])
    transcript_text = _render_transcript(segments)

    label_line = (f"SALESFORCE SESSION LABEL: {session_label}\n"
                  "(If the planned sections below clearly do not match this label's topic, "
                  "still assess each section, but this signals the rubric may be for a different day.)\n\n"
                  if session_label else "")
    topics_block = _topics_block(topics)
    section_list = "\n".join(
        f"- {s['name']} (allotted ~{s.get('expected_minutes','?')} min): {s['detail']}"
        for s in sections
    )
    user = (
        f"DAY TITLE: {sections_cfg.get('title','')}  "
        f"(planned total ~{sections_cfg.get('total_minutes','?')} min)\n\n"
        f"{label_line}{topics_block}"
        f"PLANNED SECTIONS (with the time the plan allots each):\n{section_list}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        "For each section decide status, how its depth compares to the time it was "
        "allotted (a section allotted 45 min that got one passing sentence is 'under_covered'), "
        "and estimate approx_minutes_spent on it from the transcript timestamps (0 if not covered). "
        "Match sections by MEANING using their descriptions, not by the trainer literally naming them. "
        "Do NOT penalise ordering or exact clock position — only whether it was covered, to what "
        "depth, and roughly how long. Quote the trainer's own words as evidence wherever possible.\n"
        "Respond with JSON of exactly this shape:\n"
        "{\n"
        '  "coverage": [\n'
        '    {"section": "<name>", "status": "covered|partial|not_covered", '
        '"depth_vs_allotted": "adequate|under_covered|not_applicable", '
        '"approx_minutes_spent": <number>, '
        '"evidence": "<=15 word quote or null", "approx_time": "mm:ss or null"}\n'
        "  ],\n"
        '  "integrity_flags": [\n'
        '    {"type": "proxy_interview_coaching|fabricated_experience|scripted_deception|other", '
        '"confidence": "low|medium|high", "evidence": "<=15 word quote", "approx_time": "mm:ss or null"}\n'
        "  ]\n"
        "}"
    )
    result = oa.chat_json(COVERAGE_SYSTEM, user)

    # GPT output is untrusted: JSON mode guarantees valid JSON, not the requested
    # shape. Null lists and non-dict rows are real-world model behaviour.
    if not isinstance(result, dict):
        result = {"_invalid_response": True, "raw": str(result)[:2000]}
    cov = result.get("coverage") or []
    if not isinstance(cov, list):
        cov = []
    cov = [c for c in cov if isinstance(c, dict)]
    result["coverage"] = cov
    iflags = result.get("integrity_flags") or []
    result["integrity_flags"] = [f for f in iflags if isinstance(f, dict)] \
        if isinstance(iflags, list) else []

    # Attach each section's allotted minutes. GPT may echo names loosely
    # ('GitHub Why Ladder' vs 'GitHub "Why Ladder"'), so match fuzzily and
    # canonicalise to the plan's name so downstream exact joins keep working.
    def _match_section(name):
        if not name or not sections:
            return None
        best = max(sections, key=lambda s: _sim(s["name"], str(name)))
        return best if _sim(best["name"], str(name)) >= 0.6 else None

    for c in cov:
        planned = _match_section(c.get("section"))
        if planned:
            c["section"] = planned["name"]
            c["expected_minutes"] = planned.get("expected_minutes")

    # Cross-check against the topic analysis (a separate pass over the same
    # transcript): the two views must not contradict. Sum the minutes the topic
    # pass attributed to each planned section; if it credits >=3 minutes to a
    # section this call marked "not_covered", upgrade to partial and say why.
    if topics:
        attributed: dict = {}
        for tpc in topics:
            if not isinstance(tpc, dict) or tpc.get("on_curriculum") is not True:
                continue
            ps = str(tpc.get("plan_section") or "").split(":")[0].strip()
            mins = tpc.get("approx_minutes")
            if not ps or not isinstance(mins, (int, float)):
                continue
            m = _match_section(ps)
            if m:
                attributed[m["name"]] = attributed.get(m["name"], 0.0) + float(mins)
        for c in cov:
            tm = attributed.get(c.get("section"))
            if tm is None:
                continue
            c["topic_minutes"] = round(tm, 1)
            if tm >= 3 and c.get("status") == "not_covered":
                c["status"] = "partial"
                if c.get("depth_vs_allotted") in (None, "not_applicable"):
                    c["depth_vs_allotted"] = "under_covered"
                if not c.get("approx_minutes_spent"):
                    c["approx_minutes_spent"] = round(tm)
                c["reconciled"] = (f"upgraded to partial: the topic analysis attributes "
                                   f"~{round(tm)} min of discussion to this section")

    covered = sum(1 for c in cov if c.get("status") == "covered")
    partial = sum(1 for c in cov if c.get("status") == "partial")
    total = len(sections) or 1
    result["coverage_pct"] = round(100 * (covered + 0.5 * partial) / total)
    result["expected_section_count"] = len(sections)
    # sections the plan weights heavily (>=25 min) but that were skipped or rushed
    heavy = {s["name"] for s in sections if s.get("expected_minutes", 0) >= 25}
    result["under_covered_key_sections"] = [
        c.get("section") for c in cov
        if c.get("section") in heavy
        and (c.get("status") == "not_covered" or c.get("depth_vs_allotted") == "under_covered")
    ]
    return result


# ── 4. Full meeting description + per-topic QA ──────────────────────────────
DESCRIBE_SYSTEM = (
    "You are a quality-assurance auditor writing the factual meeting record for an "
    "internal training-session audit. You are given the transcript of one recorded Zoom "
    "training session and the day's planned curriculum. Work like a careful human QA "
    "reviewer: reconstruct what actually happened, identify every distinct topic that was "
    "discussed, when, for how long, who led it, and whether it belongs to the planned "
    "curriculum. Assess both the trainer's delivery and the candidate's participation, "
    "neutrally and concretely. Never invent details that are not in the transcript. "
    "Return ONLY a JSON object."
)


def describe_meeting(segments: list[dict], oa, session_label: str | None = None,
                     sections_cfg: dict | None = None) -> dict:
    """One GPT call -> full meeting record + per-topic QA breakdown."""
    transcript_text = _render_transcript(segments)
    sections = (sections_cfg or {}).get("sections", [])
    plan_block = ""
    if sections:
        plan_lines = "\n".join(f"- {s['name']}: {s['detail']}" for s in sections)
        plan_block = (f"PLANNED CURRICULUM FOR THIS DAY "
                      f"({(sections_cfg or {}).get('title', '')}):\n{plan_lines}\n\n")
    user = (
        (f"SESSION LABEL: {session_label}\n\n" if session_label else "")
        + plan_block
        + f"TRANSCRIPT:\n{transcript_text}\n\n"
        "Respond with JSON of exactly this shape:\n"
        "{\n"
        '  "overview": "<3-5 paragraph plain-English narrative of the meeting from start to finish>",\n'
        '  "timeline": [{"time": "mm:ss", "event": "<what happened / what the discussion moved to>"}],\n'
        '  "topics": [\n'
        '    {"topic": "<short name>", "start": "mm:ss", "end": "mm:ss", "approx_minutes": <number>,\n'
        '     "led_by": "trainer|candidate|both", "what_happened": "<1-2 sentences>",\n'
        '     "plan_section": "<planned section NAME only (no description), or null>",\n'
        '     "on_curriculum": true|false}\n'
        "  ],\n"
        '  "trainer_summary": "<2-3 sentences: QA view of what the trainer did and how>",\n'
        '  "candidate_summary": "<2-3 sentences: QA view of what the candidate did and how>",\n'
        '  "notable_quotes": [{"time": "mm:ss", "speaker": "<who>", "quote": "<=20 word quote>"}]\n'
        "}\n"
        "topics: EVERY distinct topic in the meeting, in order, covering the whole session "
        "(estimate times from the transcript timestamps). Mark on_curriculum=false for "
        "anything not in the planned curriculum (small talk, admin, or unrelated coaching). "
        "The timeline must cover the whole meeting in order with 6-15 entries. "
        "notable_quotes: up to 5 quotes that best characterise the session."
    )
    result = oa.chat_json(DESCRIBE_SYSTEM, user)

    # untrusted model output: normalise the shape (same policy as analyze_coverage)
    if not isinstance(result, dict):
        result = {"_invalid_response": True, "raw": str(result)[:2000]}
    for key in ("timeline", "topics", "notable_quotes"):
        v = result.get(key)
        result[key] = [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
    result["available"] = bool(result.get("overview"))
    return result


def _topics_block(topics: list | None) -> str:
    """Render the describe_meeting topic analysis as working notes for the
    coverage call, so both passes reach ONE consistent judgment."""
    if not topics:
        return ""
    lines = []
    for t in topics:
        if not isinstance(t, dict) or not t.get("topic"):
            continue
        span = f"{t.get('start', '?')}-{t.get('end', '?')}"
        mins = t.get("approx_minutes")
        sec = (str(t.get("plan_section")).split(":")[0].strip()
               if t.get("on_curriculum") and t.get("plan_section") else "OFF-PLAN")
        lines.append(f"- {span} (~{mins} min) {t['topic']} -> {sec}")
    if not lines:
        return ""
    return ("TOPIC ANALYSIS (a detailed prior pass over this same transcript — your "
            "per-section judgments MUST be consistent with it: if topics totalling "
            ">=3 minutes map to a planned section, that section is at least 'partial', "
            "and approx_minutes_spent should be near the summed topic minutes):\n"
            + "\n".join(lines) + "\n\n")


def _render_transcript(segments: list[dict], max_chars: int = 60000) -> str:
    lines = []
    for s in segments:
        who = s["speaker"] or "Speaker"
        lines.append(f"[{_sec_to_ts(s['start'])}] {who}: {s['text']}")
    text = "\n".join(lines)
    if len(text) > max_chars:  # keep head + tail if enormous
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2:]
        text = head + "\n...[transcript trimmed]...\n" + tail
    return text


# ── helpers ─────────────────────────────────────────────────────────────────
def _ts_to_sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


def _sec_to_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
