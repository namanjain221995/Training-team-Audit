"""HTML report — renders result.json into output/report.html for non-technical review.

Pure stdlib (html/json/os/re) so it can also run on the host without the
container deps:  python -m analyzer.report output/result.json

Design rules:
  - Plain language: every machine reason code maps to a human title + one-line
    explanation (REASON_INFO).
  - Every red flag shows its evidence inline: the quote, the timestamp, and the
    proof images copied by proof.py (paths are relative to output/, where
    report.html lives, so <img src="proof/..."> just works when double-clicked).
  - All model/transcript text is escaped — GPT output is untrusted.
"""
import html
import json
import os
import re

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

REASON_INFO = {
    "proxy_interview_coaching": ("Possible proxy-interview coaching",
                                 "The transcript suggests coaching the candidate to have someone else attend real interviews."),
    "fabricated_experience":    ("Possible fabricated experience",
                                 "The transcript suggests coaching the candidate to claim work experience they do not have."),
    "scripted_deception":       ("Possible scripted / deceptive coaching",
                                 "Wording in the transcript suggests coaching to mislead an employer."),
    "person_change_midsession": ("Different person appeared mid-session",
                                 "Face matching indicates the on-camera person changed during the session."),
    "candidate_camera_off":     ("Candidate camera off while speaking",
                                 "The candidate was off camera for most of the time they were talking."),
    "gaze_fixed_offscreen":     ("Candidate may be reading answers",
                                 "The candidate's gaze stayed fixed at an off-screen point while answering."),
    "session_too_short":        ("Session much shorter than planned",
                                 "The recording is far shorter than the training day plan expects."),
    "key_section_rushed":       ("Key topic skipped or rushed",
                                 "A topic the plan gives a lot of time to was skipped or barely touched."),
    "non_english_heavy":        ("Large amount of non-English speech",
                                 "A significant portion of the session was not in English."),
}

STATUS_INFO = {
    "covered":     ("✓ Covered", "ok"),
    "partial":     ("◐ Partly covered", "warn"),
    "not_covered": ("✗ Not covered", "bad"),
}


TIER_STYLE = {  # tier -> (css class, banner subtitle)
    "Clean":     ("ok",   "No serious concerns found — routine spot-check only."),
    "Review":    ("warn", "Some concerns found — a human should review the items below."),
    "High-risk": ("bad",  "Serious concerns found — this session needs human review."),
}

CSS = """
* { box-sizing: border-box; margin: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
       background: #eef1f5; color: #1e293b; line-height: 1.5; }
.page { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
.banner { border-radius: 14px; padding: 26px 28px; color: #fff; margin-bottom: 18px; }
.banner .big-tier { font-size: 30px; font-weight: 800; letter-spacing: .5px; }
.banner .sub { opacity: .95; margin-top: 4px; font-size: 15px; }
.banner.ok   { background: linear-gradient(120deg,#15803d,#22c55e); }
.banner.warn { background: linear-gradient(120deg,#b45309,#f59e0b); }
.banner.bad  { background: linear-gradient(120deg,#b91c1c,#ef4444); }
.scores { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }
.score-card { flex: 1 1 180px; background: #fff; border-radius: 14px; padding: 18px 20px;
              box-shadow: 0 1px 4px rgba(15,23,42,.08); }
.score-card .num { font-size: 34px; font-weight: 800; }
.score-card .num small { font-size: 16px; font-weight: 600; color: #64748b; }
.score-card .lbl { font-weight: 600; margin-top: 2px; }
.score-card .hint { font-size: 12.5px; color: #64748b; margin-top: 4px; }
.num.ok { color: #16a34a; } .num.warn { color: #d97706; } .num.bad { color: #dc2626; }
.card { background: #fff; border-radius: 14px; padding: 20px 22px; margin-bottom: 18px;
        box-shadow: 0 1px 4px rgba(15,23,42,.08); }
.card h2 { font-size: 19px; margin-bottom: 4px; }
.card .lead { color: #64748b; font-size: 13.5px; margin-bottom: 14px; }
.kv { width: 100%; border-collapse: collapse; font-size: 14.5px; }
.kv td { padding: 6px 8px 6px 0; vertical-align: top; }
.kv td:first-child { color: #64748b; white-space: nowrap; width: 180px; }
.flag { border: 1px solid #e2e8f0; border-left-width: 6px; border-radius: 10px;
        padding: 14px 16px; margin-bottom: 12px; }
.flag.critical, .flag.high { border-left-color: #dc2626; }
.flag.medium { border-left-color: #d97706; }
.flag.low { border-left-color: #64748b; }
.flag .head { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.flag .title { font-weight: 700; font-size: 15.5px; }
.flag .pts { font-weight: 800; color: #dc2626; white-space: nowrap; }
.flag .why { color: #475569; font-size: 13.5px; margin: 3px 0 8px; }
.quote { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
         padding: 9px 12px; font-size: 14px; margin-bottom: 8px; }
.quote .t { color: #64748b; font-size: 12.5px; }
.thumbs { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.thumbs a { display: block; }
.thumbs img { height: 110px; border-radius: 8px; border: 1px solid #e2e8f0; display: block; }
.thumbs .cap { font-size: 11.5px; color: #64748b; text-align: center; margin-top: 2px; }
.chip { display: inline-block; font-size: 13px; font-weight: 700; }
.chip.ok   { color: #15803d; }
.chip.warn { color: #b45309; }
.chip.bad  { color: #b91c1c; }
.chip.na   { color: #64748b; }
table.cov { width: 100%; border-collapse: collapse; font-size: 14px; }
table.cov th { text-align: left; color: #64748b; font-size: 12.5px; text-transform: uppercase;
               letter-spacing: .4px; padding: 6px 8px; border-bottom: 1px solid #e2e8f0; }
table.cov td { padding: 9px 8px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
.bar { background: #e2e8f0; border-radius: 999px; height: 8px; width: 130px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 999px; background: #16a34a; }
.bar.warn > div { background: #d97706; } .bar.bad > div { background: #dc2626; }
.minlbl { font-size: 12px; color: #64748b; white-space: nowrap; }
.split { display: flex; border-radius: 999px; overflow: hidden; height: 14px; margin: 6px 0 4px; }
.split .a { background: #3b82f6; } .split .b { background: #a855f7; }
.legend { font-size: 12.5px; color: #64748b; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
.note { background: #f0f9ff; border: 1px solid #bae6fd; color: #0c4a6e; border-radius: 8px;
        padding: 8px 12px; font-size: 13px; margin-bottom: 12px; }
footer { color: #94a3b8; font-size: 12.5px; text-align: center; margin-top: 8px; }
@media print { body { background: #fff; } .card, .score-card { box-shadow: none; border: 1px solid #e2e8f0; } }
"""


# ── public API ───────────────────────────────────────────────────────────────
def write(result: dict, out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(render(result))
    return out_path


def render(result: dict) -> str:
    meeting = result.get("meeting") or {}
    scoring = result.get("scoring") or {}
    tier = scoring.get("tier") or "Review"
    cls, tier_sub = TIER_STYLE.get(tier, TIER_STYLE["Review"])

    parts = [
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Training Session Report — Day {_e(meeting.get('day'))}</title>",
        f"<style>{CSS}</style></head><body><div class='page'>",
        _banner(tier, cls, tier_sub, meeting),
        _score_cards(result),
        _session_card(meeting, result),
        _description_card(result),
        _flags_card(result),
        _coverage_card(result),
        _video_card(result),
        f"<footer>Generated from result.json · analyzed {_e(meeting.get('analyzed_at', ''))[:19].replace('T', ' ')} UTC"
        f" · full evidence in the <b>proof/</b> folder</footer>",
        "</div></body></html>",
    ]
    return "\n".join(parts)


# ── sections ────────────────────────────────────────────────────────────────
def _banner(tier, cls, sub, meeting):
    return (f"<header class='banner {cls}'>"
            f"<div class='big-tier'>{_e(tier).upper()}</div>"
            f"<div class='sub'>{_e(sub)}</div>"
            f"<div class='sub'>Day {_e(meeting.get('day'))} training session — "
            f"trainer <b>{_e(meeting.get('trainer_name'))}</b>, "
            f"candidate <b>{_e(meeting.get('candidate_name'))}</b></div></header>")


def _score_cards(result):
    scoring = result.get("scoring") or {}
    meeting = result.get("meeting") or {}
    integ = scoring.get("session_integrity_score")
    cov = scoring.get("trainer_coverage_score")
    i_cls = "ok" if (integ or 0) >= 80 else ("warn" if (integ or 0) >= 50 else "bad")
    c_cls = "ok" if (cov or 0) >= 70 else ("warn" if (cov or 0) >= 40 else "bad")

    planned = _planned_minutes(result)
    actual = round((meeting.get("duration_sec") or 0) / 60.0, 1)
    if planned:
        pct = round(100 * actual / planned)
        d_cls = "ok" if pct >= 50 else "bad"
        dur_num = f"<span class='num {d_cls}'>{pct}%</span>"
        dur_hint = f"{_e(actual)} of {_e(planned)} planned minutes"
    else:
        dur_num = f"<span class='num'>{_e(actual)}<small> min</small></span>"
        dur_hint = "no planned total found"

    return (
        "<section class='scores'>"
        f"<div class='score-card'><div class='num {i_cls}'>{_e(integ)}<small>/100</small></div>"
        f"<div class='lbl'>Session integrity</div>"
        f"<div class='hint'>Starts at 100 — each red flag below removes points</div></div>"
        f"<div class='score-card'><div class='num {c_cls}'>{_e(cov)}%</div>"
        f"<div class='lbl'>Day-plan coverage</div>"
        f"<div class='hint'>How much of the planned curriculum the trainer taught</div></div>"
        f"<div class='score-card'><div class='num'>{dur_num}</div>"
        f"<div class='lbl'>Session length</div><div class='hint'>{dur_hint}</div></div>"
        "</section>")


def _session_card(meeting, result):
    src = meeting.get("config_source")
    rows = [
        ("Training day", f"Day {_e(meeting.get('day'))} — {_e(meeting.get('day_title'))}"),
        ("Session label (Salesforce)", _e(meeting.get("day_step_name"))),
        ("Trainer", _e(meeting.get("trainer_name"))),
        ("Candidate", _e(meeting.get("candidate_name"))),
        ("Video file", _e(meeting.get("video_file"))),
        ("Meeting ID", _e(meeting.get("meeting_id"))),
        ("Recording length", f"{_e(round((meeting.get('duration_sec') or 0) / 60.0, 1))} minutes"),
        ("Session details from", _e(src)),
    ]
    trs = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows if v and v != "None")
    warn = ""
    title = (meeting.get("day_title") or "").lower()
    label = (meeting.get("day_step_name") or "").lower()
    if title and label and not (set(title.split()) & set(label.split()) - {"day", "and", "the", "-"}):
        warn = ("<div class='note'>⚠ The Salesforce session label and the Day plan title look like "
                "different topics — the coverage results below judge the session against the Day plan.</div>")
    return f"<section class='card'><h2>Session details</h2><div class='lead'></div>{warn}<table class='kv'>{trs}</table></section>"


def _description_card(result):
    d = result.get("meeting_description") or {}
    if not isinstance(d, dict) or not d.get("overview"):
        return ""
    paras = "".join(f"<p style='margin-bottom:10px'>{_e(p.strip())}</p>"
                    for p in str(d["overview"]).split("\n") if p.strip())

    who = ""
    if d.get("trainer_summary") or d.get("candidate_summary"):
        rows = ""
        if d.get("trainer_summary"):
            rows += f"<tr><td>Trainer</td><td>{_e(d['trainer_summary'])}</td></tr>"
        if d.get("candidate_summary"):
            rows += f"<tr><td>Candidate</td><td>{_e(d['candidate_summary'])}</td></tr>"
        who = f"<table class='kv' style='margin-top:10px'>{rows}</table>"

    topics_html = ""
    topic_rows = [t for t in (d.get("topics") or []) if isinstance(t, dict) and t.get("topic")]
    if topic_rows:
        trs = ""
        for t in topic_rows:
            span = f"{_e(t.get('start'))}–{_e(t.get('end'))}" if t.get("start") else ""
            mins = t.get("approx_minutes")
            mins_s = f"~{_e(round(mins))} min" if isinstance(mins, (int, float)) else ""
            oc = t.get("on_curriculum")
            plan = t.get("plan_section")
            if oc is True and plan:
                # keep only the section NAME (models sometimes echo the description too)
                plan_name = str(plan).split(":")[0].strip()
                chip = f"<span class='chip ok'>✓ {_e(plan_name)}</span>"
            elif oc is False:
                chip = "<span class='chip warn'>⚠ off-plan</span>"
            else:
                chip = "<span class='chip na'>—</span>"
            led = _e(t.get("led_by") or "")
            trs += (f"<tr><td><b>{_e(t['topic'])}</b><div class='minlbl'>{span} {mins_s}"
                    f"{(' · led by ' + led) if led else ''}</div></td>"
                    f"<td>{chip}</td>"
                    f"<td style='color:#475569;font-size:13.5px'>{_e(t.get('what_happened'))}</td></tr>")
        topics_html = (f"<div style='margin-top:14px'><b style='font-size:14px'>Every topic, checked "
                       f"against the day plan</b>"
                       f"<table class='cov' style='margin-top:5px'><tr><th>Topic</th><th>Curriculum?</th>"
                       f"<th>What happened</th></tr>{trs}</table></div>")
    else:  # older results: plain topic strings
        chips = "".join(f"<span class='chip na' style='margin:2px 4px 2px 0'>{_e(t)}</span>"
                        for t in (d.get("topics_discussed") or []) if t)
        if chips:
            topics_html = (f"<div style='margin-top:10px'><b style='font-size:14px'>Topics discussed</b>"
                           f"<div style='margin-top:5px'>{chips}</div></div>")

    tl_rows = "".join(
        f"<tr><td style='white-space:nowrap;color:#64748b;width:70px'>{_e(t.get('time'))}</td>"
        f"<td>{_e(t.get('event'))}</td></tr>"
        for t in (d.get("timeline") or []) if t.get("event"))
    timeline = (f"<div style='margin-top:14px'><b style='font-size:14px'>Minute-by-minute</b>"
                f"<table class='cov' style='margin-top:5px'>{tl_rows}</table></div>") if tl_rows else ""

    quotes = "".join(
        f"<div class='quote'>“{_e(q.get('quote'))}”"
        f"<span class='t'> — {_e(q.get('speaker'))} at {_e(q.get('time'))}</span></div>"
        for q in (d.get("notable_quotes") or []) if q.get("quote"))
    quotes_html = (f"<div style='margin-top:14px'><b style='font-size:14px'>Notable quotes</b>"
                   f"<div style='margin-top:6px'>{quotes}</div></div>") if quotes else ""

    return ("<section class='card'><h2>📝 What happened in this meeting</h2>"
            "<div class='lead'>A factual description of the whole session, written from the transcript.</div>"
            f"{paras}{who}{topics_html}{timeline}{quotes_html}</section>")


def _flags_card(result):
    deds = (result.get("scoring") or {}).get("deductions") or []
    body = ""
    if not deds:
        body = "<div class='note'>✅ No red flags — nothing reduced the integrity score.</div>"
    for d in deds:
        if not isinstance(d, dict):
            continue
        reason = d.get("reason", "other")
        title, why = REASON_INFO.get(reason, (reason.replace("_", " ").capitalize(), ""))
        sev = _e(d.get("severity", "medium"))
        quote = ""
        if d.get("evidence"):
            at = f"<span class='t'> — at ~{_e(d['approx_time'])} into the recording</span>" if d.get("approx_time") else ""
            quote = f"<div class='quote'>“{_e(d['evidence'])}”{at}</div>"
        body += (f"<div class='flag {sev}'><div class='head'>"
                 f"<div class='title'>{_e(title)}</div>"
                 f"<div class='pts'>−{abs(d.get('points') or 0)} points</div></div>"
                 f"<div class='why'>{_e(why)}</div>{quote}"
                 f"{_thumbs(d.get('proof') or [])}</div>")
    return ("<section class='card'><h2>🚩 Red flags (what cost points)</h2>"
            "<div class='lead'>Each item removed points from the integrity score. "
            "The quote and photos are the exact evidence — click a photo to open it full size.</div>"
            f"{body}</section>")


def _coverage_card(result):
    cov = (result.get("transcript") or {}).get("coverage_analysis") or {}
    rows = [c for c in (cov.get("coverage") or []) if isinstance(c, dict)]
    if not rows:
        return ("<section class='card'><h2>📋 Curriculum coverage</h2>"
                "<div class='note'>No coverage analysis available for this run.</div></section>")
    trs = ""
    for c in rows:
        label, s_cls = STATUS_INFO.get(c.get("status"), (str(c.get("status")), "na"))
        plan = c.get("expected_minutes")
        spent = c.get("approx_minutes_spent")
        bar = ""
        if isinstance(plan, (int, float)) and plan and isinstance(spent, (int, float)):
            ratio = max(0.0, min(1.0, spent / plan))
            b_cls = "" if ratio >= 0.7 else (" warn" if ratio >= 0.3 else " bad")
            bar = (f"<div class='bar{b_cls}'><div style='width:{round(ratio * 100)}%'></div></div>"
                   f"<div class='minlbl'>{_e(round(spent))} of {_e(round(plan))} min</div>")
        ev = ""
        if c.get("evidence"):
            at = f" <span class='t'>@{_e(c['approx_time'])}</span>" if c.get("approx_time") else ""
            ev = f"“{_e(c['evidence'])}”{at}"
        if c.get("reconciled"):
            ev += f"<div class='minlbl'>({_e(c['reconciled'])})</div>"
        trs += (f"<tr><td><b>{_e(c.get('section'))}</b></td>"
                f"<td><span class='chip {s_cls}'>{_e(label)}</span></td>"
                f"<td>{bar}</td><td style='font-size:13px;color:#475569'>{ev}</td></tr>")
    link = ""
    if cov.get("proof"):
        link = (f"<div class='legend' style='margin-top:8px'>Full table: "
                f"<a href='{_e(cov['proof'])}'>{_e(cov['proof'])}</a></div>")
    return ("<section class='card'><h2>📋 Curriculum coverage — what the trainer taught</h2>"
            "<div class='lead'>The Day plan's topics, whether the recording shows them being taught, "
            "and the time spent vs the time the plan allots.</div>"
            f"<table class='cov'><tr><th>Topic</th><th>Taught?</th><th>Time spent</th><th>Evidence</th></tr>{trs}</table>"
            f"{link}</section>")



def _video_card(result):
    video = result.get("video") or {}
    if not video.get("enabled"):
        return ""
    tt = ((result.get("transcript") or {}).get("metrics") or {}).get("talk_time") or {}
    cons = (video.get("vision") or {}).get("consistency") or {}
    cam = video.get("camera") or {}

    rows = ""
    same = cons.get("same_person_throughout")
    if same is True:
        rows += _check_row("Same candidate throughout", "ok", "Good",
                           "Face matching found no person swap during the session.")
    elif same is False:
        rows += _check_row("Same candidate throughout", "bad", "Problem",
                           _e(cons.get("note") or "The on-camera person appears to have changed."))
    for role in ("trainer", "candidate"):
        pct = (cam.get(role) or {}).get("on_camera_pct_of_speaking")
        if pct is not None:
            cls = "ok" if pct >= 0.8 else ("warn" if pct >= 0.5 else "bad")
            word = "Good" if pct >= 0.8 else ("Partly" if pct >= 0.5 else "Low")
            rows += _check_row(f"{role.capitalize()} on camera while speaking", cls, word,
                               f"On camera {round(pct * 100)}% of their speaking time.")
    gaze = (video.get("vision") or {}).get("gaze") or []
    likes = [g.get("reading_likelihood") for g in gaze if isinstance(g, dict) and g.get("reading_likelihood") is not None]
    if likes:
        worst = max(likes)
        cls = "ok" if worst < 0.6 else "bad"
        word = "Good" if worst < 0.6 else "Suspicious"
        rows += _check_row("Reading answers from a script?", cls, word,
                           f"Highest reading likelihood across sampled answers: {round(worst * 100)}% "
                           f"(flagged only at 60%+).")

    t_sec, c_sec = tt.get("trainer_sec") or 0, tt.get("candidate_sec") or 0
    split = ""
    if t_sec or c_sec:
        tp = round(100 * t_sec / (t_sec + c_sec))
        split = (f"<div style='margin-top:14px'><b style='font-size:14px'>Who talked how much</b>"
                 f"<div class='split'><div class='a' style='width:{tp}%'></div>"
                 f"<div class='b' style='width:{100 - tp}%'></div></div>"
                 f"<div class='legend'><span class='dot' style='background:#3b82f6'></span>Trainer {tp}% "
                 f"&nbsp;&nbsp;<span class='dot' style='background:#a855f7'></span>Candidate {100 - tp}%</div></div>")

    player = ""
    vid = video.get("analysis_video")
    if vid:
        player = (
            f"<div style='margin-top:16px'><b style='font-size:14px'>Watch what the models saw</b>"
            f"<video controls preload='metadata' style='width:100%;border-radius:10px;margin-top:6px;"
            f"background:#000' src='{_e(vid)}'></video>"
            f"<div class='legend' style='margin-top:4px'>"
            f"<span class='dot' style='background:#ffa03c'></span>orange = trainer &nbsp;"
            f"<span class='dot' style='background:#3cd8e6'></span>cyan = candidate &nbsp;"
            f"<span class='dot' style='background:#6edc6e'></span>green = face not yet identified &nbsp;"
            f"· corner brackets + dots = MediaPipe face; name tag = InsightFace identity "
            f"(carried from the latest check); cyan banner = gaze sample. "
            f"One frame per second of session, played at 4×.</div></div>")

    return ("<section class='card'><h2>🎥 Automatic video checks</h2>"
            "<div class='lead'>Checks done locally on the recording itself.</div>"
            f"<table class='cov'><tr><th>Check</th><th>Result</th><th>Details</th></tr>{rows}</table>{split}{player}</section>")


# ── helpers ──────────────────────────────────────────────────────────────────
def _check_row(name, cls, word, detail):
    icon = {"ok": "✓", "warn": "⚠", "bad": "✗"}.get(cls, "—")
    return (f"<tr><td><b>{_e(name)}</b></td><td><span class='chip {cls}'>{icon} {_e(word)}</span></td>"
            f"<td style='color:#475569;font-size:13.5px'>{detail}</td></tr>")


def _thumbs(paths):
    imgs = [p for p in paths if isinstance(p, str) and p.lower().endswith(_IMG_EXT)]
    if not imgs:
        return ""
    cells = "".join(
        f"<a href='{_e(p)}' title='open full size'><img src='{_e(p)}' alt='evidence frame'>"
        f"<div class='cap'>{_e(os.path.basename(p))}</div></a>"
        for p in imgs)
    return f"<div class='thumbs'>{cells}</div>"


def _planned_minutes(result):
    for f in result.get("flags") or []:
        if isinstance(f, dict) and f.get("type") == "session_too_short":
            return f.get("planned_minutes")
    m = re.search(r"of (\d+) planned", str((result.get("summary") or {}).get("duration", "")))
    return int(m.group(1)) if m else None


def _e(v) -> str:
    return html.escape("" if v is None else str(v))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "output/result.json"
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)
    out = os.path.join(os.path.dirname(os.path.abspath(src)) or ".", "report.html")
    print("wrote", write(data, out))
