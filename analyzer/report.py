"""HTML report — renders result.json into output/report.html for non-technical review.

Pure stdlib (html/json/os/re/datetime/difflib) so it can also run on the host
without the container deps:  python -m analyzer.report output/result.json

Design rules:
  - Plain language: every machine reason code maps to a human title + a short
    paragraph explaining what it means and what to check (REASON_INFO).
  - Every timestamp is CLICK-TO-SEEK: clicking it jumps the embedded analysis
    video to that moment (seekTo() maps session seconds -> video time).
  - Every red flag shows its evidence inline: the quote, the timestamp, and the
    proof images copied by proof.py (paths are relative to output/, where
    report.html lives, so <img src="proof/..."> just works when double-clicked).
  - A "Download report (PDF)" button prints the page to PDF.
  - All model/transcript text is escaped — GPT output is untrusted.
"""
import html
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

REASON_INFO = {
    "proxy_interview_coaching": ("Possible proxy-interview coaching",
                                 "The transcript suggests the trainer discussed having someone else sit a real interview on the candidate's behalf. This is surfaced for a human to read in context — it does not change the score."),
    "fabricated_experience":    ("Possible fabricated experience",
                                 "Wording in the transcript suggests coaching the candidate to claim work experience they do not have. Shown for awareness only — it does not change the score."),
    "scripted_deception":       ("Possible scripted / deceptive coaching",
                                 "Wording suggests coaching the candidate to mislead an employer. Shown for awareness only — it does not change the score."),
    "person_change_midsession": ("A different person appeared mid-session",
                                 "Face matching indicates the on-camera candidate was replaced by a different person for a sustained stretch (longer than the short grace window). In a one-to-one session that points to a possible proxy — review the evidence frames and timestamps."),
    "candidate_camera_off":     ("Candidate camera off while speaking",
                                 "The candidate was off camera for most of the time they were talking, so their identity and engagement could not be verified during their answers."),
    "gaze_fixed_offscreen":     ("Candidate may be reading answers",
                                 "The candidate's gaze stayed fixed at a steady off-screen point while answering, which can indicate reading from a script rather than speaking naturally. Treat as a prompt to review, not proof."),
    "session_too_short":        ("Session much shorter than planned",
                                 "The recording is noticeably shorter than the training day plan expects, so some planned material likely could not be covered in the time available."),
    "key_section_rushed":       ("Key topic skipped or rushed",
                                 "A topic the plan allots a lot of time to was skipped or only briefly touched."),
    "non_english_heavy":        ("Large amount of non-English speech",
                                 "A significant portion of the session was delivered in a language other than English."),
    "person_appeared_midsession": ("Extra person appeared mid-session",
                                 "In a group session, someone who was not on camera at the start appeared later and stayed on screen for a while — worth confirming who they are."),
    "person_change_brief":      ("A different face appeared briefly",
                                 "A different face was on camera only briefly (under the review threshold). No marks were deducted; it is shown purely for awareness."),
    "trainer_non_english_heavy": ("Trainer taught heavily in non-English",
                                 "The trainer delivered a large share of the session in a language other than English, beyond the allowed grace band, which can reduce accessibility of the training."),
    "topics_missed":            ("Planned topics not covered",
                                 "Topics from the day's plan that the trainer did not cover. These are penalised here because the session ran short or overall coverage was low — the time to cover them was not used."),
    "topics_shallow":           ("Topics covered too shallowly",
                                 "Topics the trainer touched on but did not take to the depth the plan expects, based on the time spent versus the time allotted."),
    "topics_left_flag":         ("Topics left uncovered (had time)",
                                 "The session ran its full length but some planned topics were still not covered — flagged for review, no marks deducted."),
    "trainer_camera_off":       ("Trainer camera off while teaching",
                                 "The trainer's camera was off for a long continuous stretch while they were speaking. Moments where only a shared screen was visible (no camera tile at all) are not counted — only a confirmed off-camera stretch."),
    "trainer_reading_screen":   ("Trainer may be reading from screen",
                                 "The trainer's gaze stayed fixed and off-centre while explaining on camera — a possible read-aloud rather than live teaching. Reading from a genuinely shared document is not counted against them."),
    "trainer_joined_late":      ("Trainer joined late",
                                 "The trainer (the meeting host) joined noticeably after the scheduled start time, so part of the booked slot was lost before teaching began."),
}

STATUS_INFO = {
    "covered":     ("✓ Covered", "ok"),
    "partial":     ("◐ Partly covered", "warn"),
    "not_covered": ("✗ Not covered", "bad"),
}

TIER_STYLE = {  # tier -> (css class, banner subtitle)
    "Clean":     ("ok",   "No serious concerns found — a routine spot-check is enough."),
    "Review":    ("warn", "Some concerns found — a human should read through the items below."),
    "High-risk": ("bad",  "Serious concerns found — this session needs a human review."),
}

CSS = """
:root{
  --bg:#f4f6fb; --card:#ffffff; --ink:#111827; --muted:#6b7280; --line:#e6e9f2;
  --accent:#4f46e5; --accent-2:#7c3aed;
  --ok:#059669; --warn:#d97706; --bad:#e11d48;
  --ok-bg:#d1fae5; --warn-bg:#fef3c7; --bad-bg:#ffe4e6; --accent-bg:#eef2ff;
  --shadow:0 1px 2px rgba(17,24,39,.06),0 10px 30px rgba(17,24,39,.07);
}
*{box-sizing:border-box;margin:0}
body{font-family:'Segoe UI',system-ui,-apple-system,Inter,Arial,sans-serif;
  background:linear-gradient(180deg,#e9edf8,#f4f6fb 280px);color:var(--ink);line-height:1.62;
  -webkit-font-smoothing:antialiased;}
a{color:var(--accent)}
.topbar{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:14px;
  padding:10px 20px;background:rgba(255,255,255,.82);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line);box-shadow:0 1px 6px rgba(15,23,42,.05);}
.topbar .brand{font-weight:800;font-size:15px;letter-spacing:.2px}
.topbar nav{display:flex;gap:4px;flex-wrap:wrap;margin-left:6px}
.topbar nav a{font-size:12.5px;color:var(--muted);text-decoration:none;padding:5px 10px;border-radius:8px}
.topbar nav a:hover{background:var(--accent-bg);color:var(--accent)}
.topbar .spacer{flex:1}
.btn{border:0;cursor:pointer;font-weight:700;font-size:13px;border-radius:11px;padding:10px 18px;
  background:linear-gradient(120deg,var(--accent),var(--accent-2));color:#fff;
  box-shadow:0 4px 14px rgba(79,70,229,.35)}
.btn:hover{filter:brightness(1.07);transform:translateY(-1px)}
.btn:active{transform:translateY(0)}
.btn.ghost{background:#fff;color:var(--accent);border:1px solid #c7d2fe;box-shadow:none}
.page{max-width:1000px;margin:0 auto;padding:22px 16px 60px}
.banner{border-radius:18px;padding:28px 30px;color:#fff;margin-bottom:18px;box-shadow:var(--shadow)}
.banner .big-tier{font-size:32px;font-weight:800;letter-spacing:.5px;display:flex;align-items:center;gap:12px}
.banner .sub{opacity:.96;margin-top:6px;font-size:15px}
.banner.ok{background:linear-gradient(120deg,#047857,#10b981)}
.banner.warn{background:linear-gradient(120deg,#b45309,#f59e0b)}
.banner.bad{background:linear-gradient(120deg,#be123c,#f43f5e)}
.scores{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.score-card{flex:1 1 200px;background:var(--card);border-radius:16px;padding:18px 20px;box-shadow:var(--shadow)}
.score-card .num{font-size:38px;font-weight:800;line-height:1.05}
.score-card .num small{font-size:16px;font-weight:600;color:var(--muted)}
.score-card .lbl{font-weight:700;margin-top:4px}
.score-card .hint{font-size:12.5px;color:var(--muted);margin-top:6px}
.num.ok{color:var(--ok)}.num.warn{color:var(--warn)}.num.bad{color:var(--bad)}
.track{height:8px;border-radius:999px;background:#eef2f7;margin-top:10px;overflow:hidden}
.track>div{height:100%;border-radius:999px}
.track>div.ok{background:var(--ok)}.track>div.warn{background:var(--warn)}.track>div.bad{background:var(--bad)}
.math{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;color:var(--muted);margin-top:8px}
.card{background:var(--card);border-radius:16px;padding:22px 24px;margin-bottom:18px;box-shadow:var(--shadow);
  border:1px solid transparent;transition:border-color .15s}
.card:hover{border-color:#e6edf6}
.card h2{font-size:20px;margin-bottom:4px;display:flex;align-items:center;gap:9px}
.card .lead{color:var(--muted);font-size:13.5px;margin-bottom:16px;max-width:75ch}
.kv{width:100%;border-collapse:collapse;font-size:14.5px}
.kv td{padding:7px 8px 7px 0;vertical-align:top}
.kv td:first-child{color:var(--muted);white-space:nowrap;width:210px}
.note{background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;border-radius:12px;padding:11px 14px;font-size:13px;margin-bottom:14px}
.flag{border:1px solid var(--line);border-left-width:6px;border-radius:12px;padding:15px 17px;margin-bottom:12px;background:#fff}
.flag.critical,.flag.high{border-left-color:var(--bad)}
.flag.medium{border-left-color:var(--warn)}
.flag.low,.flag.info{border-left-color:var(--muted)}
.flag .head{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:baseline}
.flag .title{font-weight:800;font-size:16px}
.flag .pts{font-weight:800;color:var(--bad);white-space:nowrap;font-size:15px}
.flag .why{color:#475569;font-size:13.5px;margin:5px 0 9px;max-width:80ch}
.quote{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:10px 13px;font-size:14px;margin-bottom:8px}
.quote .t{color:var(--muted);font-size:12.5px}
.thumbs{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.thumbs a{display:block}
.thumbs img{height:118px;border-radius:10px;border:1px solid var(--line);display:block}
.thumbs .cap{font-size:11.5px;color:var(--muted);text-align:center;margin-top:3px}
.chip{display:inline-block;font-size:13px;font-weight:700;padding:2px 9px;border-radius:999px}
.chip.ok{color:#166534;background:var(--ok-bg)}
.chip.warn{color:#92400e;background:var(--warn-bg)}
.chip.bad{color:#991b1b;background:var(--bad-bg)}
.chip.na{color:var(--muted);background:#eef2f7}
table.cov{width:100%;border-collapse:collapse;font-size:14px}
table.cov th{text-align:left;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.4px;
  padding:7px 8px;border-bottom:1px solid var(--line)}
table.cov td{padding:10px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top}
.bar{background:#eef2f7;border-radius:999px;height:8px;width:140px;overflow:hidden}
.bar>div{height:100%;border-radius:999px;background:var(--ok)}
.bar.warn>div{background:var(--warn)}.bar.bad>div{background:var(--bad)}
.minlbl{font-size:12px;color:var(--muted);white-space:nowrap}
.split{display:flex;border-radius:999px;overflow:hidden;height:16px;margin:8px 0 4px}
.split .a{background:#3b82f6}.split .b{background:#a855f7}
.legend{font-size:12.5px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
.ts{display:inline-flex;align-items:center;gap:3px;color:var(--accent);cursor:pointer;
  font-variant-numeric:tabular-nums;background:var(--accent-bg);border:1px solid #e0e7ff;
  border-radius:7px;padding:1px 8px;font-size:12.5px;font-weight:600;text-decoration:none;
  white-space:nowrap;transition:background .12s,color .12s}
.ts::before{content:'▸';font-size:10px;opacity:.85}
.ts:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.pills{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}
.pill{background:#f6f8fc;border:1px solid var(--line);border-radius:999px;padding:5px 11px;font-size:12.5px}
.pill.muted{color:var(--muted);background:#eef1f8}
.pill.toggle{cursor:pointer;color:var(--accent);background:var(--accent-bg);border-color:#c7d2fe;font-weight:700}
.pill.toggle:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.rhythm-row{margin-bottom:14px}
.rhythm-h{font-weight:700;font-size:14px}
.subh{font-weight:700;font-size:14px;margin:14px 0 4px}
footer{color:#94a3b8;font-size:12.5px;text-align:center;margin-top:10px}
.card{overflow-x:auto}
body{font-size:15px}
.card h2{font-size:21px;font-weight:800;letter-spacing:-.2px}
.card .lead{font-size:14px;line-height:1.65}
.subh{font-size:15px;margin:18px 0 8px;padding-bottom:5px;border-bottom:1px solid var(--line);color:#334155}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-top:4px}
.stat{background:linear-gradient(160deg,#fbfdff,#f1f6fc);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.stat .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);font-weight:800}
.stat .v{font-size:28px;font-weight:800;margin-top:8px;line-height:1.1}
.stat .v small{font-size:15px;color:var(--muted);font-weight:700}
.stat .d{font-size:12.5px;color:var(--muted);margin-top:8px}
.gauge{position:relative;height:12px;background:#e9eef5;border-radius:999px;margin-top:10px;overflow:hidden}
.gauge>div{height:100%;border-radius:999px;background:linear-gradient(90deg,#ef4444,#f59e0b,#22c55e)}
.summary-final{background:linear-gradient(160deg,#ffffff,#eef5ff);border:1px solid #d7e3f5}
.verdict{font-size:15.5px;font-weight:600;margin:6px 0 14px;color:#1e293b}
.sum-scores{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 14px}
.sum-pill{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px;text-align:center;box-shadow:var(--shadow)}
.sum-pill .n{font-size:30px;font-weight:800;line-height:1}
.sum-pill .l{font-size:12px;color:var(--muted);margin-top:4px}
ul.flags-list{margin:6px 0 0 18px;font-size:14px}
ul.flags-list li{margin:4px 0}
.dl-wrap{text-align:center;margin:28px 0 10px}
.btn.big{font-size:15px;padding:15px 30px;border-radius:14px}
.kgrid{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.kbtn{display:inline-flex;align-items:center;gap:7px;cursor:pointer;border:1px solid #e0e7ff;
  background:var(--accent-bg);color:var(--ink);border-radius:10px;padding:8px 12px;font-size:13px}
.kbtn b{color:var(--accent);font-variant-numeric:tabular-nums}
.kbtn:hover{background:var(--accent);color:#fff}.kbtn:hover b{color:#fff}
.legend-key{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:#334155}
.legend-key span{display:inline-flex;align-items:center;gap:7px}
.sw{width:13px;height:13px;border-radius:4px;display:inline-block}
.todo{list-style:none;margin:0;padding:0}
.todo li{padding:9px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:flex-start;font-size:14.5px}
.todo li:last-child{border-bottom:0}
.todo .b{color:var(--accent);font-weight:800}
@media (max-width:680px){
  .topbar{flex-wrap:wrap}
  .topbar nav{order:3;width:100%}
  .score-card{flex-basis:100%}
  .kv td:first-child{width:150px}
  .banner{padding:22px 20px}
  .banner .big-tier{font-size:26px}
}
@media print{
  body{background:#fff}
  .topbar,.noprint{display:none!important}
  .card,.score-card,.banner{box-shadow:none;border:1px solid var(--line)}
  a.ts{color:var(--ink);border:0}
}
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
    session_dur = int(meeting.get("duration_sec") or 0)

    parts = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Training Session Review — Day {_e(meeting.get('day'))}</title>",
        f"<style>{CSS}</style></head><body>",
        _topbar(result),
        "<div class='page'>",
        _banner(tier, cls, tier_sub, meeting),
        _score_cards(result),
        _howto_note(),
        _legend_card(result),
        _session_card(meeting, result),
        _attendance_card(result),
        _description_card(result),
        _flags_card(result),
        _key_moments_card(result),
        _coverage_card(result),
        _metrics_card(result),
        _rhythm_card(result),
        _language_card(result),
        _video_card(result),
        _recommendations_card(result),
        _summary_card(result),
        _download_footer(),
        f"<footer>Generated from result.json · analyzed {_e(meeting.get('analyzed_at', ''))[:19].replace('T', ' ')} UTC"
        f" · full evidence in the <b>proof/</b> folder</footer>",
        "</div>",
        _script(session_dur),
        "</body></html>",
    ]
    return "\n".join(parts)


# ── chrome ────────────────────────────────────────────────────────────────────
def _topbar(result):
    meeting = result.get("meeting") or {}
    nav = "".join(f"<a href='#{i}'>{t}</a>" for i, t in (
        ("flags", "Red flags"), ("key-moments", "Key moments"), ("coverage", "Coverage"),
        ("metrics", "Metrics"), ("rhythm", "Rhythm"), ("language", "Language"),
        ("video", "Video"), ("recommendations", "Actions")))
    model = (meeting.get("llm") or {}).get("model")
    return ("<div class='topbar noprint'>"
            f"<span class='brand'>📋 Session Review · Day {_e(meeting.get('day'))}"
            f"{' · ' + _e(model) if model else ''}</span>"
            f"<nav>{nav}</nav><span class='spacer'></span></div>")


def _script(session_dur):
    return (
        "<script>\n"
        f"var SESSION_DUR = {session_dur};\n"
        "var _factor = null;\n"
        "function _vid(){ return document.getElementById('analysisVideo'); }\n"
        "function seekTo(sec){\n"
        "  var v=_vid();\n"
        "  if(!v){ return; }\n"
        "  if(_factor===null && v.duration && SESSION_DUR){ _factor = v.duration / SESSION_DUR; }\n"
        "  var f = _factor || 0.25;\n"
        "  try { v.currentTime = Math.max(0, sec * f); } catch(e) {}\n"
        "  v.scrollIntoView({behavior:'smooth', block:'center'});\n"
        "  if(v.play){ var p=v.play(); if(p&&p.catch){ p.catch(function(){}); } }\n"
        "}\n"
        "function downloadReport(){ window.print(); }\n"
        "function toggleMore(id, el){\n"
        "  var e=document.getElementById(id); if(!e){ return; }\n"
        "  var hidden = (e.style.display==='none');\n"
        "  e.style.display = hidden ? 'inline' : 'none';\n"
        "  el.textContent = hidden ? 'show less' : ('+' + el.getAttribute('data-n') + ' more');\n"
        "}\n"
        "</script>"
    )


# ── sections ────────────────────────────────────────────────────────────────
def _banner(tier, cls, sub, meeting):
    icon = {"ok": "✅", "warn": "⚠️", "bad": "🚨"}.get(cls, "⚠️")
    return (f"<header class='banner {cls}'>"
            f"<div class='big-tier'>{icon} {_e(tier).upper()}</div>"
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

    applied = [d for d in (scoring.get("deductions") or []) if (d.get("points") or 0) != 0]
    if applied:
        math = "100 " + " ".join(f"− {abs(d.get('points') or 0)}" for d in applied) + f" = {integ}"
    else:
        math = f"100 (no deductions) = {integ}"

    planned = _planned_minutes(result)
    actual = round((meeting.get("duration_sec") or 0) / 60.0, 1)
    if planned:
        pct = round(100 * actual / planned)
        d_cls = "ok" if pct >= 50 else "bad"
        dur_num = f"<span class='num {d_cls}'>{pct}%</span>"
        dur_hint = f"{_e(actual)} of {_e(planned)} planned minutes"
        dur_bar = f"<div class='track'><div class='{d_cls}' style='width:{min(100,pct)}%'></div></div>"
    else:
        dur_num = f"<span class='num'>{_e(actual)}<small> min</small></span>"
        dur_hint = "no planned total found"
        dur_bar = ""

    return (
        "<section id='summary' class='scores'>"
        f"<div class='score-card'><div class='num {i_cls}'>{_e(integ)}<small>/100</small></div>"
        f"<div class='lbl'>Session integrity</div>"
        f"<div class='track'><div class='{i_cls}' style='width:{min(100,max(0,integ or 0))}%'></div></div>"
        f"<div class='hint'>Starts at 100; each red flag below removes points.</div>"
        f"<div class='math'>{_e(math)}</div></div>"
        f"<div class='score-card'><div class='num {c_cls}'>{_e(cov)}%</div>"
        f"<div class='lbl'>Day-plan coverage</div>"
        f"<div class='track'><div class='{c_cls}' style='width:{min(100,max(0,cov or 0))}%'></div></div>"
        f"<div class='hint'>How much of the planned curriculum the trainer actually taught.</div></div>"
        f"<div class='score-card'><div class='num'>{dur_num}</div>"
        f"<div class='lbl'>Session length</div>{dur_bar}<div class='hint'>{dur_hint}</div></div>"
        "</section>")


def _howto_note():
    return ("<div class='note noprint'>💡 <b>How to read this report.</b> Two scores sit at the top: "
            "<b>Session integrity</b> (starts at 100, red flags subtract) and <b>Day-plan coverage</b> "
            "(how much of the day's curriculum was taught). Everything below is the evidence behind those "
            "numbers plus context-only sections. <b>Any blue timestamp is clickable</b> — it jumps the "
            "analysis video at the bottom to that exact moment. Use <b>⬇ Download report (PDF)</b> (top-right) "
            "to save or share this review.</div>")


def _llm_line(meeting):
    llm = meeting.get("llm") or {}
    if not llm.get("model"):
        return ""
    effort = f", thinking: {llm['reasoning_effort']}" if llm.get("reasoning_effort") else ""
    if "reasoning_effort" in (llm.get("fields_not_supported") or []):
        effort = ", thinking: not supported by this server yet"
    return f"{llm['model']} ({llm.get('provider')}{effort})" + (" — primary" if llm.get("primary") else "")


def _transcript_line(result):
    tr = result.get("transcript") or {}
    src = tr.get("source")
    if not src:
        return ""
    comb = (tr.get("sources") or {}).get("combine") or {}
    text = {"combined": "Zoom transcript + techsara-whisper, combined (Whisper text, Zoom speaker names)",
            "vtt": "Zoom transcript",
            "techsara-whisper": "techsara-whisper (no speaker names)",
            "whisper": "local Whisper (no speaker names)"}.get(src, src)
    if comb:
        text += (f" · {comb.get('labeled_speech_pct')}% of speech has a speaker name · "
                 f"Zoom vs Whisper text agreement {comb.get('text_agreement_pct')}%")
    return text


def _other_reports(meeting):
    this = (meeting.get("llm") or {}).get("model")
    links = [f"<a href='{_e(r.get('report'))}'>{_e(r.get('model'))}</a>"
             + (" (primary)" if r.get("primary") else "")
             for r in (meeting.get("reports") or []) if r.get("model") != this]
    return " · ".join(links)


def _session_card(meeting, result):
    src = meeting.get("config_source")
    other = _other_reports(meeting)
    rows = [
        ("Training day", f"Day {_e(meeting.get('day'))} — {_e(meeting.get('day_title'))}"),
        ("Session label (Salesforce)", _e(meeting.get("day_step_name"))),
        ("Trainer", _e(meeting.get("trainer_name"))),
        ("Candidate", _e(meeting.get("candidate_name"))),
        ("Video file", _e(meeting.get("video_file"))),
        ("Meeting ID", _e(meeting.get("meeting_id"))),
        ("Recording length", f"{_e(round((meeting.get('duration_sec') or 0) / 60.0, 1))} minutes"),
        ("Session details from", _e(src)),
        ("AI model (this report)", _e(_llm_line(meeting))),
        ("Transcript", _e(_transcript_line(result))),
        ("Same session, other AI model", other),   # already-escaped links
    ]
    trs = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows if v and v != "None")
    warn = ""
    title = (meeting.get("day_title") or "").lower()
    label = (meeting.get("day_step_name") or "").lower()
    if title and label and not (set(title.split()) & set(label.split()) - {"day", "and", "the", "-"}):
        warn = ("<div class='note'>⚠ The Salesforce session label and the Day plan title look like "
                "different topics — the coverage results below judge the session against the Day plan.</div>")
    return (f"<section id='session' class='card'><h2>🗂️ Session details</h2>"
            f"<div class='lead'>The identifying facts for this recording — which training day it is, who "
            f"ran it, and where the metadata came from.</div>{warn}<table class='kv'>{trs}</table></section>")


def _parse_iso(v):
    if not isinstance(v, str) or len(v) < 19:
        return None
    try:
        return datetime.strptime(v[:19] + "Z", "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _mins_after(sched, t):
    a, b = _parse_iso(sched), _parse_iso(t)
    if a is None or b is None:
        return None
    return round((b - a).total_seconds() / 60.0, 1)


def _late_chip(mins):
    if not isinstance(mins, (int, float)):
        return "<span class='chip na'>—</span>"
    cls = "ok" if mins <= 5 else ("warn" if mins <= 10 else "bad")
    word = "on time" if mins <= 5 else f"{round(mins)} min late"
    return f"<span class='chip {cls}'>{_e(word)}</span>"


def _attendance_card(result):
    m = result.get("meeting") or {}
    sched = m.get("scheduled_start")
    dur = m.get("scheduled_duration_min")
    host_join = m.get("trainer_join")
    trainer_late = m.get("trainer_late_min")
    parts = [p for p in (m.get("participants") or []) if isinstance(p, dict)]
    has_data = bool(sched or host_join or parts)
    missing_note = ("" if has_data else
                    "<div class='note'>ℹ️ Scheduled time, join times and the participant list come from "
                    "Zoom (written into training-temp.json by the recording Lambda). They are not available "
                    "for this report — either it is a locally-tested video, or Zoom returned no "
                    "participant/scheduled data for this meeting.</div>")

    host_email = (m.get("host_email") or "").strip().lower()
    trainer_nm = (m.get("trainer_name") or "").strip().lower()
    cand_nm = (m.get("candidate_name") or "").strip().lower()

    def _role(p):
        e = (p.get("email") or "").strip().lower()
        n = (p.get("name") or "").strip().lower()
        if (host_email and e == host_email) or (trainer_nm and SequenceMatcher(None, n, trainer_nm).ratio() > 0.6):
            return "trainer"
        if cand_nm and SequenceMatcher(None, n, cand_nm).ratio() > 0.6:
            return "candidate"
        return "attendee"

    for p in parts:
        p["_role"] = _role(p)

    def _clock(v):
        if not isinstance(v, str) or len(v) < 16:
            return _e(v) if v else "—"
        return f"{_e(v[11:16])} UTC <span class='minlbl'>({_e(v[:10])})</span>"

    cand_joins = sorted(p.get("join") for p in parts if p["_role"] == "candidate" and p.get("join"))
    cand_join = cand_joins[0] if cand_joins else None
    cand_late = _mins_after(sched, cand_join)
    trainer_present = any(p["_role"] == "trainer" for p in parts) or bool(host_join)
    candidate_present = any(p["_role"] == "candidate" for p in parts)

    def _yn(ok):
        return ("<span class='chip ok'>✓ present</span>" if ok
                else "<span class='chip bad'>✗ not detected</span>")

    dur_txt = f"{_e(dur)} min" if isinstance(dur, (int, float)) else "—"
    rows = (f"<tr><td>Scheduled start</td><td><b>{_clock(sched)}</b></td></tr>"
            f"<tr><td>Scheduled duration</td><td><b>{dur_txt}</b></td></tr>"
            f"<tr><td>Trainer (host) joined</td><td><b>{_clock(host_join)}</b> &nbsp; {_late_chip(trainer_late)}</td></tr>"
            f"<tr><td>Candidate joined</td><td><b>{_clock(cand_join)}</b> &nbsp; {_late_chip(cand_late)}</td></tr>"
            f"<tr><td>Everyone present?</td><td>Trainer {_yn(trainer_present)} &nbsp; "
            f"Candidate {_yn(candidate_present)}</td></tr>")

    ptable = ""
    if parts:
        role_label = {"trainer": "Trainer (host)", "candidate": "Candidate", "attendee": "Attendee"}
        order = {"trainer": 0, "candidate": 1, "attendee": 2}
        trs = "".join(
            f"<tr><td><b>{_e(p.get('name'))}</b><div class='minlbl'>{_e(p.get('email'))}</div></td>"
            f"<td>{_e(role_label.get(p['_role'], 'Attendee'))}</td>"
            f"<td>{_clock(p.get('join'))}</td><td>{_clock(p.get('leave'))}</td></tr>"
            for p in sorted(parts, key=lambda x: (order.get(x["_role"], 3), (x.get("name") or "").lower())))
        ptable = (f"<div class='subh'>Participants</div>"
                  f"<table class='cov'><tr><th>Name</th><th>Role</th><th>Joined</th><th>Left</th></tr>{trs}</table>")

    return ("<section id='attendance' class='card'><h2>🕐 Attendance &amp; timing</h2>"
            "<div class='lead'>What the meeting was booked for, when the trainer and candidate actually "
            "joined, how punctual they were, and the full attendee list — all taken from Zoom's own "
            "participant report.</div>"
            f"{missing_note}<table class='kv'>{rows}</table>{ptable}</section>")


def _description_card(result):
    d = result.get("meeting_description") or {}
    if not isinstance(d, dict) or not d.get("overview"):
        return ""
    paras = "".join(f"<p style='margin-bottom:11px'>{_e(p.strip())}</p>"
                    for p in str(d["overview"]).split("\n") if p.strip())

    who = ""
    if d.get("trainer_summary") or d.get("candidate_summary"):
        rows = ""
        if d.get("trainer_summary"):
            rows += f"<tr><td>Trainer</td><td>{_e(d['trainer_summary'])}</td></tr>"
        if d.get("candidate_summary"):
            rows += f"<tr><td>Candidate</td><td>{_e(d['candidate_summary'])}</td></tr>"
        who = f"<div class='subh'>How each side participated</div><table class='kv'>{rows}</table>"

    topics_html = ""
    topic_rows = [t for t in (d.get("topics") or []) if isinstance(t, dict) and t.get("topic")]
    if topic_rows:
        trs = ""
        for t in topic_rows:
            start = t.get("start")
            span = f"{_ts(start)}–{_ts(t.get('end'))}" if start else ""
            mins = t.get("approx_minutes")
            mins_s = f"~{_e(round(mins))} min" if isinstance(mins, (int, float)) else ""
            oc = t.get("on_curriculum")
            plan = t.get("plan_section")
            if oc is True and plan:
                chip = f"<span class='chip ok'>✓ {_e(str(plan).split(':')[0].strip())}</span>"
            elif oc is False:
                chip = "<span class='chip warn'>⚠ off-plan</span>"
            else:
                chip = "<span class='chip na'>—</span>"
            led = _e(t.get("led_by") or "")
            trs += (f"<tr><td><b>{_e(t['topic'])}</b><div class='minlbl'>{span} {mins_s}"
                    f"{(' · led by ' + led) if led else ''}</div></td>"
                    f"<td>{chip}</td>"
                    f"<td style='color:#475569;font-size:13.5px'>{_e(t.get('what_happened'))}</td></tr>")
        topics_html = (f"<div class='subh'>Every topic, checked against the day plan</div>"
                       f"<table class='cov'><tr><th>Topic</th><th>Curriculum?</th>"
                       f"<th>What happened</th></tr>{trs}</table>")
    else:
        chips = "".join(f"<span class='chip na' style='margin:2px 4px 2px 0'>{_e(t)}</span>"
                        for t in (d.get("topics_discussed") or []) if t)
        if chips:
            topics_html = f"<div class='subh'>Topics discussed</div><div>{chips}</div>"

    tl_rows = "".join(
        f"<tr><td style='white-space:nowrap;color:#64748b;width:84px'>{_ts(t.get('time'))}</td>"
        f"<td>{_e(t.get('event'))}</td></tr>"
        for t in (d.get("timeline") or []) if t.get("event"))
    timeline = (f"<div class='subh'>Minute-by-minute</div>"
                f"<table class='cov'>{tl_rows}</table>") if tl_rows else ""

    quotes = "".join(
        f"<div class='quote'>“{_e(q.get('quote'))}”"
        f"<span class='t'> — {_e(q.get('speaker'))} at {_ts(q.get('time'))}</span></div>"
        for q in (d.get("notable_quotes") or []) if q.get("quote"))
    quotes_html = (f"<div class='subh'>Notable quotes</div><div>{quotes}</div>") if quotes else ""

    return ("<section id='description' class='card'><h2>📝 What happened in this meeting</h2>"
            "<div class='lead'>A full, plain-English account of the whole session reconstructed from the "
            "transcript: the narrative, every topic mapped to the plan, a minute-by-minute timeline, and "
            "the quotes that best characterise it. Timestamps are click-to-seek.</div>"
            f"{paras}{who}{topics_html}{timeline}{quotes_html}</section>")


def _flags_card(result):
    deds = (result.get("scoring") or {}).get("deductions") or []
    scored = [d for d in deds if isinstance(d, dict) and (d.get("points") or 0) != 0]
    info_only = [d for d in deds if isinstance(d, dict) and (d.get("points") or 0) == 0]
    body = ""
    if not scored:
        body = "<div class='note'>✅ No red flags reduced the integrity score for this session.</div>"
    for d in scored + info_only:
        reason = d.get("reason", "other")
        title, why = REASON_INFO.get(reason, (reason.replace("_", " ").capitalize(), ""))
        sev = _e(d.get("severity", "medium"))
        pts = d.get("points") or 0
        pts_html = (f"<div class='pts'>−{abs(pts)} points</div>" if pts
                    else "<div class='pts' style='color:#64748b'>review only</div>")
        quote = ""
        if d.get("evidence"):
            at = (f"<span class='t'> — at {_ts(d['approx_time'])} in the recording</span>"
                  if d.get("approx_time") else "")
            quote = f"<div class='quote'>“{_e(d['evidence'])}”{at}</div>"
        extra = _flag_extra(result, reason)
        proof = d.get("proof") or []
        body += (f"<div class='flag {sev}'><div class='head'>"
                 f"<div class='title'>{_e(title)}</div>{pts_html}</div>"
                 f"<div class='why'>{_e(why)}</div>{quote}{extra}"
                 f"{_thumbs(proof)}{_proof_links(proof)}</div>")
    return ("<section id='flags' class='card'><h2>🚩 Red flags &amp; review notes</h2>"
            "<div class='lead'>Each item here is something the system detected. Items with a point value "
            "reduced the integrity score; items marked <b>review only</b> are shown for a human to judge but "
            "did not change the score. The quote and photos are the exact evidence — click a photo to open it "
            "full size, click a timestamp to jump to the moment in the video.</div>"
            f"{body}</section>")


def _coverage_card(result):
    cov = (result.get("transcript") or {}).get("coverage_analysis") or {}
    rows = [c for c in (cov.get("coverage") or []) if isinstance(c, dict)]
    if not rows:
        return ("<section id='coverage' class='card'><h2>📋 Curriculum coverage</h2>"
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
            at = f" <span class='t'>@{_ts(c['approx_time'])}</span>" if c.get("approx_time") else ""
            ev = f"“{_e(c['evidence'])}”{at}"
        if c.get("depth_note"):
            ev += f"<div class='minlbl'>{_e(c['depth_note'])}</div>"
        if c.get("reconciled"):
            ev += f"<div class='minlbl'>({_e(c['reconciled'])})</div>"
        trs += (f"<tr><td><b>{_e(c.get('section'))}</b></td>"
                f"<td><span class='chip {s_cls}'>{_e(label)}</span></td>"
                f"<td>{bar}</td><td style='font-size:13px;color:#475569'>{ev}</td></tr>")
    link = ""
    if cov.get("proof"):
        link = (f"<div class='legend' style='margin-top:8px'>Full table: "
                f"<a href='{_e(cov['proof'])}'>{_e(cov['proof'])}</a></div>")
    return ("<section id='coverage' class='card'><h2>📋 Curriculum coverage — what the trainer taught</h2>"
            "<div class='lead'>Every topic from the Day plan, whether the recording shows it being taught, "
            "how deeply it was covered versus the time the plan allots, and a one-line note on what was "
            "actually said. Topics are matched by meaning, so the trainer did not have to name them or cover "
            "them in order.</div>"
            f"<table class='cov'><tr><th>Topic</th><th>Taught?</th><th>Time spent</th><th>Evidence</th></tr>{trs}</table>"
            f"{link}</section>")


def _metrics_card(result):
    rm = result.get("report_metrics") or {}
    if not rm:
        return ""

    def mm(v, suf=""):
        return (f"{_e(v)}{suf}" if v is not None else "—")

    trp, cap = rm.get("trainer_talk_pct"), rm.get("candidate_talk_pct")
    # Tile 1 — talk split
    if trp is not None and cap is not None:
        talk_v = f"{trp}% <small>: {cap}%</small>"
        talk_d = (f"<div class='split'><div class='a' style='width:{trp}%'></div>"
                  f"<div class='b' style='width:{cap}%'></div></div>"
                  f"<div class='legend'><span class='dot' style='background:#3b82f6'></span>Trainer "
                  f"&nbsp;<span class='dot' style='background:#a855f7'></span>Candidate</div>")
    else:
        talk_v, talk_d = "—", "Trainer vs candidate airtime"
    tile_talk = (f"<div class='stat'><div class='k'>Talk-time split</div>"
                 f"<div class='v'>{talk_v}</div><div class='d'>{talk_d}</div></div>")

    # Tile 2 — candidate:trainer ratio
    ratio = rm.get("talk_time_candidate_vs_trainer")
    tile_ratio = (f"<div class='stat'><div class='k'>Candidate : trainer ratio</div>"
                  f"<div class='v'>{mm(ratio)}</div>"
                  f"<div class='d'>1.0 = equal airtime; below 1 = trainer-led</div></div>")

    # Tile 3 — fluency gauge
    fl = rm.get("trainer_fluency_1_10")
    gauge = (f"<div class='gauge'><div style='width:{int(fl) * 10}%'></div></div>"
             if isinstance(fl, (int, float)) else "")
    fl_note = _e(rm.get("trainer_fluency_note")) if rm.get("trainer_fluency_note") else "Spoken fluency &amp; clarity"
    tile_flu = (f"<div class='stat'><div class='k'>Trainer fluency</div>"
                f"<div class='v'>{mm(fl)}<small>/10</small></div>{gauge}"
                f"<div class='d'>{fl_note}</div></div>")

    # Tile 4 — coverage vs expected
    tile_cov = (f"<div class='stat'><div class='k'>Coverage vs expected</div>"
                f"<div class='v'>{mm(rm.get('coverage_vs_expected_ratio'), '×')}</div>"
                f"<div class='d'>{mm(rm.get('coverage_pct'), '%')} of planned sections taught "
                f"(minutes spent ÷ minutes planned)</div></div>")

    return ("<section id='metrics' class='card'><h2>📊 Session metrics</h2>"
            "<div class='lead'>The shape of the session at a glance — who spoke how much, how fluent the "
            "trainer was, and how the time actually spent compares with the plan. Context only; these "
            "<b>do not change either score</b>.</div>"
            f"<div class='stats-grid'>{tile_talk}{tile_ratio}{tile_flu}{tile_cov}</div></section>")


def _rhythm_card(result):
    rh = ((result.get("transcript") or {}).get("metrics") or {}).get("conversation_rhythm") or {}
    if not rh:
        return ""
    sil = rh.get("long_silences") or []
    ov = rh.get("overlaps") or []

    def _pill(x):
        return f"<span class='pill'>{_tsec(x['start'])}–{_tsec(x['end'])} <b>({_e(x['dur'])}s)</b></span>"

    def _pill_list(items, gid):
        head = "".join(_pill(x) for x in items[:8])
        if len(items) <= 8:
            return f"<div class='pills'>{head}</div>"
        rest = "".join(_pill(x) for x in items[8:])
        n = len(items) - 8
        return (f"<div class='pills'>{head}"
                f"<span id='{gid}' style='display:none'>{rest}</span>"
                f"<span class='pill toggle' data-n='{n}' onclick=\"toggleMore('{gid}',this)\">"
                f"+{n} more</span></div>")

    if sil:
        sil_html = (f"<div class='rhythm-row'><div class='rhythm-h'>Long silences "
                    f"<span class='minlbl'>({rh.get('silence_count')} total, "
                    f"{rh.get('total_silence_sec')}s combined)</span> "
                    f"<span class='chip na'>measured</span></div>{_pill_list(sil, 'sil')}</div>")
    else:
        sil_html = "<div class='rhythm-row'>No long silences detected.</div>"

    if ov:
        ov_html = (f"<div class='rhythm-row'><div class='rhythm-h'>Overlapping speech "
                   f"<span class='minlbl'>({rh.get('overlap_count')} total, {rh.get('overlap_sec')}s)</span> "
                   f"<span class='chip warn'>cross-talk</span></div>{_pill_list(ov, 'ov')}</div>")
    else:
        ov_html = "<div class='rhythm-row'>No overlapping speech detected.</div>"

    return ("<section id='rhythm' class='card'><h2>🎚️ Conversation rhythm</h2>"
            "<div class='lead'>How the session flowed, measured from the transcript timestamps — long pauses "
            "where nobody spoke and moments where two people talked over each other. Surfaced for context "
            "only; none of this affects either score. Click any time to jump to that moment in the video.</div>"
            f"{sil_html}{ov_html}</section>")


def _language_card(result):
    tr = result.get("transcript") or {}
    lang = (tr.get("metrics") or {}).get("language") or {}
    cov = tr.get("coverage_analysis") or {}
    if not lang.get("available") and not cov.get("trainer_non_english_pct"):
        return ""

    rows = ""
    tne = cov.get("trainer_non_english_pct")
    if isinstance(tne, (int, float)):
        lg = cov.get("trainer_non_english_language")
        lg_txt = f" <span class='minlbl'>({_e(lg)})</span>" if lg else ""
        rows += (f"<tr><td>Trainer non-English share</td><td><b>{_e(round(tne))}%</b>"
                 f"{lg_txt}</td></tr>")
    if lang.get("available"):
        rows += f"<tr><td>Total non-English speech</td><td><b>{_e(lang.get('non_english_sec'))}s</b></td></tr>"
        by_role = lang.get("non_english_sec_by_role") or {}
        if by_role:
            parts = ", ".join(f"{_e(k)} {_e(v)}s" for k, v in by_role.items())
            rows += f"<tr><td>By speaker</td><td>{parts}</td></tr>"

    samples = ""
    flagged = [f for f in (lang.get("flagged_segments") or []) if isinstance(f, dict)]
    if flagged:
        def _lang_row(f):
            conf = f.get("confidence")
            conf_html = (f" <span class='minlbl'>{round(conf * 100)}%</span>"
                         if isinstance(conf, (int, float)) else "")
            return (f"<tr><td style='white-space:nowrap;width:84px'>{_ts(f.get('start'))}</td>"
                    f"<td><span class='chip na'>{_e(f.get('speaker_role'))}</span></td>"
                    f"<td><b>{_e(str(f.get('lang')).upper())}</b>{conf_html}</td>"
                    f"<td style='font-size:13px;color:#475569'>{_e(f.get('text'))}</td></tr>")
        trs = "".join(_lang_row(f) for f in flagged[:15])
        extra = (f"<div class='minlbl' style='margin-top:4px'>Showing first 15 of {len(flagged)} detected "
                 f"non-English lines.</div>" if len(flagged) > 15 else "")
        samples = (f"<div class='subh'>Non-English lines detected</div>"
                   f"<table class='cov'><tr><th>Time</th><th>Speaker</th><th>Language</th>"
                   f"<th>What was said</th></tr>{trs}</table>{extra}")
    elif lang.get("available"):
        samples = ("<div class='note' style='margin-top:12px'>✅ No non-Latin-script non-English lines were "
                   "found in the transcript. Hindi or other languages written in English letters are judged "
                   "by the transcript analysis above (the trainer non-English share), not by this per-line "
                   "check.</div>")

    if not rows and not samples:
        return ""
    note = ("<div class='minlbl' style='margin-top:8px'>How this is measured: the trainer's non-English share "
            "comes from the transcript analysis, which reads each sentence by meaning and so handles Hindi "
            "written in English letters and ignores English technical terms. The per-line table only lists "
            "lines in a genuinely non-Latin script (e.g. Devanagari), because automatic detection mislabels "
            "short English sentences and cannot be trusted on Latin text.</div>")
    return ("<section id='language' class='card'><h2>🗣️ Language</h2>"
            "<div class='lead'>How much of the session was delivered in a language other than English, split "
            "by speaker, with the actual lines detected. The trainer's non-English share drives the "
            "<i>trainer taught heavily in non-English</i> flag; the rest is context.</div>"
            f"<table class='kv'>{rows}</table>{samples}{note}</section>")


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
                               f"On camera {round(pct * 100)}% of the time they were speaking.")
    gaze = (video.get("vision") or {}).get("gaze") or []
    likes = [g.get("reading_likelihood") for g in gaze if isinstance(g, dict) and g.get("reading_likelihood") is not None]
    if likes:
        worst = max(likes)
        cls = "ok" if worst < 0.6 else "bad"
        word = "Good" if worst < 0.6 else "Suspicious"
        rows += _check_row("Candidate reading answers from a script?", cls, word,
                           f"Highest reading likelihood across sampled answers: {round(worst * 100)}% "
                           f"(flagged only at 60%+).")

    tr_cam = (video.get("camera_state") or {}).get("trainer") or {}
    lo = tr_cam.get("longest_camera_off_sec")
    if lo is not None and (tr_cam.get("assessable_sec") or 0) > 0:
        cls = "ok" if lo <= 300 else ("warn" if lo <= 480 else "bad")
        word = "Good" if lo <= 300 else ("Long" if lo <= 480 else "Very long")
        at = tr_cam.get("longest_camera_off_start")
        rows += _check_row("Trainer camera on while teaching", cls, word,
                           f"Longest continuous camera-off stretch while speaking: {round(lo / 60, 1)} min"
                           f"{(' (from ' + _ts(at) + ')') if at else ''} — screen-share-only time excluded.")

    tr_read = (video.get("vision") or {}).get("trainer_reading") or []
    tr_likes = [r.get("reading_likelihood") for r in tr_read
                if isinstance(r, dict) and not r.get("document_shared") and r.get("reading_likelihood") is not None]
    if tr_likes:
        worst_t = max(tr_likes)
        cls = "ok" if worst_t < 0.6 else "bad"
        word = "Good" if worst_t < 0.6 else "Reading?"
        rows += _check_row("Trainer reading from screen while explaining?", cls, word,
                           f"Highest reading likelihood during on-camera teaching: {round(worst_t * 100)}% "
                           f"(flagged at 60%+; reading a shared document is not counted).")

    t_sec, c_sec = tt.get("trainer_sec") or 0, tt.get("candidate_sec") or 0
    split = ""
    if t_sec or c_sec:
        tp = round(100 * t_sec / (t_sec + c_sec))
        split = (f"<div class='subh'>Who talked how much</div>"
                 f"<div class='split'><div class='a' style='width:{tp}%'></div>"
                 f"<div class='b' style='width:{100 - tp}%'></div></div>"
                 f"<div class='legend'><span class='dot' style='background:#3b82f6'></span>Trainer {tp}% "
                 f"&nbsp;&nbsp;<span class='dot' style='background:#a855f7'></span>Candidate {100 - tp}%</div>")

    player = ""
    vid = video.get("analysis_video")
    if vid:
        player = (
            f"<div class='subh'>Watch what the models saw</div>"
            f"<video id='analysisVideo' controls preload='metadata' style='width:100%;border-radius:12px;"
            f"background:#000' src='{_e(vid)}'></video>"
            f"<div class='legend' style='margin-top:5px'>"
            f"<span class='dot' style='background:#ffa03c'></span>orange = trainer &nbsp;"
            f"<span class='dot' style='background:#3cd8e6'></span>cyan = candidate &nbsp;"
            f"<span class='dot' style='background:#6edc6e'></span>green = face not yet identified &nbsp;"
            f"· corner brackets + dots = MediaPipe face; name tag = InsightFace identity; cyan banner = gaze "
            f"sample. One frame per second of session, played at 4×. Any blue timestamp above jumps here.</div>")

    return ("<section id='video' class='card'><h2>🎥 Automatic video checks</h2>"
            "<div class='lead'>Everything in this section is computed locally from the recording itself — "
            "no cloud vision. It checks who was on camera while speaking, whether the on-screen person "
            "stayed consistent, and whether gaze suggests reading. Each check is a prompt for human review, "
            "not a verdict.</div>"
            f"<table class='cov'><tr><th>Check</th><th>Result</th><th>Details</th></tr>{rows}</table>"
            f"{split}{player}</section>")


def _summary_card(result):
    sc = result.get("scoring") or {}
    integ = sc.get("session_integrity_score")
    cov = sc.get("trainer_coverage_score")
    tier = sc.get("tier") or "Review"
    i_cls = "ok" if (integ or 0) >= 80 else ("warn" if (integ or 0) >= 50 else "bad")
    c_cls = "ok" if (cov or 0) >= 70 else ("warn" if (cov or 0) >= 40 else "bad")
    verdict = {
        "Clean": "Overall this session looks clean — a routine spot-check is enough.",
        "Review": "Overall this session has some concerns a human should read through below.",
        "High-risk": "Overall this session has serious concerns and needs a human review.",
    }.get(tier, "")

    scored = [d for d in (sc.get("deductions") or []) if (d.get("points") or 0) != 0]
    if scored:
        items = "".join(
            f"<li><b>{_e(REASON_INFO.get(d.get('reason'), (d.get('reason'), ''))[0])}</b> — "
            f"−{abs(d.get('points') or 0)} pts"
            + (f", at {_ts(d['approx_time'])}" if d.get("approx_time") else "") + "</li>"
            for d in scored)
        flags_html = f"<div class='subh'>What cost points</div><ul class='flags-list'>{items}</ul>"
    else:
        flags_html = ("<div class='subh'>What cost points</div>"
                      "<p>Nothing — the integrity score stayed at 100.</p>")

    s = result.get("summary") or {}
    covline = _e(s.get("trainer_coverage") or "")
    durline = _e(s.get("duration") or "")

    return ("<section id='report-summary' class='card summary-final'><h2>🧾 Report summary</h2>"
            f"<div class='verdict'>{_e(verdict)}</div>"
            "<div class='sum-scores'>"
            f"<div class='sum-pill'><div class='n num {i_cls}'>{_e(integ)}</div>"
            f"<div class='l'>Integrity / 100 · {_e(tier)}</div></div>"
            f"<div class='sum-pill'><div class='n num {c_cls}'>{_e(cov)}%</div>"
            f"<div class='l'>Day-plan coverage</div></div></div>"
            f"<p style='font-size:14.5px'>Length: {durline}. Coverage: {covline}.</p>"
            f"{flags_html}"
            "<p class='minlbl' style='margin-top:12px'>Every point above is backed by the evidence in the "
            "sections of this report and in the <b>proof/</b> folder.</p></section>")


def _legend_card(result):
    return ("<section class='card'><h2>🔑 Legend</h2>"
            "<div class='legend-key'>"
            "<span><span class='sw' style='background:var(--ok)'></span>Good / clean</span>"
            "<span><span class='sw' style='background:var(--warn)'></span>Needs a look</span>"
            "<span><span class='sw' style='background:var(--bad)'></span>Problem / high-risk</span>"
            "<span><span class='sw' style='background:var(--accent)'></span>Blue timestamp "
            "(e.g. <a class='ts' onclick='seekTo(0)'>00:00</a>) = click to jump the video</span>"
            "</div></section>")


def _key_moments_card(result):
    moments = []
    for d in (result.get("scoring") or {}).get("deductions") or []:
        if (d.get("points") or 0) != 0 and d.get("approx_time"):
            s = _secs(d["approx_time"])
            if s is not None:
                moments.append((s, "🚩 " + REASON_INFO.get(d.get("reason"), (d.get("reason"), ""))[0]))
    desc = result.get("meeting_description") or {}
    for q in (desc.get("notable_quotes") or []):
        if isinstance(q, dict) and q.get("time"):
            s = _secs(q["time"])
            if s is not None:
                moments.append((s, "💬 " + str(q.get("speaker") or "quote")))
    cov = (result.get("transcript") or {}).get("coverage_analysis") or {}
    for c in (cov.get("coverage") or []):
        if isinstance(c, dict) and c.get("approx_time") and c.get("status") in ("not_covered", "partial"):
            s = _secs(c["approx_time"])
            if s is not None:
                moments.append((s, "📋 " + str(c.get("section"))))
    rh = ((result.get("transcript") or {}).get("metrics") or {}).get("conversation_rhythm") or {}
    for x in sorted((rh.get("long_silences") or []), key=lambda z: z.get("dur", 0), reverse=True)[:3]:
        moments.append((int(x.get("start", 0)), f"⏸ {x.get('dur')}s silence"))

    seen, uniq = set(), []
    for s, l in sorted(moments):
        if (s, l) in seen:
            continue
        seen.add((s, l))
        uniq.append((s, l))
    if not uniq:
        return ""
    btns = "".join(f"<button class='kbtn' onclick='seekTo({s})'><b>{_e(_fmt_sec(s))}</b> {_e(l)}</button>"
                   for s, l in uniq[:14])
    return ("<section id='key-moments' class='card'><h2>⭐ Key moments</h2>"
            "<div class='lead'>The moments most worth watching — red flags, notable quotes, uncovered "
            "topics and the longest pauses. Click any one to jump straight to it in the video.</div>"
            f"<div class='kgrid'>{btns}</div></section>")


def _recommendations_card(result):
    sc = result.get("scoring") or {}
    scored = [d for d in (sc.get("deductions") or []) if (d.get("points") or 0) != 0]
    rows = [c for c in ((result.get("transcript") or {}).get("coverage_analysis") or {}).get("coverage", [])
            if isinstance(c, dict)]

    def names(pred):
        got = [str(c.get("section")) for c in rows if pred(c)]
        return ", ".join(got) if got else "the flagged topics"

    acts = []
    for d in scored:
        r = d.get("reason")
        if r == "session_too_short":
            acts.append("Use the full scheduled time — the session ended early, so planned material was cut.")
        elif r == "topics_missed":
            acts.append("Cover the topics that were missed: " + names(lambda c: c.get("status") == "not_covered") + ".")
        elif r == "topics_shallow":
            acts.append("Go deeper on under-covered topics: " + names(
                lambda c: c.get("status") == "partial"
                or (c.get("status") == "covered" and c.get("depth_vs_allotted") == "under_covered")) + ".")
        elif r == "trainer_non_english_heavy":
            acts.append("Deliver more of the session in English — a large share was non-English.")
        elif r == "trainer_camera_off":
            acts.append("Keep the camera on while teaching.")
        elif r == "trainer_reading_screen":
            acts.append("Explain from understanding rather than reading from the screen.")
        elif r == "trainer_joined_late":
            acts.append("Join on time — the trainer joined after the scheduled start.")
        elif r == "person_change_midsession":
            acts.append("Verify the candidate's identity — the on-screen person appears to have changed.")
        elif r == "candidate_camera_off":
            acts.append("Ask the candidate to keep their camera on while answering.")
        else:
            acts.append(REASON_INFO.get(r, (r, ""))[0] + " — review and address.")

    seen, uniq = set(), []
    for a in acts:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    if not uniq:
        body = "<div class='note'>✅ No action items — this session met the bar on every scored check.</div>"
    else:
        body = "<ul class='todo'>" + "".join(
            f"<li><span class='b'>▸</span><span>{_e(a)}</span></li>" for a in uniq) + "</ul>"
    return ("<section id='recommendations' class='card'><h2>✅ Recommendations / action items</h2>"
            "<div class='lead'>Concrete things to improve next time, derived from the red flags above.</div>"
            f"{body}</section>")


def _download_footer():
    return ("<div class='dl-wrap noprint'>"
            "<button class='btn big' onclick='downloadReport()'>⬇ Download this report</button>"
            "<div class='minlbl' style='margin-top:8px'>Opens your browser's save dialog — choose "
            "“Save as PDF” to download the full review to your computer.</div></div>")


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


def _proof_links(paths):
    """Link to the plain-text proof file(s) saved by proof.py for this flag."""
    txts = [p for p in paths if isinstance(p, str) and p.lower().endswith(".txt")]
    if not txts:
        return ""
    links = " · ".join(f"<a href='{_e(p)}'>{_e(os.path.basename(p))}</a>" for p in txts)
    return f"<div class='legend' style='margin-top:7px'>📄 Full proof: {links}</div>"


def _find_flag(result, ftype):
    for f in result.get("flags") or []:
        if isinstance(f, dict) and f.get("type") == ftype:
            return f
    return None


def _stat_tile(k, v, small=""):
    sm = f"<small> {small}</small>" if small else ""
    return f"<div class='stat'><div class='k'>{_e(k)}</div><div class='v'>{_e(v)}{sm}</div></div>"


def _flag_extra(result, reason):
    """Evidence block tailored to each flag: timing stats for session/lateness
    flags, and per-topic timestamped quotes for the coverage flags."""
    if reason == "session_too_short":
        f = _find_flag(result, "session_too_short")
        if f:
            return ("<div class='stats-grid' style='margin-top:10px'>"
                    + _stat_tile("Planned", f.get("planned_minutes"), "min")
                    + _stat_tile("Actual", f.get("actual_minutes"), "min")
                    + _stat_tile("Short by", f.get("minutes_short"), "min")
                    + "</div>")
        return ""
    if reason == "trainer_joined_late":
        f = _find_flag(result, "trainer_joined_late")
        if f:
            def _clock(v):
                return f"{v[11:16]} UTC" if isinstance(v, str) and len(v) >= 16 else (str(v) if v else "—")
            return ("<div class='stats-grid' style='margin-top:10px'>"
                    f"<div class='stat'><div class='k'>Scheduled start</div>"
                    f"<div class='v' style='font-size:20px'>{_e(_clock(f.get('scheduled_start')))}</div></div>"
                    f"<div class='stat'><div class='k'>Trainer joined</div>"
                    f"<div class='v' style='font-size:20px'>{_e(_clock(f.get('trainer_join')))}</div></div>"
                    + _stat_tile("Late by", f.get("trainer_late_min"), "min")
                    + "</div>")
        return ""

    cov = (result.get("transcript") or {}).get("coverage_analysis") or {}
    rows = [c for c in (cov.get("coverage") or []) if isinstance(c, dict)]
    if reason == "topics_missed":
        sel = [c for c in rows if c.get("status") == "not_covered"]
    elif reason == "topics_shallow":
        sel = [c for c in rows if c.get("status") == "partial"
               or (c.get("status") == "covered" and c.get("depth_vs_allotted") == "under_covered")]
    else:
        return ""
    if not sel:
        return ""
    items = ""
    for c in sel:
        exp, sp = c.get("expected_minutes"), c.get("approx_minutes_spent")
        tline = (f"expected ~{_e(exp)} min, spent ~{_e(sp)} min"
                 if exp is not None or sp is not None else "")
        note = f"<div class='minlbl'>{_e(c['depth_note'])}</div>" if c.get("depth_note") else ""
        quote = ""
        if c.get("evidence"):
            at = f" <span class='t'>@{_ts(c['approx_time'])}</span>" if c.get("approx_time") else ""
            quote = f"<div class='quote' style='margin-top:5px'>“{_e(c['evidence'])}”{at}</div>"
        items += (f"<div style='margin:9px 0'><b>{_e(c.get('section'))}</b> "
                  f"<span class='minlbl'>{tline}</span>{note}{quote}</div>")
    return f"<div class='subh' style='margin-top:10px'>Evidence per topic</div>{items}"


def _fmt_sec(sec) -> str:
    sec = int(sec or 0)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _secs(mmss):
    """Parse 'mm:ss' or 'hh:mm:ss' -> integer seconds, or None."""
    if not isinstance(mmss, str):
        return None
    p = mmss.strip().split(":")
    if not (2 <= len(p) <= 3) or not all(x.isdigit() for x in p):
        return None
    p = [int(x) for x in p]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]


def _ts(mmss):
    """Clickable 'mm:ss' timestamp that seeks the analysis video. Plain text if unparseable."""
    s = _secs(mmss)
    if s is None:
        return _e(mmss) if mmss else "—"
    return f"<a class='ts' onclick='seekTo({s})' title='Jump to {_e(mmss)} in the video'>{_e(mmss)}</a>"


def _tsec(seconds):
    """Clickable timestamp built from a raw seconds value."""
    if not isinstance(seconds, (int, float)):
        return "—"
    lbl = _fmt_sec(seconds)
    return f"<a class='ts' onclick='seekTo({int(seconds)})' title='Jump to {lbl} in the video'>{lbl}</a>"


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
