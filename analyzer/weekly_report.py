"""Weekly report renderer — turns the aggregate from weekly.aggregate() into
week-report.html. Reuses the daily report's stylesheet + escaping so the two
reports look like one system.

Pure stdlib:  python -m analyzer.weekly ... builds + calls write() here.
"""
import os

from .report import CSS, _e, REASON_INFO


def write(week: dict, out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(render(week))
    return out_path


def _cls(score, good=80, ok=50):
    return "ok" if (score or 0) >= good else ("warn" if (score or 0) >= ok else "bad")


def _dim_bars(dims):
    rows = ""
    for d in dims:
        avg = d.get("avg")
        w = round((avg or 0) * 10)
        per = " · ".join(f"D{p['day']} {_e(p['v'])}" for p in d.get("per_day", []) if p.get("v") is not None)
        num = f"{_e(avg)}/10" if avg is not None else "—"
        rows += (f"<div class='erow'><div class='elbl'>{_e(d['label'])}"
                 f"<div class='minlbl'>{per}</div></div>"
                 f"<div class='ebar'><div style='width:{w}%'></div></div>"
                 f"<div class='enum'>{num}</div></div>")
    return f"<div class='edims'>{rows}</div>"


def _recurring_list(items, mark, cls):
    if not items:
        return "<p class='minlbl'>None noted.</p>"
    lis = "".join(f"<li><span class='m {cls}'>{mark}</span><span>{_e(i['text'])}"
                  + (f" <span class='minlbl'>(&times;{i['count']})</span>" if i.get('count', 1) > 1 else "")
                  + "</span></li>" for i in items)
    return f"<ul class='si'>{lis}</ul>"


EXTRA = (".dayblock{border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-top:14px;"
         "break-inside:avoid}"
         ".dayhead{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline}"
         ".dayhead h3{margin:0;font-size:18px;font-family:var(--serif,inherit)}"
         ".dayblock table{margin-top:8px}")


def _status_chip(status):
    lbl = {"covered": ("✓ Covered", "ok"), "partial": ("◐ Partly", "warn"),
           "not_covered": ("✗ Not covered", "bad")}.get(status, (str(status), "na"))
    return f"<span class='chip {lbl[1]}'>{_e(lbl[0])}</span>"


def _day_detail(per_day):
    blocks = ""
    for p in per_day:
        ic = _cls(p.get("integrity"))
        chips = (f"<span class='chip {ic}'>Integrity {_e(p.get('integrity'))}/100</span> "
                 f"<span class='chip na'>Coverage {_e(p.get('coverage'))}%</span> "
                 + (f"<span class='chip na'>Length {_e(p.get('length_pct'))}%</span> "
                    if p.get("length_pct") is not None else "")
                 + f"<span class='chip na'>{_e(p.get('tier'))}</span>")

        exec_p = f"<p>{_e(p.get('exec_summary'))}</p>" if p.get("exec_summary") else ""

        trows = ""
        for t in (p.get("topics") or []):
            note = f"<div class='minlbl'>{_e(t.get('note'))}</div>" if t.get("note") else ""
            spent, exp = t.get("spent"), t.get("expected")
            tcol = f"{_e(spent)}/{_e(exp)} min" if (spent is not None or exp is not None) else ""
            trows += (f"<tr><td><b>{_e(t.get('section'))}</b>{note}</td>"
                      f"<td>{_status_chip(t.get('status'))}</td>"
                      f"<td class='minlbl'>{tcol}</td></tr>")
        topics_html = (f"<div class='subh'>What was covered</div>"
                       f"<table class='cov'><tr><th>Topic</th><th>Taught?</th><th>Time spent</th></tr>"
                       f"{trows}</table>") if trows else ""

        who = ""
        tr_s, cd_s = p.get("trainer_summary"), p.get("candidate_summary")
        if tr_s or cd_s:
            rows = ""
            if tr_s:
                ov = f"<b>{_e(p.get('trainer_overall'))}/10</b> — " if p.get("trainer_overall") is not None else ""
                rows += f"<tr><td>Trainer</td><td>{ov}{_e(tr_s)}</td></tr>"
            if cd_s:
                ov = f"<b>{_e(p.get('candidate_overall'))}/10</b> — " if p.get("candidate_overall") is not None else ""
                rows += f"<tr><td>Candidate</td><td>{ov}{_e(cd_s)}</td></tr>"
            who = f"<div class='subh'>Trainer &amp; candidate</div><table class='kv'>{rows}</table>"

        fd = p.get("flags_detail") or []
        if fd:
            fchips = "".join(
                f"<span class='chip bad'>{_e(REASON_INFO.get(x['reason'], (x['reason'], ''))[0])} "
                f"(&minus;{abs(x.get('points') or 0)})</span> " for x in fd)
            flags_html = f"<div class='subh'>Flagged this day</div><div class='pills'>{fchips}</div>"
        else:
            flags_html = ("<div class='subh'>Flagged this day</div>"
                          "<div class='pills'><span class='chip ok'>No deductions</span></div>")
        title = p.get("day_title") or p.get("day_step_name") or ""
        href = p.get("report_href")
        open_link = (f"<a class='ts' href='{_e(href)}' target='_blank' rel='noopener'>"
                     f"open full day report &#8599;</a>" if href else "")
        blocks += (f"<div class='dayblock'><div class='dayhead'>"
                   f"<h3>Day {_e(p.get('day'))} <span class='minlbl'>{_e(title)}</span></h3>"
                   f"<span class='minlbl'>{_e(p.get('date'))} {open_link}</span></div>"
                   f"<div class='pills' style='margin-top:8px'>{chips}</div>"
                   f"{exec_p}{topics_html}{who}{flags_html}</div>")
    return ("<section id='detail' class='card'><h2>🗂️ Each day in detail</h2>"
            "<div class='lead'>A full breakdown of every day — what was covered (and to what depth), how "
            "the trainer and candidate did, and what was flagged. Read this to know exactly what happened "
            "on each training day.</div>"
            f"{blocks}</section>")


def render(week: dict) -> str:
    meta = week.get("meta") or {}
    rolled = week.get("rolled") or {}
    per_day = week.get("per_day") or []
    narr = week.get("narrative") or {}

    wi, wc = rolled.get("week_integrity"), rolled.get("week_coverage")
    ic, cc = _cls(wi), _cls(wc, 70, 40)

    miss = meta.get("days_missing") or []
    miss_note = (f"<div class='note'>⚠ Days {', '.join(str(x) for x in miss)} are missing — this week "
                 f"report is built from the {len(meta.get('days_found') or [])} day(s) available.</div>"
                 if miss else "")

    # ── header + headline ───────────────────────────────────────────────────
    header = (f"<header class='banner {ic}'>"
              f"<div class='big-tier'>📅 WEEKLY TRAINING REPORT</div>"
              f"<div class='sub'>Candidate <b>{_e(meta.get('candidate'))}</b> &middot; "
              f"Trainer <b>{_e(meta.get('trainer'))}</b></div>"
              f"<div class='sub'>{_e(meta.get('program'))} &middot; "
              f"{_e(meta.get('date_from'))} – {_e(meta.get('date_to'))} &middot; "
              f"Days {', '.join(str(x) for x in (meta.get('days_found') or []))}</div></header>")

    scores = (
        "<section class='scores'>"
        f"<div class='score-card'><div class='num {ic}'>{_e(wi)}<small>/100</small></div>"
        f"<div class='lbl'>Week integrity</div>"
        f"<div class='hint'>Average of the daily integrity scores (avg {_e(rolled.get('avg_integrity'))}).</div></div>"
        f"<div class='score-card'><div class='num {cc}'>{_e(wc)}%</div>"
        f"<div class='lbl'>Week coverage</div>"
        f"<div class='hint'>Planned topics taught across the whole week (cumulative).</div></div>"
        f"<div class='score-card'><div class='num'>{_e(len(meta.get('days_found') or []))}<small>/5</small></div>"
        f"<div class='lbl'>Days captured</div><div class='hint'>Sessions analysed this week.</div></div>"
        "</section>")

    # ── week executive summary ──────────────────────────────────────────────
    exec_html = ""
    if narr.get("week_executive_summary"):
        exec_html = ("<section class='card summary-final'><h2>🧾 Week in summary</h2>"
                     f"<p style='font-size:15.5px'>{_e(narr['week_executive_summary'])}</p></section>")

    # ── per-day table + trend ───────────────────────────────────────────────
    trs = ""
    for p in per_day:
        tier_cls = _cls(p.get("integrity"))
        fd = p.get("flags_detail") or []
        flags = ", ".join(
            f"{_e(REASON_INFO.get(x['reason'], (x['reason'], ''))[0])} (&minus;{abs(x.get('points') or 0)})"
            for x in fd) or "—"
        length = f"{_e(p.get('length_pct'))}%" if p.get("length_pct") is not None else "—"
        href = p.get("report_href")
        day_lbl = (f"<a href='{_e(href)}' target='_blank' rel='noopener'>Day {_e(p.get('day'))} &#8599;</a>"
                   if href else f"Day {_e(p.get('day'))}")
        trs += (f"<tr><td><b>{day_lbl}</b><div class='minlbl'>{_e(p.get('date'))}</div></td>"
                f"<td><span class='chip {tier_cls}'>{_e(p.get('integrity'))}/100</span></td>"
                f"<td>{_e(p.get('coverage'))}%</td><td>{length}</td>"
                f"<td>{_e(p.get('tier'))}</td>"
                f"<td style='font-size:13px;color:#475569'>{flags}</td></tr>")
    per_day_card = ("<section id='days' class='card'><h2>📊 Day by day</h2>"
                    "<div class='lead'>Each session's scores and what cost points. "
                    "Integrity and coverage are shown per day so you can see the trend across the week.</div>"
                    f"<table class='cov'><tr><th>Session</th><th>Integrity</th><th>Coverage</th>"
                    f"<th>Length</th><th>Tier</th><th>Flags</th></tr>{trs}</table></section>")

    # ── trainer across the week ─────────────────────────────────────────────
    tr = week.get("trainer") or {}
    tr_overall = tr.get("overall_avg")
    tr_big = (f"<div class='estat'><div class='k'>Avg impact</div>"
              f"<div class='v'>{_e(tr_overall)}<small>/10</small></div></div>" if tr_overall is not None else "")
    tr_impact = (f"<div class='subh'>Trainer's impact across the week</div><p>{_e(narr['trainer_impact'])}</p>"
                 if narr.get("trainer_impact") else "")
    tr_summary = (f"<div class='subh'>Trainer perspective &amp; consistency</div>"
                  f"<p>{_e(narr['trainer_week_summary'])}</p>" if narr.get("trainer_week_summary") else "")
    trainer_card = ("<section id='trainer' class='card'><h2>🎯 Trainer across the week</h2>"
                    "<div class='lead'>Average effectiveness across the days (each bar shows the week "
                    "average; the small text is the per-day values), the trainer's impact on the candidate, "
                    "and recurring strengths / areas. Context only — weekly numbers do not carry a score.</div>"
                    f"<div class='egrid'>{tr_big}{_dim_bars(tr.get('dims') or [])}</div>"
                    "<div class='sigrid'>"
                    f"<div><div class='subh'>Recurring strengths</div>{_recurring_list(tr.get('strengths'), '✓', 'ok')}</div>"
                    f"<div><div class='subh'>Recurring areas to improve</div>{_recurring_list(tr.get('improvements'), '➤', 'warn')}</div>"
                    f"</div>{tr_impact}{tr_summary}</section>")

    # ── candidate across the week ───────────────────────────────────────────
    cd = week.get("candidate") or {}
    cd_overall = cd.get("overall_avg")
    cd_big = (f"<div class='estat'><div class='k'>Avg overall</div>"
              f"<div class='v'>{_e(cd_overall)}<small>/10</small></div></div>" if cd_overall is not None else "")
    prog = cd.get("progression") or {}
    prog_labels = {"understanding_1_10": "Understanding", "communication_1_10": "Communication",
                   "overall_1_10": "Overall"}
    prog_chips = ""
    for k, label in prog_labels.items():
        pr = prog.get(k)
        if pr:
            delta = pr["delta"]
            cls = "ok" if delta > 0 else ("bad" if delta < 0 else "na")
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
            prog_chips += (f"<span class='chip {cls}'>{_e(label)}: {_e(pr['from'])} → {_e(pr['to'])} "
                           f"{arrow}{_e(abs(delta))}</span> ")
    prog_html = (f"<div class='subh'>Progression (Day {prog.get('understanding_1_10', {}).get('from_day', 1)} "
                 f"→ last day)</div><div class='pills'>{prog_chips}</div>" if prog_chips else "")
    cam = cd.get("camera_per_day") or []
    cam_html = ""
    if any(c.get("pct") is not None for c in cam):
        chips = "".join(f"<span class='chip {('ok' if (c['pct'] or 0) >= 0.8 else ('warn' if (c['pct'] or 0) >= 0.5 else 'bad'))}'>"
                        f"D{c['day']} {round((c['pct'] or 0) * 100)}%</span> "
                        for c in cam if c.get("pct") is not None)
        cam_html = f"<div class='subh'>On-camera while speaking (per day)</div><div class='pills'>{chips}</div>"
    cd_summary = (f"<div class='subh'>Candidate across the week</div>"
                  f"<p>{_e(narr['candidate_week_summary'])}</p>" if narr.get("candidate_week_summary") else "")
    cd_ack = (f"<div class='subh'>Acknowledgement &amp; feedback</div>"
              f"<p>{_e(narr['candidate_acknowledgement'])}</p>" if narr.get("candidate_acknowledgement") else "")
    candidate_card = ("<section id='candidate' class='card'><h2>👤 Candidate across the week</h2>"
                      "<div class='lead'>How the candidate engaged and progressed over the week, and any "
                      "feedback or acknowledgement they gave. Context only. Appearance / dress is not "
                      "assessed.</div>"
                      f"<div class='egrid'>{cd_big}{_dim_bars(cd.get('dims') or [])}</div>"
                      f"{prog_html}{cam_html}"
                      "<div class='sigrid' style='margin-top:14px'>"
                      f"<div><div class='subh'>Recurring strengths</div>{_recurring_list(cd.get('strengths'), '✓', 'ok')}</div>"
                      f"<div><div class='subh'>Recurring areas to work on</div>{_recurring_list(cd.get('improvements'), '➤', 'warn')}</div>"
                      f"</div>{cd_summary}{cd_ack}</section>")

    # ── coverage + flags across the week ────────────────────────────────────
    cov_rows = ""
    for p in per_day:
        miss_s = ", ".join(_e(s) for s in (p.get("missed_sections") or [])) or "—"
        cov_rows += (f"<tr><td><b>Day {_e(p.get('day'))}</b></td>"
                     f"<td>{_e(p.get('covered'))} covered · {_e(p.get('partial'))} partial · "
                     f"{_e(p.get('missed'))} missed of {_e(p.get('total'))}</td>"
                     f"<td style='font-size:13px;color:#475569'>{miss_s}</td></tr>")
    coverage_card = ("<section id='coverage' class='card'><h2>📋 Curriculum coverage across the week</h2>"
                     f"<div class='lead'>Cumulative week coverage: <b>{_e(wc)}%</b> of all planned topics "
                     "across the five days. Below is each day's coverage and which topics were left "
                     "uncovered.</div>"
                     f"<table class='cov'><tr><th>Session</th><th>Coverage</th><th>Topics not covered</th></tr>"
                     f"{cov_rows}</table></section>")

    tally = week.get("flag_tally") or {}
    flags_card = ""
    if tally:
        trs2 = ""
        for reason, days in sorted(tally.items(), key=lambda kv: -len(kv[1])):
            title = REASON_INFO.get(reason, (reason.replace("_", " ").capitalize(), ""))[0]
            trs2 += (f"<tr><td><b>{_e(title)}</b></td><td>{len(days)} of "
                     f"{len(meta.get('days_found') or [])} days</td>"
                     f"<td class='minlbl'>Days {', '.join(str(x) for x in sorted(set(days)))}</td></tr>")
        flags_card = ("<section id='flags' class='card'><h2>🚩 Flags across the week</h2>"
                      "<div class='lead'>How often each concern came up over the week.</div>"
                      f"<table class='cov'><tr><th>Concern</th><th>How often</th><th>Which days</th></tr>"
                      f"{trs2}</table></section>")

    parts = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Weekly Training Report — {_e(meta.get('candidate'))}</title>",
        f"<style>{CSS}{EXTRA}</style></head><body><div class='page'>",
        header, miss_note, scores, exec_html, per_day_card, _day_detail(per_day),
        trainer_card, candidate_card, coverage_card, flags_card,
        "<div class='dl-wrap noprint'>"
        "<button class='btn big' onclick='downloadReport()'>⬇ Download this report</button>"
        "<div class='minlbl' style='margin-top:8px'>Opens your browser's save dialog — choose "
        "“Save as PDF” to download the full weekly report.</div></div>",
        "<footer>Weekly cumulative report · generated from the five daily result.json files · "
        "for internal quality review.</footer>",
        "</div>",
        "<script>function downloadReport(){ window.print(); }</script>",
        "</body></html>",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    import json
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "week-result.json"
    with open(src, encoding="utf-8") as fh:
        wk = json.load(fh)
    out = os.path.join(os.path.dirname(os.path.abspath(src)) or ".", "week-report.html")
    print("wrote", write(wk, out))
