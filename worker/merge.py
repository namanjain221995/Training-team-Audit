"""Session merge — one meeting recorded in several chunks becomes ONE day result.

Zoom keeps the same meeting_id when a host stops/restarts, so one logical
training day can land as several Time-*/{meeting_id}/ recordings under the same
date. The worker analyzes each chunk on arrival, then calls merge() over ALL
sibling chunk results to (re)build the day-level file. Idempotent: a late chunk
just re-merges.

Strategies (MERGE_STRATEGY env):
  stitch  (default) — the chunks are parts of one day: CONCATENATE their
                      transcripts in start order (offsetting each later chunk's
                      timestamps by the summed duration of the earlier ones),
                      then re-run the LLM coverage + description ONCE over the
                      whole day, sum the raw video seconds, and score once. This
                      is the only way the model sees the full session, so
                      coverage / topics / effectiveness are correct for the day.
  longest           — treat short extras as false starts: keep the longest
                      chunk's result as the day result; shorter recordings are
                      dropped (false starts also land in merge.dropped_false_starts).

False starts (host opens + instantly restarts, a few-second recording) are
dropped below MERGE_MIN_CHUNK_SEC before stitching, so a 20-second stub never
pollutes the day. If EVERY chunk is short they are all kept (genuine short day).

The golden rule (why chunk results store *_sec primitives): never average
per-chunk percentages — sum the raw seconds, then compute ratios ONCE here. The
day length that drives session_too_short is the SUM of the chunk durations.
"""
import copy
import os
import re
from datetime import datetime, timezone

from analyzer import scoring
from analyzer import transcript as T
from analyzer.config import day_sections

# {Date}/Time-{h}-{mm}-{AM|PM}-IST/ in the S3 key = the recording's true start
_TIME_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/Time-(\d{1,2})-(\d{2})-(AM|PM)-IST/", re.I)

# recordings shorter than this (seconds) are treated as false starts and dropped
# before stitching (host opened + restarted). Never drops the only/all chunks.
MIN_CHUNK_SEC = float(os.environ.get("MERGE_MIN_CHUNK_SEC", "120") or 120)


def merge(chunks, strategy, cfg, oa_list=None, chunk_transcripts=None):
    """chunks: [(s3_key, chunk_result_dict), ...] — at least one.
    cfg: the job Config (carries day_plan, day_number, trainer/candidate names,
         trainer_late_min, model settings). oa_list: LLM clients (make_clients)
         for the >1 re-run, in fallback order. chunk_transcripts: {s3_key:
         segments_list} the worker pre-downloaded for each chunk. Returns the
         session (day-level) result."""
    chunks = sorted(chunks, key=_recording_sort_key)   # true recording-start order
    survivors, dropped = _drop_false_starts(chunks)
    work = survivors or chunks                          # never drop everything
    multi = len(work) > 1

    if not multi:
        session = _single(work[0])
    elif strategy == "longest":
        session = _single(max(work, key=lambda kr: (kr[1].get("meeting") or {})
                              .get("duration_sec") or 0))
    else:
        session = _stitch_llm(work, cfg, oa_list or [], chunk_transcripts or {})

    session["merge"] = {
        "strategy": "single" if not multi else strategy,
        "chunk_count": len(work),
        "recordings_total": len(chunks),
        "chunks": [{"key": k,
                    "video_file": (r.get("meeting") or {}).get("video_file"),
                    "duration_sec": (r.get("meeting") or {}).get("duration_sec")}
                   for k, r in work],
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }
    if dropped:
        session["merge"]["dropped_false_starts"] = [
            {"key": k, "duration_sec": (r.get("meeting") or {}).get("duration_sec")}
            for k, r in dropped]
    if not multi or strategy == "longest":
        # deductions carried from a chunk keep their proof/... paths, which are
        # relative to THAT chunk's meeting folder (see merge.chunks[].key)
        session["merge"]["proof_location"] = "chunk meeting folder (see merge.chunks[].key)"
    return session


# ── false-start drop ─────────────────────────────────────────────────────────
def _drop_false_starts(chunks):
    """Split chunks into (survivors, dropped). A chunk under MIN_CHUNK_SEC is a
    false start. Never returns an empty survivors list (if all are short they are
    all kept and treated as genuine)."""
    if len(chunks) <= 1:
        return list(chunks), []
    survivors, dropped = [], []
    for k, r in chunks:
        dur = (r.get("meeting") or {}).get("duration_sec") or 0
        (dropped if dur < MIN_CHUNK_SEC else survivors).append((k, r))
    if not survivors:
        return list(chunks), []
    return survivors, dropped


# ── single / keep-longest (no re-score: the one recording IS the whole day) ──
def _single(kr):
    return copy.deepcopy(kr[1])


def _force_incomplete(result, reason):
    """Withhold scores on a day we could not fully analyse (NOT a real 0)."""
    meta = result.get("meeting") or {}
    meta["analysis_status"] = "incomplete"
    meta["analysis_reason"] = reason
    sc = result.get("scoring") or {}
    sc["analysis_status"] = "incomplete"
    sc["analysis_reason"] = reason
    sc["trainer_coverage_score"] = None
    sc["session_integrity_score"] = None
    sc["tier"] = None
    return result


# ── stitch: concat transcripts + re-run the LLM ONCE over the whole day ──────
def _stitch_llm(work, cfg, oa_list, chunk_transcripts):
    # work is already sorted by true recording-start order (merge() sort)
    results = [r for _, r in work]
    longest = max(results, key=lambda r: (r.get("meeting") or {}).get("duration_sec") or 0)
    first_meta = (work[0][1].get("meeting") or {})   # the EARLIEST recording

    # 1) concatenate the chunk transcripts in recording order, offsetting each
    #    later chunk by the cumulative duration of the earlier ones -> one day.
    #    Track chunks whose transcript failed to load (missing) — a day stitched
    #    on only a FRACTION of its speech must not be scored as if complete.
    merged_segs, offset, seams, missing = [], 0.0, [], 0
    for idx, (key, res) in enumerate(work):
        segs = chunk_transcripts.get(key) or []
        rec_dur = (res.get("meeting") or {}).get("duration_sec") or 0
        if not segs:
            missing += 1
        if idx > 0:
            seams.append(offset)
        if segs:
            base = min(s.get("start") or 0 for s in segs)
            span = (max(s.get("end") or 0 for s in segs) - base)
            for s in segs:
                merged_segs.append({**s,
                                    "start": (s.get("start") or 0) - base + offset,
                                    "end": (s.get("end") or 0) - base + offset})
            offset += max(span, rec_dur)
        else:
            offset += rec_dur

    summed_dur = round(sum((r.get("meeting") or {}).get("duration_sec") or 0
                           for r in results), 1)

    # trainer lateness belongs to the FIRST recording (the real session start),
    # not whichever chunk happened to trigger this merge. Override on a copy so
    # the caller's job Config is not mutated.
    cfg_day = copy.copy(cfg)
    cfg_day.trainer_late_min = first_meta.get("trainer_late_min")
    cfg_day.scheduled_start = first_meta.get("scheduled_start")
    cfg_day.trainer_join = first_meta.get("trainer_join")

    # 2) meta: summed duration, identity carried from the longest chunk, but the
    #    attendance/lateness fields from the FIRST recording
    meta = dict(longest.get("meeting") or {})
    meta["duration_sec"] = summed_dur
    meta["video_file"] = f"{len(results)} recordings (merged)"
    meta["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    meta["config_source"] = "session-merge"
    meta["scheduled_start"] = first_meta.get("scheduled_start")
    meta["trainer_join"] = first_meta.get("trainer_join")
    meta["trainer_late_min"] = first_meta.get("trainer_late_min")
    meta.pop("analysis_status", None)      # recomputed by scoring.assemble below
    meta.pop("analysis_reason", None)
    meta.pop("llm_failures", None)

    sections_cfg = day_sections(cfg_day) or {}

    # 3) deterministic metrics over the merged transcript (talk-time/roles/lang),
    #    but the day length is the SUMMED recordings, not the transcript span
    metrics = T.compute_metrics(merged_segs, cfg_day) if merged_segs else {"duration_sec": 0}
    metrics["duration_sec"] = summed_dur
    _filter_seam_silences(metrics, seams)

    # 4) ONE LLM pass over the whole day — describe first, then coverage (same
    #    order api.run uses). Fall back across providers; record the WINNING
    #    client + any failures so a model that can't produce usable output
    #    becomes a provable failure (incomplete) rather than a fake 0% day.
    description, coverage, used_oa, failures = None, None, None, []
    for oa in oa_list:
        try:
            description = T.describe_meeting(
                merged_segs, oa, session_label=meta.get("day_step_name"),
                sections_cfg=sections_cfg)
            coverage = T.analyze_coverage(
                merged_segs, sections_cfg, oa, session_label=meta.get("day_step_name"),
                topics=(description or {}).get("topics"))
            used_oa = oa
            break
        except Exception as exc:
            failures.append({"provider": getattr(oa, "provider", None),
                             "model": getattr(oa, "model", None),
                             "error": repr(exc)[:500]})
            description, coverage = None, None
    if failures:
        meta["llm_failures"] = failures
    if used_oa is not None:
        stats = getattr(used_oa, "stats", {}) or {}
        meta["llm"] = {"provider": getattr(used_oa, "provider", None),
                       "model": getattr(used_oa, "model", None),
                       "calls": stats.get("calls"), "corrections": stats.get("corrections")}

    transcript_result = {
        "source": "merged-llm",
        "segment_count": len(merged_segs),
        "metrics": metrics,
        "coverage_analysis": coverage or {
            "coverage": [], "integrity_flags": [], "coverage_pct": None,
            "expected_section_count": len(sections_cfg.get("sections") or [])},
        "meeting_description": description or {"available": False},
    }

    # 5) video is merged from the per-chunk results — never re-derived from pixels
    video = _merge_video(results)

    result = scoring.assemble(meta, transcript_result, video, cfg_day)

    # a day analysed on only part of its transcript (some chunks failed to load)
    # is NOT a real score — withhold it
    if missing:
        result = _force_incomplete(result, "partial_transcript")
    return result


def _filter_seam_silences(metrics, seams):
    """A chunk boundary looks like a long silence to conversation_rhythm (the gap
    between recordings, whose END = the seam). Drop those so the merged rhythm
    isn't polluted, and recompute the derived counts so they stay consistent."""
    if not seams or not isinstance(metrics.get("conversation_rhythm"), dict):
        return
    rh = metrics["conversation_rhythm"]
    lst = rh.get("long_silences")
    if isinstance(lst, list):
        kept = [g for g in lst if not _near_seam(g, seams)]
        rh["long_silences"] = kept
        rh["silence_count"] = len(kept)
        rh["total_silence_sec"] = round(sum((g.get("dur") or 0)
                                            for g in kept if isinstance(g, dict)), 1)


def _near_seam(gap, seams, tol=4.0):
    """A seam silence has end ~= seam (and start = seam - dur). Match on EITHER
    endpoint so the inter-recording gap is caught regardless of its length."""
    if not isinstance(gap, dict):
        return False
    for pt in (gap.get("end"), gap.get("start")):
        if pt is not None and any(abs(pt - s) <= tol for s in seams):
            return True
    return False


# ── video merge (sum raw seconds, recompute ratios once) ─────────────────────
def _merge_video(results) -> dict:
    vids = [r.get("video") or {} for r in results]
    if not any(v.get("enabled") for v in vids):
        return {"enabled": False}

    cam = {"trainer": {"speaking_sec": 0.0, "on_camera_sec": 0.0},
           "candidate": {"speaking_sec": 0.0, "on_camera_sec": 0.0}}
    # camera_state drives trainer_camera_off; trainer_reading drives
    # trainer_reading_screen — both are SCORED, so they MUST survive the merge
    cstate = {role: {"on_sec": 0.0, "camera_off_sec": 0.0, "unassessable_sec": 0.0,
                     "assessable_sec": 0.0, "longest_camera_off_sec": 0.0,
                     "longest_camera_off_start": None}
              for role in ("trainer", "candidate")}
    trainer_reading = []
    face = noface = frames = 0.0
    gaze = []
    cons_samples = 0
    same_flags, trainer_p, cand_p, cand_ids = [], [], [], []
    for v in vids:
        if not v.get("enabled"):
            continue
        frames += v.get("frames_sampled") or 0
        face += v.get("face_visible_sec") or 0
        noface += v.get("screen_share_or_noface_sec") or 0
        for role in cam:
            c = (v.get("camera") or {}).get(role) or {}
            cam[role]["speaking_sec"] += c.get("speaking_sec") or 0
            cam[role]["on_camera_sec"] += c.get("on_camera_sec") or 0
        cs = v.get("camera_state") or {}
        for role in cstate:
            rc = cs.get(role) or {}
            acc = cstate[role]
            acc["on_sec"] += rc.get("on_sec") or 0
            acc["camera_off_sec"] += rc.get("camera_off_sec") or 0
            acc["unassessable_sec"] += rc.get("unassessable_sec") or 0
            acc["assessable_sec"] += rc.get("assessable_sec") or 0
            # a continuous off-stretch cannot span two separate recordings, so the
            # worst single-chunk stretch is the day's worst
            lo = rc.get("longest_camera_off_sec") or 0
            if lo > acc["longest_camera_off_sec"]:
                acc["longest_camera_off_sec"] = lo
                acc["longest_camera_off_start"] = rc.get("longest_camera_off_start")
        vis = v.get("vision") or {}
        gaze += (vis.get("gaze") or [])
        trainer_reading += (vis.get("trainer_reading") or [])
        cons = vis.get("consistency") or {}
        cons_samples += cons.get("samples") or 0
        if cons.get("same_person_throughout") is not None:
            same_flags.append(bool(cons["same_person_throughout"]))
        if cons.get("trainer_present") is not None:
            trainer_p.append(bool(cons["trainer_present"]))
        if cons.get("candidate_present") is not None:
            cand_p.append(bool(cons["candidate_present"]))
        if cons.get("candidate_identities") is not None:
            cand_ids.append(int(cons["candidate_identities"]))

    for role, d in cam.items():
        spk = d["speaking_sec"]
        cam[role] = {"speaking_sec": round(spk, 1),
                     "on_camera_sec": round(d["on_camera_sec"], 1),
                     "on_camera_pct_of_speaking": round(d["on_camera_sec"] / spk, 3) if spk else None}

    for role, acc in cstate.items():
        for k in ("on_sec", "camera_off_sec", "unassessable_sec", "assessable_sec",
                  "longest_camera_off_sec"):
            acc[k] = round(acc[k], 1)

    return {
        "enabled": True,
        "frames_sampled": int(frames),
        "face_visible_sec": round(face, 1),
        "screen_share_or_noface_sec": round(noface, 1),
        "camera": cam,
        "camera_state": cstate,
        "vision": {
            "gaze": gaze[:12],
            "trainer_reading": trainer_reading[:12],
            "consistency": {
                "samples": cons_samples,
                # one chunk showing a swap means the day had a swap
                "same_person_throughout": (all(same_flags) if same_flags else None),
                "trainer_present": (any(trainer_p) if trainer_p else None),
                "candidate_present": (any(cand_p) if cand_p else None),
                "candidate_identities": (max(cand_ids) if cand_ids else None),
            },
        },
    }


# ── helpers ──────────────────────────────────────────────────────────────────
def _recording_sort_key(kr):
    """Sort chunks by TRUE recording-start order. The S3 key embeds
    {Date}/Time-{h}-{mm}-{AM|PM}-IST/, which is the recording start — unlike
    meeting.analyzed_at, which is analysis wall-clock (recordings can be analysed
    out of order). Falls back to analyzed_at then key when the Time folder is
    absent (non-standard layout)."""
    key, r = kr
    m = _TIME_RE.search(key or "")
    if m:
        date, hh, mm, ap = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4).upper()
        h = (hh % 12) + (12 if ap == "PM" else 0)
        return (0, date, h * 60 + mm, key)
    at = (r.get("meeting") or {}).get("analyzed_at") or ""
    return (1, "", 0, at or key)
