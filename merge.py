"""Session merge — one meeting recorded in several chunks becomes ONE session result.

Zoom keeps the same meeting_id when a host stops/restarts, so one logical session
can land as several Time-*/{meeting_id}/ recordings under the same date. The worker
analyzes each chunk on arrival, then calls merge() over ALL sibling chunk results to
(re)build the session-level file. Idempotent: a late chunk just re-merges.

Strategies (MERGE_STRATEGY env):
  stitch  (default) — the chunks are parts of one session: SUM raw seconds, UNION
                      coverage/flags, recompute every ratio + the scores at the end.
  longest           — treat short extras as false starts: keep the longest chunk's
                      result as the session result (others recorded in `merge.ignored`).

The golden rule (why chunk results store *_sec primitives): never average per-chunk
percentages — sum the raw seconds/sets, then compute ratios ONCE here.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

from analyzer import scoring

_STATUS_RANK = {"covered": 2, "partial": 1, "not_covered": 0}
_DEPTH_RANK = {"adequate": 2, "not_applicable": 1, "under_covered": 0}


def merge(chunks: list[tuple[str, dict]], strategy: str, day_plan: dict) -> dict:
    """chunks: [(s3_key, chunk_result_dict), ...] — at least one. Returns session result."""
    chunks = sorted(chunks, key=_chunk_start)          # stable, oldest first
    if strategy == "longest" or len(chunks) == 1:
        session = _longest(chunks)
    else:
        session = _stitch(chunks, day_plan)
    session["merge"] = {
        "strategy": strategy if len(chunks) > 1 else "single",
        "chunk_count": len(chunks),
        "chunks": [{"key": k,
                    "video_file": (r.get("meeting") or {}).get("video_file"),
                    "duration_sec": (r.get("meeting") or {}).get("duration_sec")}
                   for k, r in chunks],
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }
    if strategy == "longest" and len(chunks) > 1:
        keep = session["meeting"].get("video_file")
        session["merge"]["ignored"] = [c["video_file"] for c in session["merge"]["chunks"]
                                       if c["video_file"] != keep]
    if session["merge"]["strategy"] in ("longest", "single"):
        # deductions carried over from a chunk keep their proof/... paths, which are
        # relative to THAT chunk's meeting folder (see merge.chunks[].key)
        session["merge"]["proof_location"] = "chunk meeting folder (see merge.chunks[].key)"
    return session


# ── keep-longest ─────────────────────────────────────────────────────────────
def _longest(chunks):
    import copy
    _, best = max(chunks, key=lambda kr: (kr[1].get("meeting") or {}).get("duration_sec") or 0)
    return copy.deepcopy(best)


# ── stitch ───────────────────────────────────────────────────────────────────
def _stitch(chunks, day_plan: dict) -> dict:
    results = [r for _, r in chunks]
    longest = max(results, key=lambda r: (r.get("meeting") or {}).get("duration_sec") or 0)

    # meta: sum duration, carry identity fields from the longest chunk
    meta = dict(longest.get("meeting") or {})
    meta["duration_sec"] = round(sum((r.get("meeting") or {}).get("duration_sec") or 0
                                     for r in results), 1)
    meta["video_file"] = f"{len(results)} recordings (merged)"
    meta["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    meta["config_source"] = "session-merge"

    transcript = _merge_transcript(results, longest)
    video = _merge_video(results)

    cfg = SimpleNamespace(day_plan=day_plan, day_number=meta.get("day") or 0)
    session = scoring.assemble(meta, transcript, video, cfg)
    return session


def _merge_transcript(results, longest) -> dict:
    ts = [r.get("transcript") or {} for r in results]

    # talk time: sum raw seconds, ratios recomputed
    talk = {"trainer_sec": 0.0, "candidate_sec": 0.0, "other_sec": 0.0}
    dur = 0.0
    non_en = 0.0
    flagged = []
    roles = {}
    for t in ts:
        m = t.get("metrics") or {}
        dur += m.get("duration_sec") or 0
        tt = m.get("talk_time") or {}
        for k in talk:
            talk[k] += tt.get(k) or 0
        lang = m.get("language") or {}
        non_en += lang.get("non_english_sec") or 0
        flagged += (lang.get("flagged_segments") or [])
        roles.update(m.get("speaker_roles") or {})
    spoken = sum(talk.values())
    talk = {k: round(v, 1) for k, v in talk.items()}
    talk["trainer_ratio"] = round(talk["trainer_sec"] / spoken, 3) if spoken else None
    talk["candidate_ratio"] = round(talk["candidate_sec"] / spoken, 3) if spoken else None

    # coverage: best status per section across chunks
    per_section: dict[str, dict] = {}
    expected = 0
    under_union: set[str] = set()
    iflags, seen = [], set()
    for t in ts:
        cov = t.get("coverage_analysis") or {}
        expected = max(expected, cov.get("expected_section_count") or 0)
        under_union |= set(cov.get("under_covered_key_sections") or [])
        for row in cov.get("coverage") or []:
            if not isinstance(row, dict):
                continue
            name = row.get("section")
            cur = per_section.get(name)
            if cur is None or _row_rank(row) > _row_rank(cur):
                per_section[name] = row
        for f in cov.get("integrity_flags") or []:
            if not isinstance(f, dict):
                continue
            key = (f.get("type"), f.get("evidence"))
            if key not in seen:
                seen.add(key)
                iflags.append(f)

    rows = list(per_section.values())
    expected = expected or len(rows) or 1
    n_cov = sum(1 for r in rows if r.get("status") == "covered")
    n_par = sum(1 for r in rows if r.get("status") == "partial")
    # a section is no longer "under-covered" if any chunk covered it adequately
    under = sorted(s for s in under_union
                   if not (per_section.get(s, {}).get("status") == "covered"
                           and per_section.get(s, {}).get("depth_vs_allotted") == "adequate"))

    coverage_analysis = {
        "coverage": rows,
        "integrity_flags": iflags,
        "coverage_pct": round(100 * (n_cov + 0.5 * n_par) / expected),
        "expected_section_count": expected,
        "under_covered_key_sections": under,
    }

    return {
        "source": "merged",
        "segment_count": sum(t.get("segment_count") or 0 for t in ts),
        "metrics": {
            "duration_sec": round(dur, 1),
            "talk_time": talk,
            "language": {"available": any((t.get("metrics") or {}).get("language", {}).get("available")
                                          for t in ts),
                         "non_english_sec": round(non_en, 1),
                         "flagged_segments": flagged[:25]},
            "speaker_roles": roles,
        },
        "coverage_analysis": coverage_analysis,
        # the "what happened" narrative: keep the longest chunk's description
        "meeting_description": (longest.get("meeting_description")
                                or (longest.get("transcript") or {}).get("meeting_description")
                                or {"available": False}),
    }


def _merge_video(results) -> dict:
    vids = [r.get("video") or {} for r in results]
    if not any(v.get("enabled") for v in vids):
        return {"enabled": False}

    cam = {"trainer": {"speaking_sec": 0.0, "on_camera_sec": 0.0},
           "candidate": {"speaking_sec": 0.0, "on_camera_sec": 0.0}}
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
        vis = v.get("vision") or {}
        gaze += (vis.get("gaze") or [])
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

    return {
        "enabled": True,
        "frames_sampled": int(frames),
        "face_visible_sec": round(face, 1),
        "screen_share_or_noface_sec": round(noface, 1),
        "camera": cam,
        "vision": {
            "gaze": gaze[:12],
            "consistency": {
                "samples": cons_samples,
                # one chunk showing a swap means the session had a swap
                "same_person_throughout": (all(same_flags) if same_flags else None),
                "trainer_present": (any(trainer_p) if trainer_p else None),
                "candidate_present": (any(cand_p) if cand_p else None),
                "candidate_identities": (max(cand_ids) if cand_ids else None),
            },
        },
    }


# ── helpers ──────────────────────────────────────────────────────────────────
def _row_rank(row: dict) -> tuple:
    return (_STATUS_RANK.get(row.get("status"), 0),
            _DEPTH_RANK.get(row.get("depth_vs_allotted"), 0))


def _chunk_start(kr) -> str:
    key, r = kr
    return (r.get("meeting") or {}).get("analyzed_at") or key
