"""Programmatic API — run the full analysis from Python (used by the EC2 worker).

Two entry points:

  build_config(...)  -> Config pointing at arbitrary local paths (no /data mounts)
  run(cfg)           -> executes the whole pipeline, writes result.json/report/proof
                        into cfg's output dir, returns 0 on success

`python -m analyzer` (docker/local mode) is now a thin wrapper: load_config() -> run().

One model analyzes the transcript (LLM_PROVIDERS, default techsara = techsara-35b).
A comma list is a FALLBACK order: the next model runs only if the previous one
fails. Output is always ONE result.json + report.html + proof/ (what the session
merge and Salesforce read).
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import Config, day_sections, model_settings, _apply_training_temp, _humanize, _bool, _float, _int
from .openai_client import make_clients
from . import transcript as T
from . import video as V
from . import annotate as A
from . import proof as PR
from . import report as R
from . import scoring

_REPO_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

# Bump when the result.json / report schema meaningfully changes, so the worker
# backfill can tell an old report from a current one and reprocess only the stale.
# 3 = trainer effectiveness + candidate profile + executive summary + attendance
#     + topics_missed_full (-3) + weekly report.
SCHEMA_VERSION = 3


def build_config(
    video_path: str,
    output_dir: str,
    transcript_path: str | None = None,
    trainer_image_path: str | None = None,
    training_temp_path: str | None = None,
    *,
    day_number: int | None = None,
    trainer_name: str = "",
    candidate_name: str = "",
    meeting_id: str | None = None,
    day_step_name: str | None = None,
    s3_prefix: str | None = None,
    s3_bucket: str | None = None,
    session_type: str | None = None,
    person_change_grace_sec: int | None = None,
    session_start_window_sec: int | None = None,
    config_dir: str | None = None,
    frames_work_dir: str | None = None,
    save_frames: bool = False,
    model_cache_dir: str | None = None,
    openai_api_key: str | None = None,
) -> Config:
    """Build a Config from explicit paths. Env vars fill anything not passed.

    If training_temp_path exists, day/trainer/candidate/meeting_id are read FROM IT
    (production parity with the SQS job) — same behavior as docker mode.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    settings = model_settings()
    if openai_api_key is not None:
        settings["openai_api_key"] = openai_api_key
    cfg = Config(
        **settings,

        video_path=os.path.abspath(video_path),
        transcript_path=os.path.abspath(transcript_path) if transcript_path else None,
        trainer_image_path=os.path.abspath(trainer_image_path) if trainer_image_path else None,
        training_temp_path=os.path.abspath(training_temp_path) if training_temp_path else None,

        day_number=day_number if day_number is not None else _int("DAY_NUMBER", 1),
        trainer_name=_humanize(trainer_name) if trainer_name else "",
        candidate_name=_humanize(candidate_name) if candidate_name else "",
        meeting_id=meeting_id,
        day_step_name=day_step_name,
        s3_prefix=s3_prefix,
        s3_bucket=s3_bucket,
        config_source="worker-job",

        enable_video=_bool("ENABLE_VIDEO_ANALYSIS", True),
        frame_fps=_float("FRAME_FPS", 1.0),
        whisper_model=os.environ.get("WHISPER_MODEL", "base").strip(),
        save_frames=save_frames,
        identity_sample_sec=_int("IDENTITY_SAMPLE_SEC", 20),
        session_type=(session_type or os.environ.get("SESSION_TYPE", "one_on_one").strip().lower() or "one_on_one"),
        person_change_grace_sec=(person_change_grace_sec if person_change_grace_sec is not None
                                 else _int("PERSON_CHANGE_GRACE_SEC", 120)),
        session_start_window_sec=(session_start_window_sec if session_start_window_sec is not None
                                  else _int("SESSION_START_WINDOW_SEC", 300)),
        save_analysis_video=_bool("SAVE_ANALYSIS_VIDEO", True),
        model_cache_dir=(model_cache_dir
                         or os.environ.get("MODEL_CACHE_DIR", "").strip()
                         or os.path.expanduser("~/.cache/training-analysis")),

        output_path=os.path.join(output_dir, "result.json"),
        frames_dir=os.path.join(output_dir, "frames"),
        proof_dir=os.path.join(output_dir, "proof"),
        frames_work_dir=frames_work_dir or os.path.join(output_dir, "_frames_work"),
    )

    _apply_training_temp(cfg)   # temp file (if present) overrides names/day/meeting_id

    cdir = config_dir or os.environ.get("CONFIG_DIR", "").strip() or _REPO_CONFIG_DIR
    with open(os.path.join(cdir, "day_plan.json"), encoding="utf-8") as fh:
        cfg.day_plan = json.load(fh)
    return cfg


def sections_cfg_topic_mismatch(cfg) -> bool:
    """Rough check: does the Salesforce day_step_name share almost no words with the rubric title?"""
    title = (day_sections(cfg).get("title") or "").lower()
    label = (cfg.day_step_name or "").lower()
    if not title or not label:
        return False
    stop = {"day", "and", "the", "of", "during", "a", "for", "-", "to", "in"}
    tw = {w for w in title.replace("-", " ").split() if w not in stop and len(w) > 2}
    lw = {w for w in label.replace("-", " ").split() if w not in stop and len(w) > 2}
    if not tw or not lw:
        return False
    return len(tw & lw) == 0


def run(cfg: Config) -> int:
    """Execute the full pipeline. Writes result.json / report.html / proof/ into the
    output dir. Returns 0 on success, 2 on missing video."""
    print("=" * 70)
    print(f"Analyzing: {cfg.video_path}")
    print(f"Day {cfg.day_number} | video={cfg.enable_video} | mock_openai={cfg.mock_openai} | "
          f"config={cfg.config_source} | models={','.join(cfg.llm_providers)} | "
          f"techsara_whisper={cfg.techsara_whisper}")
    if cfg.day_step_name and sections_cfg_topic_mismatch(cfg):
        print(f"  NOTE: session labelled '{cfg.day_step_name}' but Day {cfg.day_number} rubric is "
              f"'{day_sections(cfg).get('title')}' — check day_plan.json matches your real program.")
    print("=" * 70)

    if not os.path.exists(cfg.video_path):
        print(f"ERROR: video not found at {cfg.video_path}.")
        return 2

    clients = make_clients(cfg)          # fails fast on a missing key
    sections_cfg = day_sections(cfg)
    out_dir = os.path.dirname(cfg.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # ── [1/5] Transcript track ──────────────────────────────────────────────
    print("\n[1/5] Transcript track...")
    segments, source, tinfo = T.load_transcript(
        cfg, out_dir=out_dir,
        work_dir=os.path.join(os.path.dirname(os.path.abspath(cfg.frames_work_dir)), "_whisper"))
    metrics = T.compute_metrics(segments, cfg)
    roles = metrics.get("speaker_roles", {})
    print(f"      source={source}  segments={len(segments)}  duration={metrics['duration_sec']}s")

    # the transcript analysis runs in the background (network-bound) while the
    # video track below runs on the local CPU. LLM_PROVIDERS is a FALLBACK order:
    # the first model is used; the next is tried only if it fails.
    failures: list = []

    def analyze(oa):
        t0 = time.time()
        # topic analysis runs FIRST — coverage is anchored on it so the two views
        # can't contradict each other (same mock-Q&A counted in one, ignored in the other)
        description = T.describe_meeting(segments, oa, session_label=cfg.day_step_name,
                                         sections_cfg=sections_cfg)
        coverage = T.analyze_coverage(segments, sections_cfg, oa, session_label=cfg.day_step_name,
                                      topics=description.get("topics"))
        print(f"      [{oa.model}] coverage={coverage.get('coverage_pct')}%  "
              f"integrity_flags={len(coverage.get('integrity_flags', []) or [])}  "
              f"description={'ok' if description.get('available') else 'unavailable'}  "
              f"({time.time() - t0:.0f}s)")
        return {"description": description, "coverage": coverage,
                "elapsed_sec": round(time.time() - t0, 1)}

    def analyze_first_working():
        for oa in clients:
            try:
                return oa, analyze(oa)
            except Exception as exc:
                print(f"      [{oa.model}] FAILED: {exc!r}")
                failures.append({"provider": oa.provider, "model": oa.model, "error": repr(exc)[:500]})
        raise RuntimeError(f"every model failed: {failures}")

    pool = ThreadPoolExecutor(max_workers=1)
    analysis = pool.submit(analyze_first_working)
    print(f"      transcript analysis started on: {clients[0].model}"
          + (f" (fallback: {', '.join(oa.model for oa in clients[1:])})" if len(clients) > 1 else ""))

    # ── [2/5] Video track ───────────────────────────────────────────────────
    frames, presence = [], []
    if cfg.enable_video:
        print("\n[2/5] Video track (local CPU models)...")
        frames = V.extract_frames(cfg.video_path, cfg.frame_fps, cfg.frames_work_dir)
        video_result, presence = V.run_video(frames, segments, roles, cfg)
        print(f"      face_visible={video_result['face_visible_sec']}s  "
              f"screen_share/noface={video_result['screen_share_or_noface_sec']}s")
    else:
        print("\n[2/5] Video track skipped (ENABLE_VIDEO_ANALYSIS=false)")
        video_result = {"enabled": False}

    # ── [3/5] Annotated analysis video (what the models saw) ────────────────
    if cfg.enable_video and cfg.save_analysis_video and presence:
        print("\n[3/5] Rendering analysis video (what the models saw)...")
        try:
            out = A.build_analysis_video(
                presence, video_result, segments, roles,
                os.path.join(out_dir, "analysis-video.mp4"),
                names={"trainer": cfg.trainer_name, "candidate": cfg.candidate_name})
            if out:
                video_result["analysis_video"] = os.path.basename(out)
        except Exception as exc:  # visualization only: never kill the run
            print(f"      analysis video failed ({exc}); continuing without it")
    else:
        reason = ("video track disabled" if not cfg.enable_video
                  else "SAVE_ANALYSIS_VIDEO=false" if not cfg.save_analysis_video
                  else "no frames analyzed")
        print(f"\n[3/5] Analysis video skipped ({reason})")

    # ── [4/5] Score ─────────────────────────────────────────────────────────
    print("\n[4/5] Waiting for transcript analysis, then scoring...")
    try:
        oa, an = analysis.result()
    finally:
        pool.shutdown()

    meta = {
        "video_file": os.path.basename(cfg.video_path),
        "meeting_id": cfg.meeting_id,
        "day": cfg.day_number,
        "day_title": sections_cfg.get("title"),       # from day_plan.json rubric
        "day_step_name": cfg.day_step_name,            # from Salesforce/temp (actual session label)
        "trainer_name": cfg.trainer_name or None,
        "candidate_name": cfg.candidate_name or None,
        "s3_prefix": cfg.s3_prefix,
        "s3_bucket": cfg.s3_bucket,
        "duration_sec": metrics["duration_sec"],
        "config_source": cfg.config_source,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,   # used by the worker backfill to skip up-to-date reports
        # trainer attendance / lateness (from training-temp.json; None on local runs)
        "scheduled_start": cfg.scheduled_start,
        "scheduled_duration_min": cfg.scheduled_duration_min,
        "trainer_join": cfg.trainer_join,
        "trainer_late_min": cfg.trainer_late_min,
        "host_email": cfg.host_email,
        "participants": cfg.participants_timing,
        "transcript_source": source,
        "llm_failures": failures,                      # models tried before this one
        "llm": {
            "provider": oa.provider, "model": oa.model,
            "reasoning_effort": oa.reasoning_effort or None,
            # fields this server refused (e.g. reasoning_effort on techsara today)
            "fields_not_supported": sorted(oa._dropped),
            # model calls made, and how many were correction requests (bad JSON /
            # missing sections / too-coarse topics sent back to the model to fix)
            "calls": oa.stats["calls"], "corrections": oa.stats["corrections"],
            "elapsed_sec": an["elapsed_sec"]},
    }
    transcript_result = {
        "source": source,
        "segment_count": len(segments),
        "metrics": metrics,
        "coverage_analysis": an["coverage"],
        "meeting_description": an["description"],
        **({"sources": tinfo} if tinfo else {}),
    }
    result = scoring.assemble(meta, transcript_result, video_result, cfg)

    # ── [5/5] Proof folder + report + write ─────────────────────────────────
    print("\n[5/5] Writing proof folder + report...")
    PR.build(result, cfg, frames)

    # frames were analyzed in a local work dir; publish them to the output dir
    # in ONE bulk copy (after proof has taken its copies)
    if cfg.enable_video and frames:
        import shutil
        saved = False
        if cfg.save_frames:
            shutil.rmtree(cfg.frames_dir, ignore_errors=True)
            try:
                shutil.copytree(cfg.frames_work_dir, cfg.frames_dir)
                saved = True
            except Exception as exc:
                print(f"      could not copy frames to output ({exc}); "
                      f"evidence images are still in proof/")
        if not saved:
            result["video"].pop("evidence_frames_dir", None)
            result["video"]["evidence_frames_saved"] = False

    with open(cfg.output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    written = ["result.json"]
    try:
        R.write(result, os.path.join(out_dir, "report.html"))
        written.append("report.html")
    except Exception as exc:  # the report must never kill a finished analysis
        print(f"      report.html failed ({exc}); the json is unaffected")

    sc = result["scoring"]
    print("\n" + "=" * 70)
    print(f"RESULT  ->  {cfg.output_path}   (model: {oa.model})")
    print(f"  coverage {sc['trainer_coverage_score']}%  |  integrity "
          f"{sc['session_integrity_score']}/100 ({sc['tier']})  |  {an['elapsed_sec']}s")
    for d in sc["deductions"]:
        print(f"    {d['points']:>4}  {d['reason']}  ({d.get('evidence')})")
    for f in failures:
        print(f"  [{f['model']}] FAILED (fell back): {f['error'][:200]}")
    print(f"  files            : {', '.join(written + (tinfo.get('files') or []))}")
    print("=" * 70)
    return 0

