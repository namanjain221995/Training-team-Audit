"""Programmatic API — run the full analysis from Python (used by the EC2 worker).

Two entry points:

  build_config(...)  -> Config pointing at arbitrary local paths (no /data mounts)
  run(cfg)           -> executes the whole pipeline, writes result.json/report/proof
                        into cfg's output dir, returns 0 on success

`python -m analyzer` (docker/local mode) is now a thin wrapper: load_config() -> run().
"""
import json
import os
import sys
from datetime import datetime, timezone

from .config import Config, day_sections, _apply_training_temp, _humanize, _bool, _float, _int
from .openai_client import OpenAIClient
from . import transcript as T
from . import video as V
from . import annotate as A
from . import proof as PR
from . import report as R
from . import scoring

_REPO_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


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

    cfg = Config(
        openai_api_key=(openai_api_key if openai_api_key is not None
                        else os.environ.get("OPENAI_API_KEY", "").strip()),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o").strip(),
        openai_reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "").strip(),
        mock_openai=_bool("MOCK_OPENAI", False),

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
    print(f"Day {cfg.day_number} | video={cfg.enable_video} | mock_openai={cfg.mock_openai} | config={cfg.config_source}")
    if cfg.day_step_name and sections_cfg_topic_mismatch(cfg):
        print(f"  NOTE: session labelled '{cfg.day_step_name}' but Day {cfg.day_number} rubric is "
              f"'{day_sections(cfg).get('title')}' — check day_plan.json matches your real program.")
    print("=" * 70)

    if not os.path.exists(cfg.video_path):
        print(f"ERROR: video not found at {cfg.video_path}.")
        return 2

    oa = OpenAIClient(cfg.openai_api_key, cfg.openai_model, cfg.mock_openai,
                      reasoning_effort=cfg.openai_reasoning_effort)
    sections_cfg = day_sections(cfg)

    # ── [1/5] Transcript track ──────────────────────────────────────────────
    print("\n[1/5] Transcript track...")
    segments, source = T.load_transcript(cfg)
    metrics = T.compute_metrics(segments, cfg)
    roles = metrics.get("speaker_roles", {})
    # topic analysis runs FIRST — coverage is anchored on it so the two views
    # can't contradict each other (same mock-Q&A counted in one, ignored in the other)
    description = T.describe_meeting(segments, oa, session_label=cfg.day_step_name,
                                     sections_cfg=sections_cfg)
    coverage = T.analyze_coverage(segments, sections_cfg, oa, session_label=cfg.day_step_name,
                                  topics=description.get("topics"))
    transcript_result = {
        "source": source,
        "segment_count": len(segments),
        "metrics": metrics,
        "coverage_analysis": coverage,
        "meeting_description": description,
    }
    print(f"      duration={metrics['duration_sec']}s  coverage={coverage.get('coverage_pct')}%  "
          f"integrity_flags={len(coverage.get('integrity_flags', []) or [])}  "
          f"description={'ok' if description.get('available') else 'unavailable'}")

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
                os.path.join(os.path.dirname(cfg.output_path), "analysis-video.mp4"),
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
    print("\n[4/5] Scoring...")
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
    }
    result = scoring.assemble(meta, transcript_result, video_result, cfg)

    # ── [5/5] Proof folder + report + write ─────────────────────────────────
    print("\n[5/5] Writing proof folder + report...")
    PR.build(result, cfg, frames)

    # frames were analyzed in a local work dir; publish them to the output dir
    # in ONE bulk copy (after proof has taken its copies)
    if cfg.enable_video and frames:
        import shutil
        if cfg.save_frames:
            shutil.rmtree(cfg.frames_dir, ignore_errors=True)
            try:
                shutil.copytree(cfg.frames_work_dir, cfg.frames_dir)
            except Exception as exc:
                print(f"      could not copy frames to output ({exc}); "
                      f"evidence images are still in proof/")
                result["video"].pop("evidence_frames_dir", None)
                result["video"]["evidence_frames_saved"] = False
        else:
            result["video"].pop("evidence_frames_dir", None)
            result["video"]["evidence_frames_saved"] = False

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    # human-friendly report next to result.json (opens in any browser)
    report_path = os.path.join(os.path.dirname(cfg.output_path), "report.html")
    try:
        R.write(result, report_path)
    except Exception as exc:  # the report must never kill a finished analysis
        print(f"      report generation failed ({exc}); result.json is unaffected")
        report_path = None

    sc = result["scoring"]
    print("\n" + "=" * 70)
    print(f"RESULT  ->  {cfg.output_path}")
    print(f"  trainer coverage : {sc['trainer_coverage_score']}%")
    print(f"  session integrity: {sc['session_integrity_score']}/100  ({sc['tier']})")
    for d in sc["deductions"]:
        print(f"    {d['points']:>4}  {d['reason']}  ({d.get('evidence')})")
    print(f"  proof            : {cfg.proof_dir}")
    if report_path:
        print(f"  report           : {report_path}  (open in a browser)")
    print("=" * 70)
    return 0
