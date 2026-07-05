"""Entrypoint: python -m analyzer

Transcript track  -> deterministic metrics + GPT-4o day-plan coverage.
Video track       -> 100% local models (MediaPipe + InsightFace), CPU only.
"""
import json
import os
import sys
from datetime import datetime, timezone

from .config import load_config, day_sections
from .openai_client import OpenAIClient
from . import transcript as T
from . import video as V
from . import annotate as A
from . import presentation as P
from . import proof as PR
from . import report as R
from . import scoring


def sections_cfg_topic_mismatch(cfg) -> bool:
    """Rough check: does the Salesforce day_step_name share almost no words with the rubric title?"""
    from .config import day_sections
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


def main() -> int:
    cfg = load_config()

    print("=" * 70)
    print(f"Analyzing: {cfg.video_path}")
    print(f"Day {cfg.day_number} | video={cfg.enable_video} | mock_openai={cfg.mock_openai} | config={cfg.config_source}")
    if cfg.day_step_name and sections_cfg_topic_mismatch(cfg):
        print(f"  NOTE: session labelled '{cfg.day_step_name}' but Day {cfg.day_number} rubric is "
              f"'{day_sections(cfg).get('title')}' — check day_plan.json matches your real program.")
    print("=" * 70)

    if not os.path.exists(cfg.video_path):
        print(f"ERROR: video not found at {cfg.video_path}. "
              f"Put your file in ./input and set VIDEO_FILE in .env.")
        return 2

    oa = OpenAIClient(cfg.openai_api_key, cfg.openai_model, cfg.mock_openai,
                      reasoning_effort=cfg.openai_reasoning_effort)
    sections_cfg = day_sections(cfg)

    # ── Transcript track (uses GPT-4o for coverage) ─────────────────────────
    print("\n[1/6] Transcript track...")
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

    # ── Video track (100% local models) ─────────────────────────────────────
    frames, presence = [], []
    if cfg.enable_video:
        print("\n[2/6] Video track (local CPU models)...")
        frames = V.extract_frames(cfg.video_path, cfg.frame_fps, cfg.frames_work_dir)
        video_result, presence = V.run_video(frames, segments, roles, cfg)
        print(f"      face_visible={video_result['face_visible_sec']}s  "
              f"screen_share/noface={video_result['screen_share_or_noface_sec']}s")
    else:
        print("\n[2/6] Video track skipped (ENABLE_VIDEO_ANALYSIS=false)")
        video_result = {"enabled": False}

    # ── Annotated analysis video (what the models saw) ──────────────────────
    if cfg.enable_video and cfg.save_analysis_video and presence:
        print("\n[3/6] Rendering analysis video (what the models saw)...")
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
        print(f"\n[3/6] Analysis video skipped ({reason})")

    # ── Candidate presentation (GPT-4o vision, informational only) ──────────
    if cfg.enable_presentation and cfg.enable_video and presence:
        print("\n[4/6] Candidate presentation check (GPT-4o vision, informational)...")
        try:
            presentation = P.analyze_presentation(frames, presence, segments, roles, oa, cfg)
            print(f"      person_visible={presentation.get('person_visible')}  "
                  f"frames_assessed={len(presentation.get('frames_used', []) or [])}")
        except Exception as exc:  # informational-only step: never kill the run
            print(f"      presentation check failed ({exc}); continuing without it")
            presentation = {"enabled": True, "informational_only": True, "error": str(exc)}
    else:
        reason = ("ENABLE_PRESENTATION_ANALYSIS=false" if not cfg.enable_presentation
                  else "video track disabled" if not cfg.enable_video
                  else "no frames analyzed")
        print(f"\n[4/6] Candidate presentation check skipped ({reason})")
        presentation = {"enabled": False}

    # ── Score + write ───────────────────────────────────────────────────────
    print("\n[5/6] Scoring...")
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
    result = scoring.assemble(meta, transcript_result, video_result, presentation, cfg)

    # ── Proof folder (receipts for every red flag) + HTML report ────────────
    print("\n[6/6] Writing proof folder + report...")
    PR.build(result, cfg, frames)

    # frames were analyzed in the container-local work dir; publish them to the
    # mounted output in ONE bulk copy (after proof has taken its copies)
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


if __name__ == "__main__":
    sys.exit(main())
