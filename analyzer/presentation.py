"""Candidate presentation check (GPT-4o vision) — descriptive, for human review.

Assesses ONLY the controllable presentation factors the program itself trains
(Day 7 'Interview Environment Setup': camera at eye level, neutral background,
professional attire, upright posture, adequate lighting).

Deliberate boundaries:
  - Descriptive observations, not judgments of the person. The prompt forbids
    comments on physical characteristics (body shape, skin, age, ethnicity,
    attractiveness).
  - Informational only: it does NOT feed the integrity score. It exists so a
    human reviewer / trainer can coach the candidate on presentation.

Frame selection is local (no AI): frames where the CANDIDATE is speaking and a
face is on screen, spread across the session. One GPT-4o vision call total.
"""
import os

MAX_FRAMES = 4

PRESENTATION_SYSTEM = (
    "You are a quality-control assistant for an interview-readiness training program. "
    "You are shown a few frames of the CANDIDATE from a recorded Zoom training session. "
    "Describe, objectively and neutrally, only the controllable presentation factors the "
    "program coaches: attire formality, grooming/hair tidiness, posture and body language, "
    "background, camera framing, and lighting. "
    "STRICT RULES: never comment on body shape, weight, skin, age, ethnicity, gender, or "
    "attractiveness. Judge only what the candidate can change for an interview. "
    "If the person is not clearly visible in the frames, say so and leave fields null. "
    "Return ONLY a JSON object."
)

PRESENTATION_PROMPT = (
    "These frames show the candidate during a training session (frames may also show a "
    "screen-share or another participant — assess only frames showing a person).\n"
    "Respond with JSON of exactly this shape (use null where you cannot tell):\n"
    "{\n"
    '  "person_visible": true|false (false if no person is clearly visible in any frame),\n'
    '  "attire": {"observation": "<what they are wearing, e.g. collared shirt>", "interview_appropriate": true|false|null},\n'
    '  "grooming_hair": {"observation": "<tidy/neat vs unkempt for an interview>", "tidy_for_interview": true|false|null},\n'
    '  "posture_body_language": {"observation": "<upright/engaged vs slouched/distracted>", "upright_and_engaged": true|false|null},\n'
    '  "background": {"observation": "<neutral wall / cluttered room / virtual bg / other people visible>", "neutral_and_professional": true|false|null},\n'
    '  "camera_setup": {"observation": "<eye level and framed vs too low/high/cut off>", "eye_level_and_framed": true|false|null},\n'
    '  "lighting": {"observation": "<face clearly lit vs dark/backlit>", "adequate": true|false|null},\n'
    '  "overall_notes": "<1-2 sentence neutral summary>",\n'
    '  "coaching_suggestions": ["<concrete, controllable improvement>", "..."]\n'
    "}"
)


def analyze_presentation(frames: list[dict], presence: list[dict], segments: list[dict],
                         roles: dict, oa, cfg) -> dict:
    """One vision call on candidate-speaking frames. Returns an informational block."""
    picked = _pick_candidate_frames(frames, presence, segments, roles)
    if not picked:
        return {"enabled": True, "informational_only": True, "person_visible": False,
                "note": "no candidate face-on-screen frames found to assess"}

    result = oa.vision_json(PRESENTATION_SYSTEM, PRESENTATION_PROMPT,
                            [f["path"] for f in picked])
    if not isinstance(result, dict):
        result = {"_parse_error": True}
    result["enabled"] = True
    result["informational_only"] = True   # never feeds the integrity score
    result["frames_used"] = [os.path.basename(f["path"]) for f in picked]
    result["frame_times"] = [_sec_to_ts(f["time_sec"]) for f in picked]
    if not roles:
        result["note"] = ("no speaker labels in transcript; frames sampled across the whole "
                          "session and may show the trainer instead of the candidate")
    return result


def _pick_candidate_frames(frames: list[dict], presence: list[dict], segments: list[dict],
                           roles: dict, k: int = MAX_FRAMES) -> list[dict]:
    """Frames where the candidate is speaking AND a face is on screen, spread across the session."""
    from .video import _candidate_answers, _frames_in_window
    face_frames = [p for p in presence if p.get("face_present")]
    if not face_frames:
        return []

    answers = _candidate_answers(segments, roles) if roles else []
    if answers:
        # take frames from the longest candidate answers until we have k, spread within each
        picked, seen = [], set()
        for ans in answers:
            for f in _frames_in_window(face_frames, ans["start"], ans["end"], k=2):
                if f["path"] not in seen:
                    picked.append(f)
                    seen.add(f["path"])
                if len(picked) >= k:
                    return sorted(picked, key=lambda f: f["time_sec"])
        if picked:
            return sorted(picked, key=lambda f: f["time_sec"])

    # fallback (no roles / no candidate turns): spread across all face frames
    if len(face_frames) <= k:
        return face_frames
    step = len(face_frames) / k
    return [face_frames[int(i * step)] for i in range(k)]


def _sec_to_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"
