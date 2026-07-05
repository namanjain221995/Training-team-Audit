"""Analysis video — watch what the local models saw, frame by frame.

AR-style overlays on the already-analyzed 1-fps frames, encoded into
output/analysis-video.mp4 (playable in any browser, embedded in report.html):

  header bar       time | who is speaking (name from the VTT) | face count
  corner brackets  MediaPipe face detection, colour-coded by identity
  keypoint dots    the 6 facial keypoints MediaPipe detects (eyes/nose/mouth/ears)
  name chip        who the face is — InsightFace samples identity every
                   IDENTITY_SAMPLE_SEC; the latest match is carried forward so
                   EVERY frame shows a name, with the "checked @mm:ss" time
  cyan banner      gaze-sample frames (the reading-tell signal) with the score
  red banner       frames with no face (screen share / camera off)

Colours: orange = trainer, cyan = candidate, green = face not yet identified.

No model is re-run: everything is drawn from data captured during the analysis.
Rendering happens in a container-local work dir; the finished mp4 is copied to
the mounted output in one write (see the Windows bind-mount gotcha in CLAUDE.md).
"""
import os
import shutil
import subprocess

PLAYBACK_FPS = 4          # 1 frame per session-second, played at 4x
JPEG_QUALITY = 82
HEADER_H = 36
FONT = 0                  # cv2.FONT_HERSHEY_SIMPLEX

# BGR
_TRAINER = (60, 160, 255)    # orange
_CANDIDATE = (230, 216, 60)  # cyan
_NEUTRAL = (110, 220, 110)   # green
_CYAN = (220, 200, 40)
_RED = (60, 60, 230)
_WHITE = (255, 255, 255)


def build_analysis_video(presence: list[dict], video_result: dict, segments: list[dict],
                         roles: dict, out_path: str, names: dict | None = None,
                         work_dir: str = "/tmp/annotated") -> str | None:
    """Draw overlays on every analyzed frame and encode an mp4. Returns out_path or None."""
    import cv2

    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    names = names or {}

    vision = video_result.get("vision") or {}
    cons = vision.get("consistency") or {}
    id_samples = sorted((s for s in (cons.get("samples_detail") or [])
                         if isinstance(s, dict) and s.get("frame")),
                        key=lambda s: s.get("time_sec", 0))
    sample_frames = {s["frame"] for s in id_samples}
    gaze_map = {}
    for g in (vision.get("gaze") or []):
        if isinstance(g, dict):
            for fname in (g.get("evidence_frames") or []):
                gaze_map[fname] = g

    role_lookup = _RoleLookup(segments, roles)

    written, id_ptr = 0, 0
    last_id = id_samples[0] if id_samples else None   # pre-fill before first sample
    for p in presence:
        t = p.get("time_sec", 0)
        name = os.path.basename(p["path"])

        # identity state updates FIRST — they must happen even if this frame's
        # image can't be read, or a stale label would carry across a person switch
        while id_ptr < len(id_samples) and id_samples[id_ptr]["time_sec"] <= t:
            last_id = id_samples[id_ptr]
            id_ptr += 1
        # the on-screen person likely just changed; drop the stale label unless
        # this very frame was re-identified (track breaks are sampled, so it
        # normally is — this only fires when the face was too small/blurred)
        if p.get("track_break") and name not in sample_frames:
            last_id = None

        img = cv2.imread(p["path"])
        if img is None:
            continue
        h, w = img.shape[:2]

        # ── header bar ──
        cv2.rectangle(img, (0, 0), (w, HEADER_H), (18, 18, 18), -1)
        role = role_lookup.at(t)
        speaking = f"{role} ({names.get(role)})" if role and names.get(role) else (role or "(silence)")
        head = f"{_ts(t)}  |  speaking: {speaking}  |  faces: {p.get('faces', 0)}"
        cv2.putText(img, head, (10, 24), FONT, 0.6, _WHITE, 1, cv2.LINE_AA)

        # ── identity chip text + colour (from the carried InsightFace sample) ──
        chip_text, color = _identity_chip(last_id, cons, names)
        if not p.get("face_present"):
            cv2.putText(img, "NO FACE - screen share / camera off", (10, HEADER_H + 30),
                        FONT, 0.7, _RED, 2, cv2.LINE_AA)
        else:
            boxes = p.get("face_boxes") or []
            points = p.get("face_points") or []
            for i, bb in enumerate(boxes):
                x, y = int(bb[0] * w), int(bb[1] * h)
                bw, bh = int(bb[2] * w), int(bb[3] * h)
                _brackets(cv2, img, x, y, bw, bh, color)
                for kp in (points[i] if i < len(points) else []):
                    kx, ky = int(kp[0] * w), int(kp[1] * h)
                    cv2.circle(img, (kx, ky), 3, color, -1)
                    cv2.circle(img, (kx, ky), 6, color, 1)
                if chip_text:
                    _chip(cv2, img, x, min(y + bh + 10, h - 34), chip_text, color)
            if not boxes and chip_text:   # face present but box list empty (shouldn't happen)
                _chip(cv2, img, 10, h - 60, chip_text, color)

        # this exact frame was an identity-check sample
        if name in sample_frames:
            _chip(cv2, img, w - 190, HEADER_H + 10, "identity check: NOW", color)

        # gaze sample frame
        g = gaze_map.get(name)
        if g:
            cv2.putText(img,
                        f"gaze sample (answer @{g.get('answer_time')}): "
                        f"reading likelihood {g.get('reading_likelihood')}",
                        (10, h - 12), FONT, 0.6, _CYAN, 2, cv2.LINE_AA)

        # only consume a sequence number on a successful write: a gap in
        # ann_%06d.jpg makes ffmpeg silently truncate the video (and exit 0)
        ok = cv2.imwrite(os.path.join(work_dir, f"ann_{written + 1:06d}.jpg"), img,
                         [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            print(f"warn: could not write annotated frame for {name}; skipping")
            continue
        written += 1

    if not written:
        shutil.rmtree(work_dir, ignore_errors=True)
        return None

    tmp_out = os.path.join(work_dir, "analysis.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(PLAYBACK_FPS),
           "-i", os.path.join(work_dir, "ann_%06d.jpg"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # h264 needs even dimensions
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
           tmp_out]
    subprocess.run(cmd, check=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.copy2(tmp_out, out_path)   # ONE big write to the bind mount
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Analysis video: {written} frames at {PLAYBACK_FPS} fps -> {out_path}")
    return out_path


# ── drawing helpers ──────────────────────────────────────────────────────────
def _brackets(cv2, img, x, y, bw, bh, color, thick=2):
    """AR-style corner brackets instead of a full rectangle."""
    L = max(14, int(0.22 * min(bw, bh)))
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + bw, y, -1, 1),
                           (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)


def _chip(cv2, img, x, y, text, color):
    """Translucent rounded-feel name tag."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    pad = 7
    x = max(0, min(x, w - tw - 2 * pad - 1))
    y = max(HEADER_H, min(y, h - th - 2 * pad - 1))
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad
    roi = img[y:y2, x:x2]
    dark = roi.copy()
    dark[:] = (22, 22, 22)
    cv2.addWeighted(dark, 0.68, roi, 0.32, 0, roi)
    cv2.rectangle(img, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x + pad, y2 - pad), FONT, 0.55, color, 1, cv2.LINE_AA)


def _identity_chip(last_id: dict | None, cons: dict, names: dict) -> tuple[str, tuple]:
    """Label + colour for the carried identity sample."""
    if not last_id:
        return "", _NEUTRAL
    label = last_id.get("label", "")
    at = _ts(last_id.get("time_sec", 0))
    sim = last_id.get("sim")
    match = f"match {int(sim * 100)}%, " if isinstance(sim, (int, float)) else ""
    if label == "trainer":
        who = names.get("trainer") or "trainer"
        return f"TRAINER  {who}  ({match}@{at})", _TRAINER
    # single non-trainer identity + a trainer reference present -> the candidate
    if (label == "person-1" and cons.get("trainer_present")
            and cons.get("candidate_identities") == 1):
        who = names.get("candidate") or "candidate"
        return f"CANDIDATE  {who}  (@{at})", _CANDIDATE
    return f"{label.upper()}  (@{at})", _CANDIDATE


class _RoleLookup:
    """Who is speaking at time t — forward-scanning pointer over sorted segments."""

    def __init__(self, segments: list[dict], roles: dict):
        self._segs = sorted((s for s in segments if s.get("speaker")),
                            key=lambda s: s["start"])
        self._roles = roles or {}
        self._i = 0

    def at(self, t: float) -> str | None:
        while self._i < len(self._segs) and self._segs[self._i]["end"] < t:
            self._i += 1
        for s in self._segs[self._i:]:
            if s["start"] > t:
                break
            if s["start"] <= t <= s["end"]:
                return self._roles.get(s["speaker"], "other")
        return None


def _ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"
