"""Video track — 100% local, open-source, CPU-only. No GPU, no cloud vision.

Models (downloaded once on first run, cached under /data/.cache):
  - MediaPipe Face Detection : dense per-frame presence (face vs screen-share)   [bundled, no download]
  - MediaPipe Face Mesh      : iris + head pose -> gaze / reading tell            [bundled, no download]
  - InsightFace buffalo_l    : ArcFace embeddings -> identity, trainer-match, swap [~300MB, downloaded once]

Cost control: MediaPipe (fast) runs on every sampled frame; InsightFace (heavier)
runs only on frames sampled every IDENTITY_SAMPLE_SEC seconds.
"""
import os
import subprocess

import numpy as np


# ── 1. Frames ───────────────────────────────────────────────────────────────
def extract_frames(video_path: str, fps: float, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%06d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", video_path, "-vf", f"fps={fps}", "-q:v", "3", pattern]
    subprocess.run(cmd, check=True)
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    frames = [{"index": i, "time_sec": i / fps, "path": os.path.join(out_dir, f)}
              for i, f in enumerate(files)]
    print(f"Extracted {len(frames)} frames at {fps} fps")
    return frames


# ── 2. Presence (MediaPipe Face Detection, dense) ───────────────────────────
# Zoom records the ACTIVE-SPEAKER view. During screen-share the speaker's camera
# shrinks to a thumbnail tile in the top-right corner (confirmed on real frames:
# ~48% of "no full-frame face" frames actually have a detectable face in that
# corner). So when the full frame has no face we (a) retry inside the corner tile
# to recover the speaker's face, then (b) if still none, decide whether the frame
# is a BLANK/dark "camera off" tile (camera_state="camera_off") or live screen
# content with no visible camera (camera_state="screen_only" — NOT assessable, so
# it must never be scored as camera-off). A frame with a face is "on".
_CORNER = (0.66, 0.0, 1.0, 0.36)   # (x0,y0,x1,y1) as fractions: top-right tile region
_CORNER_UPSCALE = 3                 # the tile is small; upscale before re-detecting


def _crop_fractions(img, box):
    h, w = img.shape[:2]
    x0, y0 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
    x1, y1 = min(w, int(box[2] * w)), min(h, int(box[3] * h))
    return img[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else None


def _region_features(region) -> dict:
    """Local appearance features used to tell a blank/dark 'camera off' tile from
    live screen-share content. Thresholds calibrated on real session frames:
    a camera-off tile has almost no edges, one colour dominates, little white, and
    low pixel variance; screen-share content fails all of these."""
    import cv2
    if region is None or region.size == 0:
        return {"edge": 0.0, "dom_color": 1.0, "near_white": 0.0, "gray_std": 0.0}
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge = float(np.count_nonzero(edges)) / edges.size
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    dom = float(hist.max()) / float(hist.sum() or 1)
    near_white = float(np.count_nonzero(gray >= 235)) / gray.size
    return {"edge": edge, "dom_color": dom, "near_white": near_white,
            "gray_std": float(np.std(gray))}


def _looks_blank_tile(f: dict) -> bool:
    return (f["edge"] < 0.02 and f["dom_color"] > 0.75
            and f["near_white"] < 0.20 and f["gray_std"] < 25)


def detect_presence(frames: list[dict]) -> list[dict]:
    import cv2
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    # second, more sensitive detector just for the small upscaled corner tile
    fd_corner = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    out = []
    recovered = cam_off = screen_only = 0
    cx0, cy0, cx1, cy1 = _CORNER
    for fr in frames:
        img = cv2.imread(fr["path"])
        n, boxes, points = 0, [], []
        face_source, camera_state = None, "screen_only"
        if img is not None:
            res = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.detections:
                n = len(res.detections)
                for det in res.detections:  # relative coords, kept for the analysis video
                    bb = det.location_data.relative_bounding_box
                    boxes.append([round(bb.xmin, 4), round(bb.ymin, 4),
                                  round(bb.width, 4), round(bb.height, 4)])
                    # 6 facial keypoints (eyes, nose, mouth, ears) — free from the detector
                    points.append([[round(k.x, 4), round(k.y, 4)]
                                   for k in det.location_data.relative_keypoints])
                face_source, camera_state = "full", "on"
            else:
                # no full-frame face: try to recover the active-speaker corner tile
                crop = _crop_fractions(img, _CORNER)
                best = None
                if crop is not None:
                    up = cv2.resize(crop, None, fx=_CORNER_UPSCALE, fy=_CORNER_UPSCALE,
                                    interpolation=cv2.INTER_CUBIC)
                    cres = fd_corner.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
                    if cres.detections:
                        best = max(cres.detections, key=lambda d: (d.location_data.relative_bounding_box.width
                                                                   * d.location_data.relative_bounding_box.height))
                if best is not None:
                    rb = best.location_data.relative_bounding_box
                    # map the corner-relative box back to full-frame relative coords
                    boxes.append([round(cx0 + rb.xmin * (cx1 - cx0), 4),
                                  round(cy0 + rb.ymin * (cy1 - cy0), 4),
                                  round(rb.width * (cx1 - cx0), 4),
                                  round(rb.height * (cy1 - cy0), 4)])
                    points.append([[round(cx0 + k.x * (cx1 - cx0), 4),
                                    round(cy0 + k.y * (cy1 - cy0), 4)]
                                   for k in best.location_data.relative_keypoints])
                    n, face_source, camera_state = 1, "corner", "on"
                    recovered += 1
                else:
                    # no face anywhere: blank camera tile (camera off) vs live screen?
                    blank_whole = _looks_blank_tile(_region_features(img))
                    blank_corner = crop is not None and _looks_blank_tile(_region_features(crop))
                    if blank_whole or blank_corner:
                        camera_state = "camera_off"
                        cam_off += 1
                    else:
                        camera_state = "screen_only"   # unassessable — never scored as off
                        screen_only += 1
        out.append({**fr, "faces": int(n), "face_present": n > 0,
                    "face_boxes": boxes, "face_points": points,
                    "face_source": face_source, "camera_state": camera_state})
    fd.close()
    fd_corner.close()
    present = sum(1 for f in out if f["face_present"])
    print(f"Face present in {present}/{len(out)} frames "
          f"(corner-recovered {recovered}; camera-off {cam_off}; screen-only {screen_only})")
    return out


def presence_summary(presence: list[dict], fps: float) -> dict:
    face = sum(1 for p in presence if p["face_present"])
    cam_off = sum(1 for p in presence if p.get("camera_state") == "camera_off")
    screen_only = sum(1 for p in presence if p.get("camera_state") == "screen_only")
    per = 1.0 / fps
    return {
        "frames_sampled": len(presence),
        "fps": fps,
        "face_visible_sec": round(face * per, 1),
        "screen_share_or_noface_sec": round((len(presence) - face) * per, 1),
        "camera_off_sec": round(cam_off * per, 1),
        "screen_only_sec": round(screen_only * per, 1),
    }


# ── 2b. Face-track breaks (active-speaker switches) ─────────────────────────
def _primary_box(p: dict):
    boxes = p.get("face_boxes") or []
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def _iou(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def mark_track_breaks(presence: list[dict]) -> list[dict]:
    """Flag frames where the on-screen face likely CHANGED (Zoom active-speaker
    switch): the face box jumped vs the previous face frame, or a face reappears
    after an absence. Identity is re-checked at every break so labels update the
    moment the person on screen changes."""
    prev_box, prev_idx = None, None
    breaks = 0
    for i, p in enumerate(presence):
        p["track_break"] = False
        box = _primary_box(p)
        if not p.get("face_present") or box is None:
            continue
        if prev_box is None:
            p["track_break"] = True          # first face seen
        elif i - prev_idx > 2:
            p["track_break"] = True          # face was gone (screen share / camera off)
        elif _iou(prev_box, box) < 0.25:
            p["track_break"] = True          # box jumped -> likely a different person
        breaks += 1 if p["track_break"] else 0
        prev_box, prev_idx = box, i
    print(f"Face-track breaks (likely person switches): {breaks}")
    return presence


# ── 3. Camera on/off aligned to speaking turns (local, free) ────────────────
def camera_analysis(presence: list[dict], segments: list[dict], roles: dict, fps: float) -> dict:
    def face_at(t: float) -> bool:
        idx = int(round(t * fps))
        return presence[idx]["face_present"] if 0 <= idx < len(presence) else False

    stats = {"trainer": {"speaking_sec": 0.0, "face_sec": 0.0},
             "candidate": {"speaking_sec": 0.0, "face_sec": 0.0}}
    for s in segments:
        role = roles.get(s["speaker"])
        if role not in stats:
            continue
        t = s["start"]
        while t < s["end"]:
            stats[role]["speaking_sec"] += 1.0 / fps
            if face_at(t):
                stats[role]["face_sec"] += 1.0 / fps
            t += 1.0 / fps

    out = {}
    for role, d in stats.items():
        spk = d["speaking_sec"]
        out[role] = {
            "speaking_sec": round(spk, 1),
            "on_camera_sec": round(d["face_sec"], 1),
            "on_camera_pct_of_speaking": round(d["face_sec"] / spk, 3) if spk else None,
        }
    return out


# ── 3b. Camera state aligned to speaking turns (trainer camera-off, local) ──
def _speaker_windows(segments: list[dict], roles: dict, role: str, gap: float = 8.0) -> list[dict]:
    """Merge a role's speaking segments into windows (gaps < `gap` joined),
    longest first. Used for the trainer camera-off and trainer-reading checks."""
    wins, cur = [], None
    for s in segments:
        if roles.get(s["speaker"]) == role:
            if cur and s["start"] - cur["end"] < gap:
                cur["end"] = s["end"]
            else:
                if cur:
                    wins.append(cur)
                cur = {"start": s["start"], "end": s["end"]}
    if cur:
        wins.append(cur)
    wins.sort(key=lambda a: a["end"] - a["start"], reverse=True)
    return wins


def camera_state_analysis(presence: list[dict], segments: list[dict], roles: dict, fps: float) -> dict:
    """Per role, measure camera state WHILE THAT ROLE IS THE ACTIVE SPEAKER (the
    corner tile / full frame shows them). Returns on / camera_off / unassessable
    seconds and the LONGEST CONTINUOUS camera-off stretch — the number the trainer
    camera-off rule scores. Screen-only (unassessable) frames break an off-stretch
    and are counted separately, so a pure screen-share moment is never a deduction.
    The stretch counter resets between speaking windows (the tile switches away)."""
    def state_at(t: float):
        idx = int(round(t * fps))
        return presence[idx].get("camera_state") if 0 <= idx < len(presence) else None

    per = 1.0 / fps
    out = {}
    for role in ("trainer", "candidate"):
        on = off = unassess = 0.0
        longest = 0.0
        longest_start = None
        for w in _speaker_windows(segments, roles, role):
            cur, cur_start = 0.0, None
            t = w["start"]
            while t < w["end"]:
                st = state_at(t)
                if st == "camera_off":
                    off += per
                    if cur == 0.0:
                        cur_start = t
                    cur += per
                    if cur > longest:
                        longest, longest_start = cur, cur_start
                else:
                    if st == "on":
                        on += per
                    elif st == "screen_only":
                        unassess += per
                    cur, cur_start = 0.0, None
                t += per
        out[role] = {
            "on_sec": round(on, 1),
            "camera_off_sec": round(off, 1),
            "unassessable_sec": round(unassess, 1),
            "assessable_sec": round(on + off, 1),
            "longest_camera_off_sec": round(longest, 1),
            "longest_camera_off_start": _sec_to_ts(longest_start) if longest_start is not None else None,
        }
    return out


# ── 4a. Gaze / reading tell (MediaPipe Face Mesh, curated answer frames) ────
# Landmark indices for the 478-point refined mesh
_L_EYE = (33, 133)      # left eye outer, inner corners
_R_EYE = (362, 263)     # right eye inner, outer corners
_L_IRIS = list(range(468, 473))
_R_IRIS = list(range(473, 478))
# 6-point model for head pose (nose, chin, eye corners, mouth corners)
_POSE_IDX = [1, 152, 33, 263, 61, 291]
_POSE_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
], dtype=np.float64)


def analyze_gaze(presence: list[dict], segments: list[dict], roles: dict, fps: float) -> list[dict]:
    import cv2
    import mediapipe as mp

    face_frames = [p for p in presence if p["face_present"]]
    answers = _candidate_answers(segments, roles)
    if not answers:  # no speaker roles (e.g. Whisper) -> sample across whole session
        answers = [{"start": face_frames[0]["time_sec"], "end": face_frames[-1]["time_sec"]}] if face_frames else []

    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                         max_num_faces=1, min_detection_confidence=0.5)
    results = []
    for ans in answers[:3]:  # cap: 3 longest answers
        frames = _frames_in_window(face_frames, ans["start"], ans["end"], k=5)
        yaws, pitches, hgrs, vgrs, used = [], [], [], [], []
        for f in frames:
            img = cv2.imread(f["path"])
            if img is None:
                continue
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            h, w = img.shape[:2]
            yaw, pitch = _head_pose(lm, w, h)
            hgr, vgr = _eye_gaze(lm)
            if yaw is not None:
                yaws.append(yaw); pitches.append(pitch)
            hgrs.append(hgr); vgrs.append(vgr); used.append(f)
        if not used:
            continue
        results.append(_score_reading(yaws, pitches, hgrs, vgrs, ans, used))
    fm.close()
    return results


def analyze_trainer_reading(presence: list[dict], segments: list[dict], roles: dict, fps: float) -> list[dict]:
    """Reading-from-screen tell for the TRAINER while explaining.

    Only full-frame camera-on frames are used (face_source == 'full'): when a
    document is being shared the trainer's face is a tiny corner tile or absent,
    so such windows have too few full frames and are SKIPPED — reading a shared
    document is legitimate and must not be penalised. A window assessed here is
    therefore one where the trainer is on camera explaining, not screen-sharing."""
    import cv2
    import mediapipe as mp

    wins = _speaker_windows(segments, roles, "trainer")
    if not wins:
        return []
    full_frames = [p for p in presence if p.get("face_present") and p.get("face_source") == "full"]
    if not full_frames:
        return []

    MIN_FULL = 3   # fewer full-frame faces than this -> document likely shared; skip
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                         max_num_faces=1, min_detection_confidence=0.5)
    results = []
    for w in wins[:3]:   # cap: 3 longest trainer windows
        inwin = [p for p in full_frames if w["start"] <= p["time_sec"] <= w["end"]]
        if len(inwin) < MIN_FULL:
            continue
        frames = _frames_in_window(inwin, w["start"], w["end"], k=5)
        yaws, pitches, hgrs, vgrs, used = [], [], [], [], []
        for f in frames:
            img = cv2.imread(f["path"])
            if img is None:
                continue
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            h, iw = img.shape[:2]
            yaw, pitch = _head_pose(lm, iw, h)
            hgr, vgr = _eye_gaze(lm)
            if yaw is not None:
                yaws.append(yaw); pitches.append(pitch)
            hgrs.append(hgr); vgrs.append(vgr); used.append(f)
        if not used:
            continue
        rec = _score_reading(yaws, pitches, hgrs, vgrs, w, used)
        rec["role"] = "trainer"
        rec["document_shared"] = False   # only non-screen-share full-frame frames used
        results.append(rec)
    fm.close()
    return results


def _score_reading(yaws, pitches, hgrs, vgrs, ans, used):
    """Reading tell = gaze both OFF-CENTRE and STABLE across the answer."""
    def mstd(a):
        return (float(np.mean(a)), float(np.std(a))) if a else (0.0, 0.0)
    yaw_m, yaw_s = mstd(yaws)
    pit_m, pit_s = mstd(pitches)
    hgr_m, hgr_s = mstd(hgrs)
    vgr_m, vgr_s = mstd(vgrs)

    # off-centre magnitude (head yaw/pitch in deg, eye ratios in [-1,1]); mean of the
    # strongest cues so a clear head-turn OR clear eye-cast registers, not just their average
    off_cues = sorted([abs(yaw_m) / 20, abs(pit_m) / 15, abs(hgr_m) / 0.6, abs(vgr_m) / 0.6], reverse=True)
    off = min(1.0, (off_cues[0] * 0.6 + off_cues[1] * 0.4))
    # stability: low variance -> high stability
    instab = min(1.0, (yaw_s / 20 + pit_s / 15 + hgr_s + vgr_s) / 4)
    stability = 1.0 - instab
    reading_likelihood = round(max(0.0, off * stability), 2)

    gaze = "fixed_offscreen" if reading_likelihood >= 0.6 else ("natural" if off < 0.3 else "unclear")
    return {
        "answer_time": _sec_to_ts(ans["start"]),
        "reading_likelihood": reading_likelihood,
        "gaze": gaze,
        "head_yaw_mean": round(yaw_m, 1), "head_pitch_mean": round(pit_m, 1),
        "eye_h_mean": round(hgr_m, 2), "eye_v_mean": round(vgr_m, 2),
        "stability": round(stability, 2),
        "frames_used": len(used),
        "evidence_frames": [os.path.basename(f["path"]) for f in used],
    }


def _head_pose(lm, w, h):
    try:
        import cv2
        img_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in _POSE_IDX], dtype=np.float64)
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(_POSE_3D, img_pts, cam, np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None
        rot, _ = cv2.Rodrigues(rvec)
        sy = (rot[0, 0] ** 2 + rot[1, 0] ** 2) ** 0.5
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
        # normalise yaw to [-90,90] range around frontal
        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180
        return float(yaw), float(pitch)
    except Exception:
        return None, None


def _eye_gaze(lm):
    """Horizontal/vertical iris offset within the eye, averaged over both eyes. [-1,1]"""
    def ratio(eye, iris):
        ox, oy = lm[eye[0]].x, lm[eye[0]].y
        ix, iy = lm[eye[1]].x, lm[eye[1]].y
        cx = float(np.mean([lm[k].x for k in iris]))
        cy = float(np.mean([lm[k].y for k in iris]))
        ew = (ix - ox) or 1e-6
        h = (cx - (ox + ix) / 2) / (abs(ew) / 2)
        v = (cy - (oy + iy) / 2) / (abs(ew) / 2)
        return h, v
    lh, lv = ratio(_L_EYE, _L_IRIS)
    rh, rv = ratio(_R_EYE, _R_IRIS)
    return float(np.clip((lh + rh) / 2, -1, 1)), float(np.clip((lv + rv) / 2, -1, 1))


# ── 4b. Identity + consistency (InsightFace, sampled frames) ────────────────
MAX_IDENTITY_SAMPLES = 400        # compute ceiling: cadence + switch samples combined
TRAINER_SIM = 0.42                # >= this vs the trainer centroid -> trainer
TRAINER_ENROL_SIM = 0.55          # strong matches refine the centroid (glasses/lighting robustness)
TRAINER_REJECT_SIM = 0.32         # <= this -> definitely not the trainer; between = decide by nearest centroid


def analyze_identity(presence: list[dict], cfg) -> dict:
    import cv2
    face_frames = [p for p in presence if p["face_present"]]
    if not face_frames:
        return {"same_person_throughout": None, "note": "no face frames"}

    app = _load_insightface(cfg.model_cache_dir)
    if app is None:
        return {"same_person_throughout": None, "note": "insightface unavailable"}

    # sample every N seconds PLUS every track break (active-speaker switch), so
    # the identity label updates the moment the on-screen person changes
    step = max(1, int(cfg.identity_sample_sec * cfg.frame_fps))
    picked = {f["path"]: f for f in face_frames[::step]}
    for f in face_frames:
        if f.get("track_break"):
            picked[f["path"]] = f
    sampled = sorted(picked.values(), key=lambda f: f["time_sec"])
    if len(sampled) > MAX_IDENTITY_SAMPLES:   # bound compute on very long/switchy sessions
        # keep ALL break samples (they're the re-identification guarantee);
        # the cadence samples absorb the cap
        brk = [f for f in sampled if f.get("track_break")]
        cad = [f for f in sampled if not f.get("track_break")]
        if len(brk) >= MAX_IDENTITY_SAMPLES:
            stride = len(brk) / MAX_IDENTITY_SAMPLES
            kept = {id(brk[int(i * stride)]) for i in range(MAX_IDENTITY_SAMPLES)}
            for f in brk:                     # dropped breaks must not blank labels downstream
                if id(f) not in kept:
                    f["track_break"] = False
            sampled = [f for f in brk if id(f) in kept]
        else:
            budget = MAX_IDENTITY_SAMPLES - len(brk)
            if len(cad) > budget:
                stride = len(cad) / budget
                cad = [cad[int(i * stride)] for i in range(budget)]
            sampled = sorted(brk + cad, key=lambda f: f["time_sec"])

    embeds = []  # (time_sec, path, embedding, bbox[x1,y1,x2,y2])
    for f in sampled:
        img = cv2.imread(f["path"])
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        biggest = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embeds.append((f["time_sec"], f["path"], _norm(biggest.normed_embedding),
                       [int(v) for v in biggest.bbox]))

    if not embeds:
        return {"same_person_throughout": None, "note": "no faces embedded"}

    trainer_emb = _trainer_embedding(app, cfg)
    # count switch-triggered checks among frames that actually got embedded,
    # so "samples" and "switch_samples" share the same denominator
    break_paths = {f["path"] for f in sampled if f.get("track_break")}
    result = {"samples": len(embeds),
              "switch_samples": sum(1 for _, p, _, _ in embeds if p in break_paths)}
    detail = []  # per-sample record: feeds the analysis video + audit trail

    if trainer_emb is not None:
        # pass 1: strong matches against the raw reference photo
        strong = [e for _, _, e, _ in embeds if _cos(e, trainer_emb) >= TRAINER_ENROL_SIM]
        # self-enrolment: refine the reference into a session centroid so glasses,
        # lighting, and camera angle differences don't drop real matches
        centroid = _norm(np.mean([trainer_emb] + strong, axis=0)) if strong else trainer_emb
        result["trainer_enrolled_samples"] = len(strong)

        # pass 2: score everyone against the centroid with a hysteresis band
        # (records carry the RAW sim; rounding happens only at display time)
        trainer_hits, candidate, uncertain = [], [], []
        for t, p, e, bb in embeds:
            s = _cos(e, centroid)
            rec = (t, p, e, bb, s)
            if s >= TRAINER_SIM:
                trainer_hits.append(rec)
            elif s <= TRAINER_REJECT_SIM:
                candidate.append(rec)
            else:
                uncertain.append(rec)

        # borderline faces: nearest centroid decides (trainer vs candidate cluster).
        # With no candidate cluster to compare against, require the midpoint of the
        # hysteresis band — never accept everyone above the reject line as trainer.
        cand_clusters = _cluster([e for _, _, e, _, _ in candidate])
        cand_centroid = (_norm(np.mean([candidate[i][2] for i in max(cand_clusters, key=len)], axis=0))
                         if cand_clusters else None)
        midpoint = (TRAINER_SIM + TRAINER_REJECT_SIM) / 2
        for rec in uncertain:
            s_tr = rec[4]
            if cand_centroid is None:
                (trainer_hits if s_tr >= midpoint else candidate).append(rec)
            else:
                s_cand = _cos(rec[2], cand_centroid)
                (trainer_hits if s_tr >= s_cand else candidate).append(rec)
        cand_clusters = _cluster([e for _, _, e, _, _ in candidate])

        result["trainer_present"] = len(trainer_hits) > 0
        result["candidate_present"] = len(candidate) > 0
        result["candidate_identities"] = len(cand_clusters)
        # per-identity on-screen presence -> feeds the duration-gated proxy-swap
        # policy in scoring (a brief blip must not cost the same as a real 2nd person)
        ip = []
        for ci, cluster in enumerate(cand_clusters):
            ct = sorted(candidate[i][0] for i in cluster)
            ip.append({"label": f"person-{ci + 1}", "samples": len(cluster),
                       "presence_sec": round(len(cluster) * cfg.identity_sample_sec, 1),
                       "first_time_sec": round(ct[0], 1), "last_time_sec": round(ct[-1], 1),
                       "primary": False})
        if ip:
            max(ip, key=lambda d: d["samples"])["primary"] = True
        result["identity_presence"] = ip
        # "same person" allows brief blips: only a SUSTAINED second identity (present
        # longer than the grace window) counts as a real change. Session-type policy
        # (1:1 vs group, -40 vs -25) is decided later in scoring.
        grace = getattr(cfg, "person_change_grace_sec", 120) or 120
        sustained = any((not p["primary"]) and p["presence_sec"] > grace for p in ip)
        result["same_person_throughout"] = not sustained
        for t, p, _, bb, s in trainer_hits:
            detail.append({"time_sec": t, "frame": os.path.basename(p),
                           "label": "trainer", "bbox": bb, "sim": round(s, 2)})
        for ci, cluster in enumerate(cand_clusters):
            for i in cluster:
                t, p, _, bb, s = candidate[i]
                detail.append({"time_sec": t, "frame": os.path.basename(p),
                               "label": f"person-{ci + 1}", "bbox": bb, "sim": round(s, 2)})
        if len(cand_clusters) > 1:
            odd = min(cand_clusters, key=len)
            times = [_sec_to_ts(candidate[i][0]) for i in odd]
            result["note"] = f"{len(cand_clusters)} distinct non-trainer faces; odd segment(s) ~{times[:5]}"
            result["evidence_frames"] = [os.path.basename(candidate[i][1]) for i in odd[:5]]
    else:
        clusters = _cluster([e for _, _, e, _ in embeds])
        result["distinct_identities"] = len(clusters)
        # expected 2 (trainer + candidate); >2 is worth a look
        result["same_person_throughout"] = len(clusters) <= 2
        result["note"] = ("no trainer reference image; "
                          f"{len(clusters)} distinct identities seen (2 = trainer+candidate is normal)")
        for ci, cluster in enumerate(clusters):
            for i in cluster:
                t, p, _, bb = embeds[i]
                detail.append({"time_sec": t, "frame": os.path.basename(p),
                               "label": f"person-{ci + 1}", "bbox": bb})
        if len(clusters) > 2:
            result["evidence_frames"] = [os.path.basename(embeds[c[0]][1]) for c in clusters]

    result["samples_detail"] = sorted(detail, key=lambda d: d["time_sec"])
    return result


def _load_insightface(cache_dir: str):
    try:
        from insightface.app import FaceAnalysis
        os.makedirs(cache_dir, exist_ok=True)
        print("Loading InsightFace buffalo_l (first run downloads ~300MB)...")
        app = FaceAnalysis(name="buffalo_l", root=cache_dir,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 = CPU
        return app
    except Exception as exc:
        print(f"InsightFace load failed ({exc}); identity/consistency skipped")
        return None


def _trainer_embedding(app, cfg):
    if not (cfg.trainer_image_path and os.path.exists(cfg.trainer_image_path)):
        return None
    import cv2
    img = cv2.imread(cfg.trainer_image_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        print("No face found in trainer reference image")
        return None
    biggest = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return _norm(biggest.normed_embedding)


def _cluster(embs, thr=0.45):
    """Greedy cosine clustering; returns list of clusters (each a list of indices)."""
    clusters = []          # list of {"centroid": vec, "idx": [i,...]}
    for i, e in enumerate(embs):
        best, bi = -1.0, -1
        for j, c in enumerate(clusters):
            s = _cos(e, c["centroid"])
            if s > best:
                best, bi = s, j
        if best >= thr:
            clusters[bi]["idx"].append(i)
            m = clusters[bi]["idx"]
            clusters[bi]["centroid"] = _norm(np.mean([embs[k] for k in m], axis=0))
        else:
            clusters.append({"centroid": e, "idx": [i]})
    return [c["idx"] for c in clusters]


# ── orchestrator ────────────────────────────────────────────────────────────
def run_video(frames: list[dict], segments: list[dict], roles: dict, cfg) -> tuple[dict, list[dict]]:
    """Returns (video_result, presence). presence is reused by the annotation
    check and the proof builder so faces aren't re-detected."""
    presence = detect_presence(frames)
    mark_track_breaks(presence)
    summary = presence_summary(presence, cfg.frame_fps)
    camera = camera_analysis(presence, segments, roles, cfg.frame_fps)
    camera_state = camera_state_analysis(presence, segments, roles, cfg.frame_fps)
    print("Analyzing gaze (MediaPipe Face Mesh)...")
    gaze = analyze_gaze(presence, segments, roles, cfg.frame_fps)
    print("Analyzing trainer reading-from-screen...")
    trainer_reading = analyze_trainer_reading(presence, segments, roles, cfg.frame_fps)
    print("Analyzing identity / consistency (InsightFace)...")
    consistency = analyze_identity(presence, cfg)
    result = {
        "enabled": True,
        **summary,
        "camera": camera,
        "camera_state": camera_state,
        "vision": {"gaze": gaze, "trainer_reading": trainer_reading, "consistency": consistency},
        "evidence_frames_dir": "frames/",
    }
    return result, presence


# ── helpers ─────────────────────────────────────────────────────────────────
def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def _cos(a, b):
    return float(np.dot(a, b))


def _candidate_answers(segments, roles):
    return _speaker_windows(segments, roles, "candidate")


def _frames_in_window(face_frames, start, end, k=5):
    inside = [f for f in face_frames if start <= f["time_sec"] <= end]
    if not inside:
        return []
    if len(inside) <= k:
        return inside
    step = len(inside) / k
    return [inside[int(i * step)] for i in range(k)]


def _sec_to_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"
