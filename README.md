# Training Session Integrity Analyzer

Local, Dockerised tool that audits a recorded training session and writes a
`result.json`. Built for **CPU-only** machines (no GPU needed).

## What it does

**Transcript track** — did the trainer cover the day's planned sections (and for
roughly the allotted minutes), how the talk-time split, whether other languages were
used, any integrity concerns in what was said, plus a **full meeting description**
(narrative, minute-by-minute timeline, topics, notable quotes). Deterministic
metrics + **two GPT-4o text calls**. (The optional presentation check below adds one
vision call — everything else is local.)

**Video track — 100% local, open-source models, no GPU:**
- **MediaPipe Face Detection** → per-frame face-vs-screen-share, and camera on/off
  during each person's speaking turns.
- **MediaPipe Face Mesh** (iris + head pose) → gaze / reading-tell (is the candidate
  staring at a fixed off-screen point while answering).
- **InsightFace `buffalo_l`** (ArcFace embeddings) → trainer-vs-candidate identity and
  whether the on-camera person changed mid-session (proxy swap).

**Candidate presentation check** (optional, `ENABLE_PRESENTATION_ANALYSIS`) — one
GPT-4o vision call on a few candidate frames: attire, grooming tidiness, posture,
background, camera framing, lighting. **Descriptive and informational only** — it
never changes the integrity score; it exists as coaching input for a human reviewer.

Output: **trainer coverage %** (per-section, planned vs actually-spent minutes) and a
**session integrity score /100** (Clean / Review / High-risk) — the integrity score
answers *"should a human review this session?"*. Every deduction carries its evidence,
and a **`proof/` folder** holds the receipts: one folder per red flag with the quote,
timestamp, and evidence frames, plus a per-section coverage report.

---

## Hardware

Runs on CPU. Tested target: **i7, 32 GB RAM, no GPU** — comfortable. Models are small
(InsightFace `buffalo_l` ≈ 300 MB, Whisper `base` ≈ 140 MB) and run via ONNX Runtime /
MediaPipe on CPU.

## Models download **once**

On the **first** `docker compose run`, InsightFace downloads `buffalo_l` (~300 MB) and,
if you don't supply a VTT, Whisper downloads its model. Both are cached in the
`model-cache` Docker volume, so every later run is offline and instant. MediaPipe's
models ship inside the pip package — no download.

---

## Setup

1. Install Docker + Docker Compose.
2. `cp .env.example .env` and edit it.
3. Put files in place:
   - video → `./input/` (e.g. `./input/session.mp4`)
   - VTT   → `./input/` (optional; else Whisper transcribes the audio)
   - trainer face → `./trainer/` (optional but needed for reliable swap detection)
4. Set at least `VIDEO_FILE`, `DAY_NUMBER`, and either `OPENAI_API_KEY` or `MOCK_OPENAI=true`.

### Using a real `training-temp.json` (production parity)

In production the EC2 worker gets `day`, `trainer`, `candidate`, and `meeting_id` from
the SQS job, which is built from the `training-temp.json` the Lambda writes to S3. To
test the exact same way, **drop that `training-temp.json` into `./input/`**. The
analyzer auto-detects it and reads those fields from it — no need to set `DAY_NUMBER`,
`TRAINER_NAME`, or `CANDIDATE_NAME` in `.env` (the temp file overrides them). Example:

```
./input/Divya-Prajapati.mp4
./input/Divya-Prajapati.vtt
./input/training-temp.json      <-- day, trainer, candidate, meeting_id read from here
./trainer/Divya-Prajapati.png
```

`result.json` then records `meeting_id`, `s3_prefix`, and `day_step_name` from the temp
file, and `config_source: "training-temp.json"`.

> The rubric is chosen by the `day` **number**. If the temp file's `day_step_name`
> (e.g. "Day 8 - Coding and support...") doesn't match `day_plan.json`'s Day 8 title,
> the analyzer prints a warning and records both — update `day_plan.json` so its day
> definitions match your real program.

## Run

```bash
docker compose build          # first build compiles InsightFace; takes a few minutes
docker compose run --rm analyzer
```

`result.json`, evidence `frames/`, the `proof/` folder, and **`report.html`** land in
`./output/`. Open `report.html` in any browser for a plain-language view of the whole
result — scores, red flags with their proof photos and quotes, per-topic coverage,
and the candidate presentation feedback. Regenerate it from an existing result with
`python -m analyzer.report output/result.json` (no Docker needed).

### Suggested first runs

- **Plumbing only (free, offline):** `MOCK_OPENAI=true`, `ENABLE_VIDEO_ANALYSIS=false`.
- **Transcript only (real):** your key, `ENABLE_VIDEO_ANALYSIS=false`. Fast, highest-value half.
- **Full run:** `ENABLE_VIDEO_ANALYSIS=true`. First run downloads the InsightFace model.

Point at a folder elsewhere by editing the left side of the volume lines in
`docker-compose.yml`.

---

## What's inside `result.json`

```jsonc
{
  "meeting":  { "video_file", "day", "day_title", "trainer_name", "candidate_name",
                "duration_sec", "analyzed_at" },

  "summary": {                          // plain-language read of everything below
    "session", "duration",              // "10.8 of 120 planned minutes (9%)"
    "trainer_coverage", "session_integrity",
    "red_flags": ["scripted_deception (-20 pts): '...' — at ~10:10"],
    "candidate_presentation", "proof_folder"
  },

  "scoring": {
    "trainer_coverage_score": 60,       // % of the day's planned sections covered
    "session_integrity_score": 45,      // 100 = clean; deductions applied below
    "tier": "High-risk",                // Clean 80-100 / Review 50-79 / High-risk <50
    "deductions": [                     // every point lost, with reason + evidence
      {"reason": "candidate_camera_off", "points": -20, "severity": "high",
       "evidence": "candidate on camera 22% of speaking time"}
      // proxy_interview_coaching is graduated: -5 for a single mention,
      // -10 when discussed repeatedly ("occurrences" records the count)
    ]
  },

  "flags": [ /* every raw signal that fired, transcript + video, with evidence */ ],

  "transcript": {
    "source": "vtt",                    // or "whisper"
    "metrics": {
      "duration_sec": 5220.0,
      "talk_time": {"trainer_sec", "candidate_sec", "trainer_ratio", "candidate_ratio"},
      "language":  {"non_english_sec", "flagged_segments":[...]},
      "speaker_roles": {"Ronak": "trainer", "Bala": "candidate"}
    },
    "coverage_analysis": {
      "coverage_pct": 60,
      "coverage": [ {"section","status","expected_minutes","approx_minutes_spent",
                     "evidence","approx_time"} ],
      "integrity_flags": [ {"type","confidence","evidence","approx_time"} ],
      "proof": "proof/section_coverage/coverage_report.txt"
    }
  },

  "candidate_presentation": {           // informational only — never scores
    "attire", "grooming_hair", "posture_body_language", "background",
    "camera_setup", "lighting", "overall_notes", "coaching_suggestions",
    "frames_used", "proof": ["proof/candidate_presentation/..."]
  },

  "video": {
    "enabled": true,
    "face_visible_sec": 4320.0, "screen_share_or_noface_sec": 900.0,
    "camera": {
      "trainer":   {"speaking_sec","on_camera_sec","on_camera_pct_of_speaking"},
      "candidate": {"speaking_sec","on_camera_sec","on_camera_pct_of_speaking"}
    },
    "vision": {
      "gaze": [ {"answer_time","reading_likelihood","gaze","head_yaw_mean",
                 "head_pitch_mean","eye_h_mean","eye_v_mean","stability","evidence_frames"} ],
      "consistency": {"same_person_throughout","trainer_present","candidate_present",
                      "candidate_identities","note","evidence_frames"}
    },
    "evidence_frames_dir": "frames/"
  }
}
```

**Why raw seconds are stored** (not only percentages): so multiple recordings of the
same session (host restarts) can be merged correctly — sum the seconds, then compute
ratios once. Averaging per-chunk percentages gives wrong numbers.

---

## Notes & limits

- Speaker roles are reliable with a Zoom VTT (speaker-labelled). With Whisper there are
  no speaker labels, so talk-time split and per-answer gaze degrade — supply the VTT.
- Swap/consistency is most reliable **with** a trainer reference image; without one it
  reports the number of distinct identities (2 = trainer + candidate is normal).
- Gaze is a **flag for human review**, not a verdict — head-pose + iris estimation on a
  webcam is inherently noisy.
- If the build hits an OpenCV import clash (MediaPipe vs InsightFace both pull OpenCV),
  rebuild with `docker compose build --no-cache`; the pinned versions are chosen to avoid it.
