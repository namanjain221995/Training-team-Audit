# CLAUDE.md — Training Session Integrity Analyzer

> This file is auto-loaded by Claude Code as project memory. It describes the
> local analyzer in this repo **and** the production pipeline it plugs into, so
> you (Claude Code) have full context when helping with this codebase.

---

## 1. What this project is

A system that audits recorded **Zoom training sessions** for a staffing company
(TechSara). A Zoom recording + transcript are analyzed to produce a
`result.json` containing:

- **Trainer coverage** — did the trainer cover the day's planned sections (from a
  fixed rubric), for roughly the allotted minutes, talk-time split, language use,
  and on-camera presence.
- **Session integrity flags** — detection signals for human review: candidate
  camera off during answers, gaze fixed off-screen (reading tell), on-camera
  person changing mid-session (proxy swap), session far shorter than the day
  plan, and integrity concerns spotted in the transcript.
- **Candidate presentation check** — descriptive, informational-only observations
  (attire, grooming tidiness, posture, background, camera setup, lighting).
- Two scores: `trainer_coverage_score` (%) and `session_integrity_score` (/100),
  plus a `proof/` folder holding the evidence behind every flag.

**This repo is the analyzer** — it runs locally in Docker on a single video for
testing, and the same logic is what the production **EC2 worker** will run at
scale (see §8).

## 2. Scope (keep the codebase aligned to this)

This is a **quality-control / integrity audit**. It flags sessions for a human to
review; it does **not** make automated judgments about people.

- Candidate video signals are **detection** signals: camera on/off, gaze /
  reading tell, and person-consistency (proxy swap).
- **Candidate presentation check** (added 2026-07-02 at owner request): GPT-4o
  vision describes ONLY controllable presentation factors the program itself
  trains (Day 7 rubric: attire, grooming tidiness, posture/body language,
  background, camera framing, lighting). Hard boundaries, enforced in the prompt
  (`analyzer/presentation.py`): no comments on body shape, weight, skin, age,
  ethnicity, gender, or attractiveness; observations are **descriptive coaching
  input for a human reviewer** and **never change the integrity score**
  (`informational_only: true`).
- Still out of scope: any automated scoring of a person's physical
  characteristics, or using presentation observations in the integrity score.

## 3. System architecture (production, big picture)

```
Zoom meeting ends
   │  fires TWO webhooks (at different times)
   │    1) recording.completed            → video (MP4), audio (M4A), chat
   │    2) recording.transcript_completed → transcript (VTT), arrives later
   ▼
API Gateway ──▶ SQS ──▶ Lambda  "zoom-recording-processor"
   │  Lambda uploads files to S3 under a structured prefix (see §9),
   │  then (Training only) runs a "rendezvous" on the TRANSCRIPT event:
   │    • confirm the MP4 is already in S3
   │    • look up the training Day from Salesforce (see §7)
   │    • write training-temp.json into the meeting prefix
   │    • enqueue ONE job to SQS "training-analysis-jobs"
   ▼
SQS "training-analysis-jobs"
   ▼
EC2 worker  (NOT BUILT YET — see §8/§11)
   │  drains the queue, runs THIS analyzer's logic on the recording,
   │  writes result.json, merges multi-recording chunks, and
   ▼
Salesforce  (writes the score + flags back to the Session record)
```

The Lambda side is deployed and working. **The EC2 worker is the main remaining
piece**, and the code in this repo is the reference implementation for it.

## 4. This repo — the local analyzer

```
training-analysis/
├── analyzer/
│   ├── __main__.py       # entrypoint: 5 steps — transcript, video, presentation, scoring, proof
│   ├── config.py         # env + training-temp.json → Config; builds container paths
│   ├── openai_client.py  # GPT-4o wrapper (text + vision), with MOCK_OPENAI stub mode
│   ├── transcript.py     # VTT parse / Whisper fallback, metrics, GPT-4o coverage + integrity
│   ├── video.py          # frames + MediaPipe (presence, gaze) + InsightFace (identity/swap)
│   ├── presentation.py   # GPT-4o vision candidate presentation check (informational only)
│   ├── proof.py          # builds output/proof/ — evidence folder per red flag
│   ├── report.py         # renders result.json → output/report.html (non-technical view;
│   │                     #  pure stdlib: `python -m analyzer.report output/result.json` on host)
│   └── scoring.py        # weighted deductions → integrity score; coverage score; summary; tiers
├── config/
│   └── day_plan.json     # the 10-day rubric; details carry the FULL "Trainer Focus / Key
│                         #  Details" text from Training Plan (4).pdf (rebuilt 2026-07-02)
├── input/                # drop the video (+ optional .vtt, training-temp.json) here
├── output/               # result.json + frames/ + proof/ land here
├── trainer/              # optional trainer reference face image
├── Dockerfile            # python:3.11-slim + ffmpeg + build tools; installs the models' deps
├── docker-compose.yml    # mounts input/output/trainer + a model-cache volume
├── requirements.txt
├── .env.example          # copy to .env
└── README.md
```

### The tracks

**Transcript track** (`transcript.py`) — cheap, deterministic + one GPT-4o call:
1. Parse the Zoom **VTT** into `{speaker, start, end, text}` segments. If no VTT is
   given, transcribe the audio locally with **faster-whisper** (no speaker labels
   in that path — see gotchas).
2. Deterministic metrics: duration, per-role talk-time in **seconds**, language
   detection (langdetect) → non-English flags.
3. **GPT-4o** (`analyze_coverage`): given the day's sections (full detail text +
   `expected_minutes`) and the transcript, returns per-section coverage
   (`covered/partial/not_covered`, `depth_vs_allotted`, **`approx_minutes_spent`**
   estimated from timestamps, evidence quote + approx_time) and
   `integrity_flags`, all as strict JSON. Sections are matched by MEANING (the
   detail text), not by the trainer literally naming them.
4. **GPT** (`describe_meeting`): a second text call producing the full meeting
   record — `overview` (3-5 paragraph narrative), `timeline` (mm:ss events),
   **`topics[]` per-topic QA** (start/end, approx_minutes, led_by, plan_section
   mapping, on_curriculum true/false — given the day's planned sections), QA-style
   trainer/candidate summaries, `notable_quotes`. Lifted to top-level
   `meeting_description` in result.json by `scoring.assemble`.
   Both text calls + the vision call run on `OPENAI_MODEL` (default `gpt-5.5`
   with `OPENAI_REASONING_EFFORT=high`); `openai_client.py` auto-drops
   temperature/seed for reasoning models that reject them.

**Video track** (`video.py`) — 100% local, CPU-only, no GPU, no cloud vision:
- **MediaPipe Face Detection** → dense per-frame presence (face vs screen-share).
- **MediaPipe Face Mesh** (iris + head pose via solvePnP) → gaze / reading tell.
- **InsightFace `buffalo_l`** (ArcFace embeddings, ONNX Runtime CPU) → trainer-vs-
  candidate identity and person-consistency (proxy swap).
- Cost control: MediaPipe runs on every frame; InsightFace only every
  `IDENTITY_SAMPLE_SEC` seconds of face-visible video.
- `run_video` returns `(result, presence)` — presence is reused by the
  presentation check and proof builder.

**Presentation check** (`presentation.py`) — one GPT-4o vision call on ≤4 frames
picked locally from candidate-speaking, face-on-screen moments. See §2 for its
boundaries. Toggle with `ENABLE_PRESENTATION_ANALYSIS`.

**Proof folder** (`proof.py`) — after scoring, every applied deduction gets
`output/proof/<reason>/` containing `proof.txt` (quote, timestamp, numbers) and
evidence frames (detector-named frames, or frames nearest the quoted moment for
transcript flags). Also writes `section_coverage/coverage_report.txt` (per-section
plan-vs-spent table) and `candidate_presentation/` (assessed frames +
observations). Deductions/flags in result.json carry `proof` path lists. Proof is
built BEFORE the optional `SAVE_EVIDENCE_FRAMES=false` cleanup so copies survive.

### Scoring (`scoring.py`)

Two numbers. `trainer_coverage_score` = % of the day's sections covered.
`session_integrity_score` starts at 100 and takes deductions, each carrying its
evidence (timestamp, transcript quote, `evidence_frames`, and `proof` paths):

| reason | points |
|---|---|
| `proxy_interview_coaching` (repeated; **−5/medium if mentioned only once** — owner-tuned 2026-07-04) | −10 |
| `person_change_midsession` | −40 |
| `fabricated_experience` | −30 |
| `scripted_deception` | −20 |
| `candidate_camera_off` (< 50% on-camera while speaking, >30s spoken) | −20 |
| `session_too_short` (ran < 50% of the day plan's minutes; **−25/critical** if < 25%) | −15 |
| `gaze_fixed_offscreen` (reading_likelihood ≥ 0.6) | −15 |
| `key_section_rushed` (heavy section, ≥25 min allotted, skipped/under-covered) | −10 |
| `non_english_heavy` (≥ 120s) | −5 |

Same reason counts once (worst occurrence). Tiers: **Clean 80–100 · Review 50–79
· High-risk <50**. `session_too_short` caveat: multi-recording chunks each look
short — the merge step (§9) must recompute it from summed chunk durations (raw
minutes are stored in the flag for that purpose).

### `result.json` shape (top level)

`meeting` (video_file, meeting_id, day, day_title [from rubric], day_step_name
[from Salesforce/temp], trainer_name, candidate_name, s3_prefix, s3_bucket,
duration_sec, config_source) · **`summary`** (plain-language: session, duration
vs plan, coverage line, integrity line, red_flags[], presentation line) ·
**`meeting_description`** (overview narrative, timeline[], topics_discussed[],
trainer/candidate summaries, notable_quotes[], available) ·
`scoring` (the two scores, tier, deductions[] each with `proof` paths) ·
`flags[]` (flat list of everything that fired, with `proof`) ·
**`candidate_presentation`** (informational vision observations + frames_used) ·
`transcript` (source, metrics {talk_time, language, speaker_roles},
coverage_analysis incl. per-section expected vs approx_minutes_spent + `proof`) ·
`video` (presence seconds, camera{} per role, vision{gaze[], consistency}) ·
`proof_dir`.

## 5. Inputs and the `training-temp.json` contract

For local testing you drop files in `./input`. If a **`training-temp.json`** is
present (auto-detected), the analyzer reads `day`, `trainer`, `candidate`,
`meeting_id`, `day_step_name` FROM IT — overriding the manual `.env` values. This
mirrors production, where the EC2 worker gets the same fields from the SQS job.

`training-temp.json` (written by the Lambda into the S3 meeting prefix):
```json
{
  "meeting_id": "93488261018",
  "prefix": "Training/{Trainer}/{Year}/{Month}/{Candidate}/{Date}/{Time}/{MeetingID}/",
  "bucket": "zoom-automation-bucket",
  "candidate": "Abhinaya_Sree_-_Talluri",
  "trainer": "Divya_Prajapati",
  "day": 8,
  "day_step_name": "Day 8 - Coding and support during coding interview",
  "video_ready": true, "transcript_ready": true, "enqueued": true,
  "created_at": "...", "enqueued_at": "..."
}
```
The SQS message is a lean subset: `meeting_id, bucket, prefix, day, day_step_name,
candidate, trainer`. Folder-style names (`Divya_Prajapati`) are humanized to
`Divya Prajapati` for speaker matching. If `day` is null (Salesforce lookup
failed), the analyzer falls back to `DAY_NUMBER`.

## 6. `config/day_plan.json` — the rubric

Keyed by day number ("1".."10"). Each day: `title`, `total_minutes`, and
`sections[]` where each section has `name`, `detail`, `planned_start`,
`planned_end`, `expected_minutes`. The **`day` number** selects which rubric to
audit against; times are treated as *expected emphasis* (to flag rushed heavy
sections), NOT a rigid schedule — real sessions run out of order.

**2026-07-02:** `detail` fields now carry the complete "Trainer Focus / Key
Details" text from the source `Training Plan (4).pdf`, so GPT-4o matches
transcript content to sections by meaning.

> **Known mismatch (still open):** Salesforce `day_step_name` values differ from
> the rubric titles per candidate (e.g. Day 8 rubric = "Active Listening and
> Adaptive Discussion" vs Salesforce = "Day 8 - Coding and support during coding
> interview"). The analyzer prints a warning and records both. If the Salesforce
> step names reflect a *different real curriculum* (not just different labels),
> the rubric must be reconciled with it.

## 7. Salesforce integration (Lambda side)

Day lookup, matched on the **numeric meeting_id** (not uuid):
```sql
SELECT Candidate_Training_Step__r.Name, Candidate__r.Name
FROM Session__c
WHERE External_Meeting_ID__c = '<meeting_id>' LIMIT 1
```
The day number is parsed by regex from the step name ("Day 8 - ..." → 8). The
worker will write the score/flags back to this `Session__c` record.

> **Casing gotcha:** the Session field is `External_Meeting_ID__c` (uppercase ID),
> but the Interview object uses `Zoom_Meeting_Id__c` (lowercase Id). Verify field
> API names before querying — a wrong case returns no rows and yields `day: null`.

## 8. Production context — AWS resources

- Account `985100584614`, region `us-east-1`
- S3 bucket `zoom-automation-bucket`
- Salesforce secret `sf/jwt/credentials` (keys: SF_CLIENT_ID, SF_USERNAME,
  SF_LOGIN_URL, PRIVATE_KEY_B64)
- Zoom secret `zoom/general-oauth`
- SQS `training-analysis-jobs` (VisibilityTimeout=1800, MessageRetention=345600)
- Lambda env: `ANALYSIS_QUEUE_URL`, `SF_SESSION_OBJECT_API_NAME=Session__c`,
  `SF_SESSION_MEETING_FIELD_API_NAME=External_Meeting_ID__c`,
  `SF_SESSION_STEP_RELATION=Candidate_Training_Step__r.Name`,
  `SF_SESSION_CANDIDATE_RELATION=Candidate__r.Name`

## 9. S3 layout and the multi-recording merge (design — NOT built yet)

Files land at:
```
Training/{Trainer}/{Year}/{Month}/{Candidate}/{Date}/{Time}/{MeetingID}/{MP4|M4A|TRANSCRIPT|CHAT}/
```

**Multi-recording problem:** if a host stops and restarts, Zoom keeps the **same
`meeting_id`** but a **new `uuid` + `start_time`**, so one logical session lands in
several `Time-*` folders (all resolving to the same Day). The worker must group by
**`meeting_id` + date** and produce ONE `session-result.json`.

Rules for the merge step:
- Analyze each chunk once → per-chunk `result.json` in its Time folder.
- Merge into `.../{Candidate}/{Date}/session-result-{meeting_id}.json` → this is
  what goes to Salesforce.
- **Store raw seconds / sets / counts per chunk; compute ratios and percentages
  ONLY at merge time. Never average per-chunk percentages.** (This is why
  `result.json` stores `*_sec` values.) Applies to `session_too_short` too: sum
  chunk minutes before comparing to the plan.
- Idempotent: a late chunk re-triggers the merge and overwrites the session file.

**Open decision:** when a session splits, is it a **false start** (keep the
longest chunk, drop short ones) or a **genuine continuation** (stitch all chunks in
`start_time` order)? Resolve before writing the merge.

## 10. Running & testing locally

```bash
cp .env.example .env          # set OPENAI_API_KEY (or MOCK_OPENAI=true)
docker compose build          # first build compiles InsightFace (a few min)
docker compose run --rm analyzer
```

- First real run downloads InsightFace `buffalo_l` (~300 MB) into the
  `model-cache` volume — once. MediaPipe models ship in the pip package.
- **Cheapest sanity check:** `MOCK_OPENAI=true` + `ENABLE_VIDEO_ANALYSIS=false`
  → ~10s, no key, proves plumbing + temp-file parsing.
- **Transcript-only real run:** key set, `ENABLE_VIDEO_ANALYSIS=false` (highest
  value, fastest).
- Output: `./output/result.json` + `./output/frames/` + `./output/proof/` +
  `./output/report.html` (browser-openable plain-language report).

### Environment variables

`OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-5.5` reasoning model; `gpt-4o` =
cheaper), `OPENAI_REASONING_EFFORT` (gpt-5.x/o* only: none..xhigh),
`MOCK_OPENAI` · `VIDEO_FILE`,
`TRANSCRIPT_FILE`, `TRAINER_IMAGE_FILE`, `TRAINING_TEMP_FILE` (blank = auto-detect
`training-temp.json`) · `DAY_NUMBER`, `TRAINER_NAME`, `CANDIDATE_NAME` (ignored if
a temp file is present) · `ENABLE_VIDEO_ANALYSIS`, `FRAME_FPS`,
`IDENTITY_SAMPLE_SEC`, `WHISPER_MODEL`, `SAVE_EVIDENCE_FRAMES` ·
`ENABLE_PRESENTATION_ANALYSIS` (informational vision check) · `OUTPUT_FILE`.

## 11. Remaining work / roadmap

1. **EC2 worker** — long-poll `training-analysis-jobs`, pull the recording from S3
   using `prefix`, run this analyzer's logic, write `result.json` back to the
   meeting prefix. (Lambda → SQS → EC2; never invoke EC2 directly.)
2. **Multi-recording merge** — implement §9 (grouping + the chosen false-start vs
   stitch rule + raw-seconds merge; recompute `session_too_short` from summed
   minutes).
3. **Salesforce write-back** — push `session_integrity_score`, `tier`, and the
   flag summary onto the `Session__c` record.
4. **Verify Salesforce step names** vs the rubric (see §6): same curriculum with
   different labels, or genuinely different per-day content?

## 12. Conventions & gotchas

- **numpy pinned `<2`** — InsightFace / MediaPipe / older ONNX Runtime break on
  numpy 2.x.
- **mediapipe pinned `==0.10.18`** — MediaPipe DELETED the legacy `mp.solutions`
  API (face_detection / face_mesh, used by `video.py`) in 0.10.2x+ releases.
  0.10.18 is the newest release that still ships `solutions` AND uses protobuf 4
  (required by modern onnx/onnxruntime). Do not loosen to a range; 0.10.9 forces
  protobuf 3 and conflicts with onnxruntime.
- **Env vars: blank ≠ missing.** docker-compose passes `VAR=` as an empty string;
  `config.py` uses `_int`/`_float`/`_bool` helpers that treat blank as "use the
  default". Never call `int(os.environ.get(...))` directly.
- **Windows bind-mount I/O:** rapid small-file writes to a mounted host dir can
  fail mid-run with `Input/output error` under Docker Desktop (ffmpeg died at
  frame ~1383 on a 28-min video). Frames are therefore extracted and analyzed in
  container-local `/tmp/frames` (`cfg.frames_work_dir`) and bulk-copied to
  `output/frames/` once at the end. proof.py resolves evidence frames via the
  frames list, never via `cfg.frames_dir`.
- **OpenCV clash:** MediaPipe and InsightFace both pull OpenCV. If a build breaks
  on `cv2` import, rebuild with `docker compose build --no-cache`.
- **Zoom records active-speaker view**, not a grid — so camera-on/off and gaze
  reflect who was on screen when speaking, not a continuous both-faces track. High
  `screen_share_or_noface_sec` is normal for screen-share-heavy sessions.
- **Whisper has no speaker diarization** — without a VTT, talk-time-per-role,
  per-answer gaze, and presentation frame-picking degrade (frames may show the
  trainer). Prefer the Zoom VTT.
- **Person-consistency needs a trainer reference image** to reliably separate
  trainer vs candidate; without it, it reports the count of distinct identities
  (2 = trainer + candidate is normal).
- Vision signals (gaze especially) are **flags for human review, not verdicts.**
  The presentation check is coaching input, never a score.
- langdetect misfires on very short segments (“Yeah, yeah” → id); harmless below
  the 120s threshold but don't trust single flagged lines.
- Keep new code in the existing module layout; prefer complete, runnable files.
