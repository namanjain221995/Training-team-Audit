"""Configuration loaded from environment variables (+ optional training-temp.json).

Host paths are mounted into the container at fixed locations:
    ./input   -> /data/input
    ./output  -> /data/output
    ./trainer -> /data/trainer
    (model cache volume) -> /data/.cache

PRODUCTION PARITY: in production the EC2 worker receives day / trainer / candidate /
meeting_id from the SQS job (built from training-temp.json). For local testing, drop
that same `training-temp.json` in ./input and this module reads those fields from it —
overriding the manual .env values. If no temp file is present, the .env values are used.
"""
import json
import os
from dataclasses import dataclass, field

INPUT_DIR   = "/data/input"
OUTPUT_DIR  = "/data/output"
TRAINER_DIR = "/data/trainer"
CACHE_DIR   = "/data/.cache"
CONFIG_DIR  = os.environ.get("CONFIG_DIR", "/app/config")


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    """Like int(env), but a missing OR blank value falls back to the default."""
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    """Like float(env), but a missing OR blank value falls back to the default."""
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


@dataclass
class Config:
    # OpenAI (transcript reasoning)
    openai_api_key: str = ""
    openai_model:   str = "gpt-4o"
    # reasoning models (gpt-5.x / o*) only: none|low|medium|high|xhigh; blank = model default
    openai_reasoning_effort: str = ""
    mock_openai:    bool = False

    # Input
    video_path:          str = ""
    transcript_path:     str | None = None
    trainer_image_path:  str | None = None
    training_temp_path:  str | None = None

    # Session (may be overridden by training-temp.json)
    day_number:     int = 1
    trainer_name:   str = ""
    candidate_name: str = ""
    meeting_id:     str | None = None
    day_step_name:  str | None = None
    s3_prefix:      str | None = None
    s3_bucket:      str | None = None
    config_source:  str = ".env"

    # Video analysis (local CPU models)
    enable_video:        bool = True
    frame_fps:           float = 1.0
    whisper_model:       str = "base"
    save_frames:         bool = True
    model_cache_dir:     str = CACHE_DIR
    identity_sample_sec: int = 20

    # Annotated "what the models saw" video (local rendering + ffmpeg encode)
    save_analysis_video: bool = True

    # Output
    output_path: str = ""
    frames_dir:  str = ""
    proof_dir:   str = ""
    # Frames are extracted/analyzed here (container-local scratch): rapid small-file
    # writes to the Windows bind mount are slow and can fail with I/O errors mid-run.
    # They are bulk-copied to frames_dir once, at the end, if SAVE_EVIDENCE_FRAMES.
    frames_work_dir: str = "/tmp/frames"

    day_plan: dict = field(default_factory=dict)


def load_config() -> Config:
    video_file      = os.environ.get("VIDEO_FILE", "session.mp4").strip()
    transcript_file = os.environ.get("TRANSCRIPT_FILE", "").strip()
    trainer_file    = os.environ.get("TRAINER_IMAGE_FILE", "").strip()
    temp_file       = os.environ.get("TRAINING_TEMP_FILE", "").strip()
    output_file     = os.environ.get("OUTPUT_FILE", "result.json").strip()

    # training-temp.json: explicit filename, else auto-detect the default name in ./input
    temp_path = (os.path.join(INPUT_DIR, temp_file) if temp_file
                 else os.path.join(INPUT_DIR, "training-temp.json"))

    cfg = Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o").strip(),
        openai_reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "").strip(),
        mock_openai=_bool("MOCK_OPENAI", False),

        video_path=os.path.join(INPUT_DIR, video_file),
        transcript_path=os.path.join(INPUT_DIR, transcript_file) if transcript_file else None,
        trainer_image_path=os.path.join(TRAINER_DIR, trainer_file) if trainer_file else None,
        training_temp_path=temp_path,

        day_number=_int("DAY_NUMBER", 1),
        trainer_name=os.environ.get("TRAINER_NAME", "").strip(),
        candidate_name=os.environ.get("CANDIDATE_NAME", "").strip(),

        enable_video=_bool("ENABLE_VIDEO_ANALYSIS", True),
        frame_fps=_float("FRAME_FPS", 1.0),
        whisper_model=os.environ.get("WHISPER_MODEL", "base").strip(),
        save_frames=_bool("SAVE_EVIDENCE_FRAMES", True),
        identity_sample_sec=_int("IDENTITY_SAMPLE_SEC", 20),

        save_analysis_video=_bool("SAVE_ANALYSIS_VIDEO", True),

        output_path=os.path.join(OUTPUT_DIR, output_file),
        frames_dir=os.path.join(OUTPUT_DIR, "frames"),
        proof_dir=os.path.join(OUTPUT_DIR, "proof"),
    )

    _apply_training_temp(cfg)

    with open(os.path.join(CONFIG_DIR, "day_plan.json"), encoding="utf-8") as fh:
        cfg.day_plan = json.load(fh)

    return cfg


def _apply_training_temp(cfg: Config) -> None:
    """Overlay day / trainer / candidate / meeting_id from training-temp.json if present."""
    if not (cfg.training_temp_path and os.path.exists(cfg.training_temp_path)):
        return
    try:
        with open(cfg.training_temp_path, encoding="utf-8") as fh:
            t = json.load(fh)
    except Exception as exc:
        print(f"Could not read training-temp.json ({exc}); falling back to .env values")
        return

    cfg.config_source = "training-temp.json"

    if t.get("day") is not None:
        try:
            cfg.day_number = int(t["day"])
        except (TypeError, ValueError):
            print(f"training-temp.json 'day' not an int ({t.get('day')!r}); keeping DAY_NUMBER={cfg.day_number}")
    if t.get("trainer"):
        cfg.trainer_name = _humanize(str(t["trainer"]))
    if t.get("candidate"):
        cfg.candidate_name = _humanize(str(t["candidate"]))

    cfg.meeting_id    = t.get("meeting_id")
    cfg.day_step_name = t.get("day_step_name")
    cfg.s3_prefix     = t.get("prefix")
    cfg.s3_bucket     = t.get("bucket")

    print(f"Loaded training-temp.json: meeting_id={cfg.meeting_id} day={cfg.day_number} "
          f"trainer={cfg.trainer_name!r} candidate={cfg.candidate_name!r}")
    if t.get("day") is None:
        print("  NOTE: 'day' was null in training-temp.json (Salesforce lookup likely failed) — "
              f"using DAY_NUMBER={cfg.day_number} from .env instead.")


def _humanize(name: str) -> str:
    """Folder-style 'Divya_Prajapati' -> 'Divya Prajapati' for speaker matching / display."""
    return name.replace("_", " ").replace("  ", " ").strip()


def day_sections(cfg: Config) -> dict:
    return cfg.day_plan.get(str(cfg.day_number), {"title": f"Day {cfg.day_number}", "sections": []})
