"""techsara-whisper — speech-to-text on the self-hosted TechSara API.

The recording's audio is cut into WHISPER_CHUNK_SEC chunks (16 kHz mono WAV),
WHISPER_PARALLEL of them are transcribed at once, and the segments are shifted
back to absolute session time. Chunking is deliberate: one 2-hour upload never
answered in testing, while 10-minute chunks all came back.

A chunk that still fails after retries is reported in `failed_windows` instead
of failing the run — the combine step (combine.py) fills those windows from the
Zoom VTT, so a partial Whisper result still improves the transcript.
"""
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

CHUNK_TIMEOUT_SEC = 1800      # one chunk may queue behind other work on the GPU
CHUNK_ATTEMPTS = 3


def transcribe(video_path: str, cfg, work_dir: str) -> dict:
    """-> {"segments": [{start,end,text}], "duration_sec", "failed_windows": [[s,e]],
           "chunks", "elapsed_sec", "model"}"""
    import httpx

    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    duration = _duration(video_path)
    chunk = max(60, int(cfg.whisper_chunk_sec or 600))
    starts = list(range(0, int(duration) + 1, chunk))
    if starts and duration - starts[-1] < 1:      # nothing left in the last window
        starts.pop()
    url = cfg.techsara_base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {cfg.techsara_api_key}"}
    print(f"techsara-whisper: {duration / 60:.1f} min of audio in {len(starts)} chunk(s) "
          f"of {chunk // 60} min, {cfg.whisper_parallel} at a time...")

    def one(i_start):
        i, start = i_start
        path = os.path.join(work_dir, f"chunk_{i:03d}.wav")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                        "-t", str(chunk), "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                        path], check=True)
        t0 = time.time()
        last = None
        for attempt in range(1, CHUNK_ATTEMPTS + 1):
            try:
                with open(path, "rb") as fh:
                    r = httpx.post(url, headers=headers,
                                   files={"file": (os.path.basename(path), fh, "audio/wav")},
                                   data={"model": cfg.techsara_whisper_model,
                                         "response_format": "verbose_json",
                                         "timestamp_granularities[]": "segment"},
                                   timeout=httpx.Timeout(CHUNK_TIMEOUT_SEC, connect=15.0))
                r.raise_for_status()
                # a slow answer is preceded by keep-alive spaces
                body = json.loads(r.text.strip())
                segs = [{"start": round(start + float(s["start"]), 2),
                         "end": round(start + float(s["end"]), 2),
                         "text": (s.get("text") or "").strip()}
                        for s in body.get("segments") or [] if (s.get("text") or "").strip()]
                print(f"      chunk {i + 1}/{len(starts)} @{start // 60:>3} min: {len(segs)} segments "
                      f"({time.time() - t0:.0f}s)")
                return {"ok": True, "start": start, "segments": segs}
            except Exception as exc:
                last = exc
                print(f"      chunk {i + 1}/{len(starts)} attempt {attempt} failed: {exc!r}")
                time.sleep(5 * attempt)
        return {"ok": False, "start": start, "segments": [], "error": repr(last)}

    t_all = time.time()
    with ThreadPoolExecutor(max(1, int(cfg.whisper_parallel or 1))) as ex:
        results = list(ex.map(one, enumerate(starts)))
    shutil.rmtree(work_dir, ignore_errors=True)

    segments = sorted((s for r in results for s in r["segments"]), key=lambda s: s["start"])
    failed = [[r["start"], min(duration, r["start"] + chunk)] for r in results if not r["ok"]]
    print(f"techsara-whisper: {len(segments)} segments in {time.time() - t_all:.0f}s"
          + (f"; {len(failed)} chunk(s) FAILED (Zoom VTT covers those windows)" if failed else ""))
    return {"model": cfg.techsara_whisper_model, "duration_sec": round(duration, 1),
            "chunks": len(starts), "failed_windows": failed,
            "elapsed_sec": round(time.time() - t_all, 1), "segments": segments}


def _duration(path: str) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", path])
    return float(out.decode().strip())
