"""Combine the Zoom VTT with the techsara-whisper transcript into ONE labeled transcript.

Each source is good at something different:
  Zoom VTT         — WHO spoke (speaker names), but Zoom's speech recognition is weak
  techsara-whisper — WHAT was said (more accurate text), but no speaker labels

Combined = Whisper's text, labeled with Zoom's speaker for the same moment:
  - a Whisper segment takes the speaker whose Zoom cues overlap it most in time;
    if two speakers share it (neither >= 70%), its words are split between them
    in proportion to time
  - a Whisper segment with no Zoom cue nearby takes the nearest cue's speaker
    (within 3s); a SHORT one (<= 3 words) with nothing nearby is dropped as a
    Whisper hallucination ("Thank you." over silence is a known Whisper habit)
  - a Zoom cue that Whisper did not cover at all (missed speech, or a failed
    Whisper chunk) is kept as-is, marked source="zoom"

Every combined segment records its `source` (whisper|zoom) so a reviewer can see
where each line came from; `stats` measures how far the two transcripts agree.
"""
import json
import re
from difflib import SequenceMatcher

DOMINANT_SHARE = 0.7          # one speaker owns a Whisper segment above this time share
NEAREST_CUE_SEC = 3.0         # unmatched Whisper text borrows a speaker this close
HALLUCINATION_MAX_WORDS = 3   # unmatched Whisper text this short (nobody speaking) is dropped
ZOOM_ONLY_MAX_COVER = 0.2     # a Zoom cue < 20% covered by Whisper is kept from Zoom


def combine(vtt: list[dict], whisper: list[dict]) -> tuple[list[dict], dict]:
    """vtt / whisper: [{speaker?, start, end, text}] -> (combined segments, stats)"""
    cues = sorted((c for c in vtt if (c.get("text") or "").strip()), key=lambda c: c["start"])
    wsegs = sorted((w for w in whisper if (w.get("text") or "").strip()), key=lambda w: w["start"])

    out, dropped, split = [], 0, 0
    for w in wsegs:
        ov = [(c, _overlap(w, c)) for c in _near(cues, w["start"], w["end"])]
        ov = [(c, o) for c, o in ov if o > 0]
        if not ov:
            near = _nearest(cues, w)
            if near is None and len(w["text"].split()) <= HALLUCINATION_MAX_WORDS:
                dropped += 1
                continue
            out.append(_seg(near["speaker"] if near else None, w["start"], w["end"], w["text"],
                            "whisper", "nearest_cue" if near else "none"))
            continue

        by_spk: dict = {}
        for c, o in ov:
            by_spk[c.get("speaker")] = by_spk.get(c.get("speaker"), 0.0) + o
        top = max(by_spk, key=by_spk.get)
        if len(by_spk) == 1 or by_spk[top] / sum(by_spk.values()) >= DOMINANT_SHARE:
            out.append(_seg(top, w["start"], w["end"], w["text"], "whisper", "zoom"))
            continue

        # two+ speakers inside one Whisper segment: split its words by time share
        split += 1
        runs = []   # [speaker, start, end] in time order, same-speaker cues merged
        for c, _ in sorted(ov, key=lambda co: co[0]["start"]):
            s, e = max(w["start"], c["start"]), min(w["end"], c["end"])
            if runs and runs[-1][0] == c.get("speaker"):
                runs[-1][2] = max(runs[-1][2], e)
            else:
                runs.append([c.get("speaker"), s, e])
        words = w["text"].split()
        total = sum(max(0.01, e - s) for _, s, e in runs)
        pos, acc = 0, 0.0
        for i, (spk, s, e) in enumerate(runs):
            if i == len(runs) - 1:
                cut = len(words)
            else:
                acc += max(0.01, e - s)
                cut = _sentence_cut(words, round(len(words) * acc / total), lo=pos)
            chunk = words[pos:cut]
            pos = max(pos, cut)
            if chunk:
                out.append(_seg(spk, s, e, " ".join(chunk), "whisper", "zoom_split"))

    # Zoom cues Whisper did not cover (missed speech / failed chunk): keep from Zoom
    zoom_only = 0
    for c in cues:
        dur = max(0.5, c["end"] - c["start"])
        covered = sum(_overlap(c, w) for w in _near(wsegs, c["start"], c["end"]))
        if covered / dur < ZOOM_ONLY_MAX_COVER and len(c["text"].split()) >= 2:
            out.append(_seg(c.get("speaker"), c["start"], c["end"], c["text"], "zoom", "zoom"))
            zoom_only += 1

    out.sort(key=lambda s: s["start"])
    return out, _stats(cues, wsegs, out, dropped, split, zoom_only)


# ── stats ────────────────────────────────────────────────────────────────────
def _stats(cues, wsegs, out, dropped, split, zoom_only) -> dict:
    speech = sum(max(0.0, s["end"] - s["start"]) for s in out)
    labeled = sum(max(0.0, s["end"] - s["start"]) for s in out if s["speaker"])
    # text agreement: per Zoom cue, its words vs the Whisper words in the same window
    num = den = 0.0
    for c in cues:
        cw = _words(c["text"])
        if len(cw) < 3:
            continue
        ww = _words(" ".join(w["text"] for w in _near(wsegs, c["start"], c["end"])
                             if _overlap(c, w) > 0))
        if not ww:
            continue                      # Whisper silent here -> counted in zoom_only
        dur = max(0.5, c["end"] - c["start"])
        num += dur * SequenceMatcher(None, cw, ww, autojunk=False).ratio()
        den += dur
    return {
        "vtt_cues": len(cues),
        "whisper_segments": len(wsegs),
        "combined_segments": len(out),
        "from_whisper": sum(1 for s in out if s["source"] == "whisper"),
        "from_zoom_only": zoom_only,
        "split_between_speakers": split,
        "dropped_whisper_hallucinations": dropped,
        "unlabeled_segments": sum(1 for s in out if not s["speaker"]),
        "labeled_speech_pct": round(100 * labeled / speech, 1) if speech else None,
        "vtt_words": sum(len(c["text"].split()) for c in cues),
        "whisper_words": sum(len(w["text"].split()) for w in wsegs),
        "text_agreement_pct": round(100 * num / den, 1) if den else None,
        "note": ("text = techsara-whisper, speaker labels = Zoom VTT; text_agreement_pct = "
                 "how closely Zoom's own text matches Whisper's in the same moments"),
    }


# ── output files ─────────────────────────────────────────────────────────────
def write_vtt(segments: list[dict], path: str) -> None:
    lines = ["WEBVTT", ""]
    for i, s in enumerate(segments, 1):
        who = f"{s['speaker']}: " if s.get("speaker") else ""
        lines += [str(i), f"{_vtt_ts(s['start'])} --> {_vtt_ts(s['end'])}", f"{who}{s['text']}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)


# ── helpers ──────────────────────────────────────────────────────────────────
def _seg(speaker, start, end, text, source, speaker_source):
    return {"speaker": speaker, "start": round(start, 2), "end": round(end, 2),
            "text": text.strip(), "source": source, "speaker_source": speaker_source}


def _sentence_cut(words: list[str], target: int, lo: int, reach: int = 4) -> int:
    """Cut index near `target` (time-proportional), moved to just after the closest
    word ending a sentence/clause within `reach` words, so a speaker change does not
    land mid-sentence. Falls back to the time-proportional point."""
    target = max(lo, min(len(words), target))
    best = None
    for k in range(max(lo + 1, target - reach), min(len(words), target + reach) + 1):
        if words[k - 1][-1:] in ".?!,;:" and (best is None or abs(k - target) < abs(best - target)):
            best = k
    return best if best is not None else target


def _overlap(a, b) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _near(items, start, end, pad=30.0):
    """Items that could overlap [start,end] (sorted input; long cues are rare, 30s pad)."""
    return [x for x in items if x["start"] < end + pad and x["end"] > start - pad]


def _nearest(cues, w):
    best, gap = None, NEAREST_CUE_SEC
    for c in _near(cues, w["start"], w["end"], pad=NEAREST_CUE_SEC):
        g = max(c["start"] - w["end"], w["start"] - c["end"], 0.0)
        if g <= gap:
            best, gap = c, g
    return best


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _vtt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
