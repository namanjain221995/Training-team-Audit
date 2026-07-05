"""Thin OpenAI wrapper: text->JSON and image+text->JSON, plus a mock mode.

Works with both model families:
  - classic (gpt-4o...):      temperature=0 + seed for reproducible audit output
  - reasoning (gpt-5.x, o*):  reasoning_effort instead; some reject temperature/seed,
                              so on an "unsupported parameter" error the call is
                              retried once without those knobs.

Mock mode (MOCK_OPENAI=true) returns clearly-labelled stub responses so the
whole pipeline can be exercised locally with no API key and no cost.
"""
import base64
import json
import mimetypes


class OpenAIClient:
    def __init__(self, api_key: str, model: str, mock: bool = False,
                 reasoning_effort: str = ""):
        self.model = model
        self.mock = mock
        self.reasoning_effort = (reasoning_effort or "").strip().lower()
        self._client = None
        if not mock:
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is empty. Set it in .env, or set MOCK_OPENAI=true "
                    "to test the pipeline without calling OpenAI."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

    # ── text -> JSON ────────────────────────────────────────────────────────
    def chat_json(self, system: str, user: str) -> dict:
        if self.mock:
            return {"_mock": True, "note": "MOCK_OPENAI is on; no real analysis performed."}
        return self._call([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    # ── images + text -> JSON ───────────────────────────────────────────────
    def vision_json(self, system: str, prompt: str, image_paths: list[str]) -> dict:
        if self.mock:
            return {"_mock": True, "note": "MOCK_OPENAI is on; vision analysis skipped.",
                    "frames_considered": len(image_paths)}
        content = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _data_url(p)}})
        return self._call([
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ])

    # ── internals ───────────────────────────────────────────────────────────
    def _call(self, messages: list) -> dict:
        kwargs = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": messages,
            # audit record: same transcript should score the same every run
            "temperature": 0,
            "seed": 42,
        }
        if self._is_reasoning_model():
            # thinking models take reasoning_effort; determinism knobs may be rejected
            if self.reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # some models reject temperature/seed — retry once without them
            if not _is_unsupported_param_error(exc):
                raise
            kwargs.pop("temperature", None)
            kwargs.pop("seed", None)
            resp = self._client.chat.completions.create(**kwargs)
        return _safe_json(resp.choices[0].message.content)

    def _is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _is_unsupported_param_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("unsupported" in text or "not supported" in text or "unknown parameter" in text) \
        and ("temperature" in text or "seed" in text or "param" in text)


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{b64}"


def _safe_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except Exception:
        return {"_parse_error": True, "raw": text[:2000]}
