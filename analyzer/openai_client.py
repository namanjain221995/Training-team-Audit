"""Thin OpenAI wrapper: text->JSON and image+text->JSON, plus a mock mode.

Works with both model families:
  - classic (gpt-4o...):      temperature=0 + seed for reproducible audit output
  - reasoning (gpt-5.x, o*):  reasoning_effort instead; some reject temperature/seed,
                              so on an "unsupported parameter" error the call is
                              retried without those knobs.

OpenAI-compatible servers (e.g. the self-hosted TechSara API, techsara-35b) are
selected with LLM_PROVIDERS=techsara (see make_clients). Servers that reject fields they don't support (response_format,
seed, reasoning_effort) are handled the same way — the rejected field is dropped,
remembered for the rest of the run, and the call retried. JSON is then enforced by
the prompt (every caller asks for "ONLY a JSON object") and extracted defensively.

Mock mode (MOCK_OPENAI=true) returns clearly-labelled stub responses so the
whole pipeline can be exercised locally with no API key and no cost.
"""
import base64
import json
import mimetypes
import os
import re

# optional request fields a compatible server may refuse -> how to strip each one
_OPTIONAL_FIELDS = ("response_format", "seed", "temperature", "reasoning_effort")


class OpenAIClient:
    def __init__(self, api_key: str, model: str, mock: bool = False,
                 reasoning_effort: str = "", base_url: str | None = None,
                 provider: str = "openai"):
        self.model = model
        self.mock = mock
        self.provider = provider
        self.reasoning_effort = (reasoning_effort or "").strip().lower()
        self._client = None
        self._dropped: set[str] = set()   # fields this server rejected; never resent
        if not mock:
            if not api_key:
                var = "TECHSARA_API_KEY" if provider == "techsara" else "OPENAI_API_KEY"
                raise RuntimeError(
                    f"{var} is empty. Set it in .env, or set MOCK_OPENAI=true "
                    "to test the pipeline without calling the model."
                )
            from openai import OpenAI, Timeout
            base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).strip() or None
            # long transcripts on a self-hosted model can take minutes: no read timeout
            # (the TechSara API keeps the connection alive every 15s)
            self._client = OpenAI(api_key=api_key, base_url=base_url,
                                  timeout=Timeout(None, connect=15.0), max_retries=3)

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
        out = self._call_once(messages)
        if out.get("_parse_error"):
            # no server-side JSON mode on some providers: one fresh attempt usually
            # comes back valid (the model is not deterministic even at temperature 0)
            print(f"      [{self.model}] reply was not valid JSON; retrying once")
            out = self._call_once(messages)
        return out

    def _call_once(self, messages: list) -> dict:
        kwargs = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": messages,
            # audit record: same transcript should score the same every run
            "temperature": 0,
            "seed": 42,
        }
        # thinking models take reasoning_effort; determinism knobs may be rejected.
        # Non-OpenAI servers get it whenever configured (dropped if they refuse it).
        if self.reasoning_effort and (self._is_reasoning_model() or self.provider != "openai"):
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        for field in self._dropped:
            _strip(kwargs, field)
        for _ in range(len(_OPTIONAL_FIELDS) + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if not _is_unsupported_param_error(exc):
                    raise
                # drop the field the server named; if it named none we know, drop
                # temperature+seed (the original OpenAI reasoning-model behaviour)
                named = [f for f in _OPTIONAL_FIELDS if f in str(exc).lower()
                         and f not in self._dropped] or \
                        [f for f in ("temperature", "seed") if f not in self._dropped]
                if not named:
                    raise
                for field in named:
                    self._dropped.add(field)
                    _strip(kwargs, field)
                print(f"      model server rejected {named}; retrying without")
        else:
            raise RuntimeError("model server kept rejecting request fields")
        return _safe_json(resp.choices[0].message.content)

    def _is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return m.startswith(("gpt-5", "o1", "o3", "o4"))


def make_clients(cfg) -> list[OpenAIClient]:
    """One client per entry in cfg.llm_providers, in order (first = primary)."""
    clients = []
    for p in cfg.llm_providers:
        if p == "openai":
            clients.append(OpenAIClient(cfg.openai_api_key, cfg.openai_model, cfg.mock_openai,
                                        reasoning_effort=cfg.openai_reasoning_effort,
                                        base_url=cfg.openai_base_url or None, provider="openai"))
        elif p == "techsara":
            if not (cfg.techsara_api_key or cfg.mock_openai):
                raise RuntimeError("LLM_PROVIDERS includes techsara but TECHSARA_API_KEY is empty")
            clients.append(OpenAIClient(cfg.techsara_api_key, cfg.techsara_model, cfg.mock_openai,
                                        reasoning_effort=cfg.techsara_reasoning_effort,
                                        base_url=cfg.techsara_base_url, provider="techsara"))
    return clients


def _strip(kwargs: dict, field: str) -> None:
    if field == "reasoning_effort":
        kwargs.pop("extra_body", None)
    else:
        kwargs.pop(field, None)


def _is_unsupported_param_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("unsupported" in text or "not supported" in text or "unknown parameter" in text) \
        and ("temperature" in text or "seed" in text or "param" in text
             or "response_format" in text or "reasoning_effort" in text or "field" in text)


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
        pass
    # no JSON mode on some servers: take the outermost {...} if prose surrounds it,
    # then repair the one malformation seen in practice (see _merge_orphan_strings)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        body = text[start:end + 1]
        for candidate in (body, _merge_orphan_strings(body)):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return {"_parse_error": True, "raw": text[:2000]}


_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]:,]|[^{}\[\]:,"\s]+|\s+', re.S)


def _merge_orphan_strings(text: str) -> str:
    """Repair a multi-paragraph value the model emitted as separate strings:

        {"overview": "para 1", "para 2", "para 3", "timeline": [...]}
     -> {"overview": "para 1\\n\\npara 2\\n\\npara 3", "timeline": [...]}

    Only inside an OBJECT, only strings right after a key's string value that are
    not themselves followed by ':' (so a real next key is never swallowed). Arrays
    of strings are left alone."""
    toks = _TOKEN.findall(text)
    if "".join(toks) != text:
        return text
    sig = [i for i, t in enumerate(toks) if not t.isspace()]
    out, drop, stack = list(toks), set(), []
    k = 0
    while k < len(sig):
        t = toks[sig[k]]
        if t in ("{", "["):
            stack.append(t)
        elif t in ("}", "]"):
            if stack:
                stack.pop()
        elif t.startswith('"') and stack and stack[-1] == "{" and k > 0 and toks[sig[k - 1]] == ":":
            target = sig[k]
            while (k + 2 < len(sig) and toks[sig[k + 1]] == "," and toks[sig[k + 2]].startswith('"')
                   and (k + 3 >= len(sig) or toks[sig[k + 3]] != ":")):
                out[target] = out[target][:-1] + "\\n\\n" + toks[sig[k + 2]][1:]
                drop.update(range(sig[k] + 1, sig[k + 2] + 1))
                k += 2
        k += 1
    return "".join(t for i, t in enumerate(out) if i not in drop)
