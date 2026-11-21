"""One call to a model, for whatever in dreamkit asks one: monitor-stats' judge, the panel's Create tab.

Keys come from the environment or the nearest .env (the loop's folder or above), never from code. Gemini models are
tried in the order given; a 429 (quota), 503 or timeout moves straight on to the next one, nothing waits. When no
Gemini model answers, Claude is asked if ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

GEMINI_RETRIES = 1  # per model, for 500 only
MIN_CALL_S = 3.0  # less time than this left before the deadline: do not start another call


class Unavailable(RuntimeError):
    """No model answered; `quota` when every one of them said 429."""

    def __init__(self, message: str, quota: bool = False):
        super().__init__(message)
        self.quota = quota


def load_env(start: Path | None = None) -> None:
    """KEY=value lines of the nearest .env (the loop's folder or above) into os.environ; what is already set wins."""
    folder = (start or Path.cwd()).resolve()
    for d in (folder, *folder.parents):
        if (d / ".env").is_file():
            for line in (d / ".env").read_text(encoding="utf-8").splitlines():
                key, sep, value = line.strip().removeprefix("export ").partition("=")
                if sep and key.strip() and not key.lstrip().startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            return


def gemini(images: list[bytes], prompt: str, models: list[str], timeout_s: float, deadline: float,
           max_tokens: int | None = None) -> tuple[str, str]:
    """(text, model): the first of `models` that answers, as JSON. Raises Unavailable."""
    import httpx2  # google-genai's HTTP client  # noqa: PLC0415
    from google import genai  # noqa: PLC0415
    from google.genai import errors, types  # noqa: PLC0415

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
    failures = []
    for model in models:
        left = deadline - time.monotonic()
        if left < MIN_CALL_S:
            failures.append("out of time")
            break
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
            timeout=int(min(timeout_s, left) * 1000),
            retry_options=types.HttpRetryOptions(attempts=1 + GEMINI_RETRIES, http_status_codes=[500]),
        ))
        try:
            resp = client.models.generate_content(
                model=model, contents=[*parts, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   max_output_tokens=max_tokens))
        except errors.APIError as e:
            if e.code not in (404, 429, 500, 503, 504):
                raise  # 400/401/403: the key or the request, another model will not help
            failures.append(f"{model}: {e.code}")
        except httpx2.TransportError as e:
            failures.append(f"{model}: {type(e).__name__}")
        else:
            if failures:
                print(f"[dreamkit.llm] answered by {model} after {', '.join(failures)}", file=sys.stderr)
            return str(resp.text), model
    quota = bool(failures) and all(f.endswith(": 429") for f in failures)
    raise Unavailable(f"no Gemini model available ({'; '.join(failures)})", quota=quota)


def claude(images: list[bytes], prompt: str, model: str, timeout_s: float, deadline: float,
           max_tokens: int = 16000, effort: str | None = None) -> str:
    timeout_s = min(timeout_s, deadline - time.monotonic())
    if timeout_s < MIN_CALL_S:
        raise RuntimeError("out of time")
    import anthropic  # noqa: PLC0415

    content: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.standard_b64encode(b).decode()}} for b in images]
    content.append({"type": "text", "text": prompt})
    client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)  # a 429 is not waited on
    messages = [{"role": "user", "content": content}]
    extra: dict[str, Any] = {"output_config": {"effort": effort}} if effort else {}
    if model.startswith("claude-haiku"):  # Haiku has no server-side refusal fallback
        resp = client.messages.create(model=model, max_tokens=max_tokens, messages=messages, **extra)
    else:
        resp = client.beta.messages.create(
            model=model, max_tokens=max_tokens, messages=messages, **extra,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # on a refusal the API reruns the request on a fallback model
        )
    if resp.stop_reason == "refusal":
        raise RuntimeError("claude declined the request")
    return "".join(b.text for b in resp.content if b.type == "text")
