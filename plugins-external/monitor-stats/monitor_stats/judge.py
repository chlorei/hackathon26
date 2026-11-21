"""The judge: frames of one attempt + its task -> a checked verdict from a vision model (Gemini or Claude).

Runs on the ctx.defer worker only (network). Gemini: Flash-Lite first (fast, cheap); on 503 "high demand" or a timeout, Flash.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dreamkit import reasons

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
GEMINI_RETRIES = 1  # per model, for 429/500; 503 and 504 (time already spent) move on to the next model
MIN_CALL_S = 3.0  # less time than this left before the deadline: do not start another call
CLAUDE_MODEL = "claude-sonnet-5-5"
MAX_REASKS = 2  # a reply that is not valid JSON of the schema is asked again this many times
PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text(encoding="utf-8")


def _ask_gemini(images: list[bytes], prompt: str, timeout_s: float, deadline: float) -> str:
    import httpx2  # google-genai's HTTP client  # noqa: PLC0415
    from google import genai  # noqa: PLC0415
    from google.genai import errors, types  # noqa: PLC0415

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
    failures = []
    for model in GEMINI_MODELS:
        left = deadline - time.monotonic()
        if left < MIN_CALL_S:
            failures.append("out of time")
            break
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
            timeout=int(min(timeout_s, left) * 1000),
            retry_options=types.HttpRetryOptions(attempts=1 + GEMINI_RETRIES, http_status_codes=[429, 500]),
        ))
        try:
            resp = client.models.generate_content(
                model=model, contents=[*parts, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
        except errors.APIError as e:
            if e.code not in (404, 429, 500, 503, 504):
                raise  # 400/401/403: the key or the request, another model will not help
            failures.append(f"{model}: {e.code}")
        except httpx2.TransportError as e:
            failures.append(f"{model}: {type(e).__name__}")
        else:
            if failures:
                print(f"[monitor-stats] answered by {model} after {', '.join(failures)}", file=sys.stderr)
            return str(resp.text)
    raise RuntimeError(f"no Gemini model available ({'; '.join(failures)})")


def _ask_claude(images: list[bytes], prompt: str, timeout_s: float, deadline: float) -> str:
    timeout_s = min(timeout_s, deadline - time.monotonic())
    if timeout_s < MIN_CALL_S:
        raise RuntimeError("out of time")
    import anthropic  # noqa: PLC0415

    content: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.standard_b64encode(b).decode()}} for b in images]
    content.append({"type": "text", "text": prompt})
    resp = anthropic.Anthropic(timeout=timeout_s).beta.messages.create(
        model=CLAUDE_MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # on a refusal the API reruns the request on a fallback model
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("claude declined to judge this attempt")
    return "".join(b.text for b in resp.content if b.type == "text")


def ask(images: list[bytes], prompt: str, provider: str, timeout_s: float, deadline: float) -> str:
    """One model call (with Gemini's fallback chain), finished before `deadline` (time.monotonic). Tests replace it."""
    if provider == "gemini":
        return _ask_gemini(images, prompt, timeout_s, deadline)
    if provider == "claude":
        return _ask_claude(images, prompt, timeout_s, deadline)
    raise ValueError(f"provider must be gemini or claude, not {provider!r}")


def _parse(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def _fragment(text: str) -> str:
    """The panel's style: a lowercase fragment without a trailing period."""
    text = " ".join(str(text).split()).rstrip(".")
    return text[:1].lower() + text[1:]


def validate(d: Any, n: int) -> dict[str, Any]:
    if not isinstance(d, dict) or not isinstance(d.get("success"), bool):
        raise ValueError("success must be a boolean")
    score = d.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
        raise ValueError("score must be a number 0-100")
    frames = d.get("frames")
    if not isinstance(frames, list) or len(frames) != n:
        raise ValueError(f"frames must have exactly {n} items")
    out = []
    for i, f in enumerate(frames):
        if not isinstance(f, dict) or f.get("status") not in ("ok", "error"):
            raise ValueError(f"frames[{i}].status must be ok or error")
        out.append({"index": i, "status": f["status"], "comment": _fragment(f.get("comment", ""))})
    success = d["success"]
    code = d.get("reason_code")
    return {
        "success": success,
        "score": round(score),
        "reason": None if success else _fragment(d.get("failure_reason") or "no reason given"),
        "reason_code": code if not success and code in reasons.FAILURES else None,
        "frames": out,
    }


def judge(images: list[bytes], task: str, provider: str = "gemini", timeout_s: float = 10.0,
          max_wait_s: float = 20.0) -> dict[str, Any]:
    """JPEG frames of one attempt + its task -> {success, score, reason, reason_code, frames}, within max_wait_s."""
    deadline = time.monotonic() + max_wait_s
    prompt = (PROMPT.replace("{n}", str(len(images))).replace("{last}", str(len(images) - 1))
              .replace("{task}", task).replace("{codes}", ", ".join(reasons.FAILURES)))
    error = None
    for _ in range(1 + MAX_REASKS):
        p = prompt if error is None else f"{prompt}\n\nYour last reply was rejected: {error}. Return valid JSON only."
        text = ask(images, p, provider, timeout_s, deadline)  # network/key errors are not re-asked
        try:
            return validate(_parse(text), len(images))
        except (ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
            error = str(e)[:300]
    raise ValueError(f"the model gave no valid verdict in {1 + MAX_REASKS} tries: {error}")
