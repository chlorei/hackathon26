"""The judge: frames of one attempt + its task -> a checked verdict from a vision model (Gemini or Claude).

Runs on the ctx.defer worker only (network). Gemini: Flash-Lite first (fast, cheap); on 429/503 or a timeout, Flash; when
no Gemini model answers, Claude Haiku if ANTHROPIC_API_KEY is set. A 429 (quota) is never waited on.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dreamkit import llm, reasons
from dreamkit.llm import Unavailable, load_env  # noqa: F401 - the judge's names, as tests and callers use them

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
CLAUDE_MODEL = "claude-sonnet-5-5"  # provider: claude
CLAUDE_FALLBACK = "claude-haiku-5-5"  # provider: gemini, when no Gemini model answers
MAX_REASKS = 2  # a reply that is not valid JSON of the schema is asked again this many times
PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text(encoding="utf-8")


def _ask_gemini(images: list[bytes], prompt: str, timeout_s: float, deadline: float) -> str:
    return llm.gemini(images, prompt, GEMINI_MODELS, timeout_s, deadline)[0]


def _ask_claude(images: list[bytes], prompt: str, timeout_s: float, deadline: float, model: str = CLAUDE_MODEL) -> str:
    return llm.claude(images, prompt, model, timeout_s, deadline, effort="low" if model == CLAUDE_FALLBACK else None)


def ask(images: list[bytes], prompt: str, provider: str, timeout_s: float, deadline: float) -> str:
    """One model call (with Gemini's fallback chain), finished before `deadline` (time.monotonic). Tests replace it."""
    load_env()
    if provider == "gemini":
        try:
            return _ask_gemini(images, prompt, timeout_s, deadline)
        except Unavailable as error:
            if not os.getenv("ANTHROPIC_API_KEY"):
                if error.quota:
                    raise Unavailable("quota", quota=True) from error
                raise
            print(f"[monitor-stats] {error}; asking {CLAUDE_FALLBACK}", file=sys.stderr)
            return _ask_claude(images, prompt, timeout_s, deadline, CLAUDE_FALLBACK)
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
          max_wait_s: float = 20.0, note: str = "") -> dict[str, Any]:
    """JPEG frames of one attempt + its task -> {success, score, reason, reason_code, frames}, within max_wait_s."""
    start = time.monotonic()
    deadline = start + max_wait_s
    prompt = (PROMPT.replace("{n}", str(len(images))).replace("{last}", str(len(images) - 1))
              .replace("{task}", task).replace("{codes}", ", ".join(reasons.FAILURES)))
    if note:
        prompt = prompt.replace("Decide whether", f"{note}\n\nDecide whether", 1)
    error = None
    for _ in range(1 + MAX_REASKS):
        p = prompt if error is None else f"{prompt}\n\nYour last reply was rejected: {error}. Return valid JSON only."
        text = ask(images, p, provider, timeout_s, deadline)  # network/key errors are not re-asked
        try:
            # model_s: from the first request to a valid verdict, fallbacks and re-asks included
            return {**validate(_parse(text), len(images)), "model_s": round(time.monotonic() - start, 2)}
        except (ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
            error = str(e)[:300]
    raise ValueError(f"the model gave no valid verdict in {1 + MAX_REASKS} tries: {error}")
