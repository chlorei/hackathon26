"""Reads the object and the target out of a natural-language task."""

from __future__ import annotations

import re

from .sim import FRUITS, TARGETS

SYNONYMS = {"apples": "apple", "oranges": "orange", "bananas": "banana", "dish": "plate"}


def parse(task: str) -> tuple[str | None, str | None]:
    """`put apple into the bowl` -> ("apple", "bowl"). Either part is None when the task does not name one."""
    words = [SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", task.lower())]
    obj = next((w for w in words if w in FRUITS), None)
    target = next((w for w in words if w in TARGETS), None)
    return obj, target
