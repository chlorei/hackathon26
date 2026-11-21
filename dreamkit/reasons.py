"""The shared failure codes (PLUGIN_STANDARD.md §7). Every plugin that names a failure uses one of these; a new code
is added here, with its human label, never invented inside a plugin."""

from __future__ import annotations

## code -> human label (lowercase fragment, as the feed shows it)
REASONS: dict[str, str] = {
    "dropped": "dropped",
    "released_off_target": "released off target",
    "grasped_air": "grasped air",
    "not_reached": "not reached",
    "collision": "collision",
    "fell_off_table": "fell off table",
    "stalled": "stalled",
    "slow_model": "slow model",
    "camera_failure": "camera failure",
    "timeout": "timeout",
    # how an attempt ended without a physical failure
    "stopped": "stopped by the user",
    "model_error": "model error",
    "error": "error in the loop",
    "marked_failed": "marked failed by a person",
}
## Codes for a failure of the task itself (what failure-monitor diagnoses and recovery can act on).
FAILURES = ("dropped", "released_off_target", "grasped_air", "not_reached", "collision",
            "fell_off_table", "stalled", "slow_model", "camera_failure", "timeout")
## Codes written by runs before the standard, still read by the panel.
LEGACY = {"stopped_by_plugin": "camera_failure", "missed_target": "released_off_target"}


def label(code: str | None) -> str:
    if code is None:
        return ""
    return REASONS.get(code) or REASONS.get(LEGACY.get(code, ""), code.replace("_", " "))


def is_reason(code: str) -> bool:
    return code in REASONS
