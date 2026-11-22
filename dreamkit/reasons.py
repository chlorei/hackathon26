"""The shared failure codes (PLUGIN_STANDARD.md §7). Every plugin that names a failure uses one of these; a new code
is added here, with its human label, never invented inside a plugin."""

from __future__ import annotations

## code -> human label (lowercase fragment, as the feed shows it)
REASONS: dict[str, str] = {
    "dropped": "dropped",
    "released_off_target": "released off target",
    "grasped_air": "grasped air",
    "released_early": "let go too early",
    "not_reached": "not reached",
    "collision": "collision",
    "fell_off_table": "fell off table",
    "stalled": "stalled",
    "still_held": "still held at the end",
    "slow_model": "slow model",
    "camera_failure": "camera failure",
    "timeout": "timeout",
    # how an attempt ended without a physical failure
    "stopped": "stopped by the user",
    "model_error": "model error",
    "error": "error in the loop",
    "marked_failed": "marked failed by a person",
    "judged_failed": "judged failed from the camera",  # vision decided (no world), with no more precise code
    "budget_exhausted": "credit budget used up",  # budget-guard: the run stops safely before the money runs out
}
## Codes for a failure of the task itself (what failure-monitor diagnoses and recovery can act on).
FAILURES = ("dropped", "released_early", "released_off_target", "grasped_air", "not_reached", "collision",
            "fell_off_table", "stalled", "still_held", "slow_model", "camera_failure", "timeout")
## Codes written by runs before the standard, still read by the panel.
LEGACY = {"stopped_by_plugin": "camera_failure", "missed_target": "released_off_target",
          # DreamGym's labels before it used these codes
          "untouched": "not_reached", "misplaced": "released_off_target", "not_settled": "timeout",
          "api_error": "model_error", "ik_error": "error", "invalid_action": "error"}


def code(value: str | None) -> str | None:
    """The shared code for a value that may be an old label (LEGACY)."""
    if value is None:
        return None
    return value if value in REASONS else LEGACY.get(value, value)


def label(code: str | None) -> str:
    if code is None:
        return ""
    return REASONS.get(code) or REASONS.get(LEGACY.get(code, ""), code.replace("_", " "))


def is_reason(code: str) -> bool:
    return code in REASONS
