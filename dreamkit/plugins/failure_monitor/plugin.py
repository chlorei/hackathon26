"""failure-monitor: why an attempt failed and when, from what a real arm reports (PLUGIN_STANDARD.md).

Only the arm's own signals (ctx.robot): each gripper's opening and its command, the joints, tracking error (command
vs measured), joint speed, the model's timing, and the live verdicts other plugins call on the cameras.

- grasped_air: the hand closed and the fingers met (nothing between them)
- dropped: it was holding something, then the fingers closed in during the carry without being told to open
- released_early: it was holding something and the policy opened the hand while the arm was still moving fast
- collision: the arm cannot follow a command that is not moving (blocked)
- stalled: no joint moved for a while (not after a release: then it is resting)
- slow_model: the model's round trips are slow (warned at once; the reason if the attempt failed for nothing else)
- camera_failure: from camera-check's verdict

Live: milestones on the timeline, a verdict as soon as a failure is certain (recovery acts on it), and the Live
signals (gripper opening and command per hand, tracking error per hand, joint speed, model round trip).
At the end: the first cause in time, written to ctx.result (reason, at_s, details, message, candidates, timeline).
Nothing found means the outcome is unknown, not success: vision-judge or a person decides.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from dreamkit import reasons
from dreamkit.plugin import Plugin, PluginContext
from dreamkit.standard import Manifest

SIDES = ("left", "right")
## Tie-break when two causes happen within tie_s of each other: the more specific one wins.
PRIORITY = ("camera_failure", "dropped", "released_early", "grasped_air", "collision", "stalled", "slow_model")
MESSAGE = {
    "grasped_air": "grasped air (the fingers closed on nothing)",
    "dropped": "dropped the object (the gripper never opened)",
    "released_early": "let go too early (opened the hand while still moving)",
    "collision": "collision (the arm could not follow its command)",
    "stalled": "arms stalled",
    "slow_model": "model too slow",
    "camera_failure": "camera failure",
}
BAD = {"separated", "collision", "stalled", "early"}


class Params(BaseModel):
    close_cmd: float = Field(0.3, ge=0, le=1, description="Gripper command below which the hand is closing",
                             json_schema_extra={"unit": ""})
    release_cmd: float = Field(0.85, ge=0, le=1, description="The hand let go only if its command reaches this",
                               json_schema_extra={"unit": ""})
    empty_below: float = Field(0.05, ge=0, le=1, description="A closed gripper measuring less than this holds nothing",
                               json_schema_extra={"unit": ""})
    hold_gap: float = Field(0.15, gt=0, le=1, description="Fingers stopped this far above their command are holding "
                            "something", json_schema_extra={"unit": ""})
    grip_settle_s: float = Field(0.6, gt=0, le=5, description="How long the fingers get to close before judging",
                                 json_schema_extra={"unit": "s"})
    release_window_s: float = Field(1.0, gt=0, le=5, description="Fingers closed in this long without a release is "
                                    "a drop", json_schema_extra={"unit": "s"})
    early_speed_rad_s: float = Field(0.6, gt=0, le=10, description="Opening the hand while a joint moves faster than "
                                     "this lets go too early", json_schema_extra={"unit": "rad/s"})
    track_err_rad: float = Field(0.12, gt=0, le=3, description="An arm this far from a steady command is blocked "
                                 "(a collision)", json_schema_extra={"unit": "rad"})
    steady_cmd_rad: float = Field(0.02, gt=0, le=1, description="A command moving less than this per step is steady",
                                  json_schema_extra={"unit": "rad"})
    blocked_s: float = Field(0.3, gt=0, le=5, description="Blocked this long is a collision",
                             json_schema_extra={"unit": "s"})
    stall_speed_rad_s: float = Field(0.03, gt=0, le=1, description="Every joint slower than this is not moving",
                                     json_schema_extra={"unit": "rad/s"})
    stall_s: float = Field(3.0, gt=0, le=60, description="Not moving this long is a stall",
                           json_schema_extra={"unit": "s"})
    slow_ms: float = Field(700.0, gt=0, le=10000, description="A model round trip above this is slow: warned at once",
                           json_schema_extra={"unit": "ms"})
    tie_s: float = Field(1.0, ge=0, le=10, description="Causes this close in time count as simultaneous",
                         json_schema_extra={"unit": "s"})
    live_verdicts: bool = Field(True, description="Call failures mid-attempt (recovery acts on them)")
    trace_every: int = Field(2, ge=1, le=20, description="Keep every Nth step in the saved trace",
                             json_schema_extra={"unit": ""})


def _fresh_grip() -> dict[str, dict[str, Any]]:
    return {s: {"closed_since": None, "judged": False, "holding": False, "gap_closed": None, "blocked_since": None}
            for s in SIDES}


class FailureMonitor(Plugin):
    """Explains failures: why an attempt failed and at which second."""

    name = "failure-monitor"
    Params = Params
    manifest = Manifest(group="core", icon="ScanSearch", capability="observe",
                        hooks=["on_episode_start", "after_model", "on_step", "on_episode_end"], step_rate_hz=20,
                        needs=["robot"], ui="ui/index.html")

    # -- recording ---------------------------------------------------------------------------------------------
    def on_episode_start(self, ctx: PluginContext) -> None:
        self.rtts: list[tuple[float, float]] = []
        self.mock_calls = False
        self.active = ctx.robot.available
        self.source = "robot"
        self.verdicts: list[dict[str, Any]] = []
        self.segment, self.verdict_given = 0, False
        self.still_since: float | None = 0.0
        self.stall: dict[str, Any] | None = None
        self.stall_warned = False
        self.milestones: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.grip = _fresh_grip()
        self.let_go = False  # the hand opened on purpose while holding something
        if not self.active:
            ctx.emit("info", "no robot state (client.step(state=...)): failure diagnosis is off")

    def _milestone(self, ctx: PluginContext, kind: str, level: str, message: str, emit: bool = True,
                   **data: Any) -> None:
        self.milestones.append({"t": round(ctx.t, 2), "kind": kind, "message": message})
        ctx.ui.mark(ctx.t, message, "warn" if kind in BAD else "info")
        if emit:  # the feed gets at most one event a second from on_step (§7); the timeline gets every mark
            ctx.emit(level, message, kind=kind, **data)

    def after_model(self, ctx: PluginContext) -> None:
        """The model's round trip: every call, a warning when slow."""
        timing = ctx.timing or {}
        if timing.get("mock"):
            self.mock_calls = True
            return
        if timing.get("rtt_ms") is None:
            return
        rtt = float(timing["rtt_ms"])
        self.rtts.append((ctx.t, rtt))
        ctx.ui.series("rtt", ctx.t, rtt, unit="ms")
        if rtt > self.params.slow_ms:
            ctx.emit("warn", f"model round trip {rtt:.0f} ms, slow", rtt_ms=rtt)

    def _latency(self) -> dict[str, Any]:
        if not self.rtts:
            return {"mock": True} if self.mock_calls else {}
        a = np.asarray([r for _, r in self.rtts])
        return {"mean_ms": round(float(a.mean()), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
                "max_ms": round(float(a.max()), 1), "calls": len(a)}

    def on_step(self, ctx: PluginContext) -> None:
        if not self.active:
            return
        if ctx.store.get("segment", 0) != self.segment:
            self._new_segment(ctx)
        speed = max(ctx.robot.speed(s) for s in SIDES)
        self._stall(ctx, speed)
        self._gripper_and_arm(ctx, speed)

    def _stall(self, ctx: PluginContext, speed: float) -> None:
        p = self.params
        if speed < p.stall_speed_rad_s:
            if self.still_since is None:
                self.still_since = ctx.t
            elif ctx.t - self.still_since >= p.stall_s and not self.stall_warned:
                self.stall_warned = True
                self.stall = {"at": round(self.still_since, 2)}
                if self.let_go:
                    self.stall = None  # it let go of the object and rests: finished, not stuck
                    return
                self._milestone(ctx, "stalled", "warn", f"arms have not moved for {p.stall_s:.0f} s")
                if p.live_verdicts and not self.verdict_given:
                    self._verdict(ctx, "stalled", self.stall["at"], "arms stalled")
        else:
            self.still_since, self.stall_warned = None, False

    def _verdict(self, ctx: PluginContext, reason: str, at: float, message: str, **details: Any) -> None:
        self.verdict_given = True
        self.verdicts.append({"reason": reason, "at": round(at, 2), "segment": self.segment, **details})
        ctx.verdict(reason, at, message, source="robot", **details)

    def _gripper_and_arm(self, ctx: PluginContext, speed: float) -> None:
        p, r = self.params, ctx.robot
        row: dict[str, Any] = {"t": round(ctx.t, 3), "speed": round(speed, 3)}
        for s in SIDES:
            g = self.grip[s]
            cmd, meas = r.grip_command(s), r.grip(s)
            err, steady = r.tracking_error(s), r.command_speed(s) < p.steady_cmd_rad
            row[f"cmd_{s}"] = cmd if cmd is not None else meas
            row[f"grip_{s}"] = meas
            row[f"err_{s}"] = err
            ctx.ui.series(f"grip_{s}", ctx.t, meas)
            ctx.ui.series(f"cmd_{s}", ctx.t, row[f"cmd_{s}"])
            ctx.ui.series(f"track_{s}", ctx.t, err, unit="rad")
            # collision: the arm cannot follow a command that is not moving
            if cmd is not None and err > p.track_err_rad and steady:
                g["blocked_since"] = ctx.t if g["blocked_since"] is None else g["blocked_since"]
                if ctx.t - g["blocked_since"] >= p.blocked_s and not g.get("blocked_told"):
                    g["blocked_told"] = True
                    self._milestone(ctx, "collision", "warn", f"{s} arm blocked: {err:.2f} rad from a steady command",
                                    side=s)
                    if p.live_verdicts and not self.verdict_given:
                        self._verdict(ctx, "collision", g["blocked_since"], f"{s} arm blocked (collision)", side=s)
            else:
                g["blocked_since"] = None
            if cmd is None:
                continue
            closed = cmd < p.close_cmd
            if closed and g["closed_since"] is None:
                g.update(closed_since=ctx.t, judged=False)
            if closed and not g["judged"] and ctx.t - g["closed_since"] >= p.grip_settle_s:
                g["judged"] = True
                if meas <= p.empty_below:
                    self._milestone(ctx, "grasp", "warn", f"{s} gripper closed on nothing", side=s)
                    if p.live_verdicts and not self.verdict_given:
                        self._verdict(ctx, "grasped_air", g["closed_since"], f"{s} gripper closed on nothing", side=s)
                elif meas - cmd >= p.hold_gap:
                    g["holding"], g["hold_width"] = True, meas
                    self._milestone(ctx, "grasp", "info", f"{s} gripper holding something ({meas:.2f} wide)", side=s)
            if not closed:
                g["closed_since"] = None
            if g["holding"]:
                if cmd >= p.release_cmd:  # the policy opened the hand: a release, not a drop
                    g["holding"], g["gap_closed"] = False, None
                    self.let_go = True
                    if speed > p.early_speed_rad_s:  # ... while the arm still swings: too early
                        self._milestone(ctx, "early", "warn", f"{s} gripper opened while moving ({speed:.1f} rad/s)",
                                        side=s)
                        if p.live_verdicts and not self.verdict_given:
                            self._verdict(ctx, "released_early", ctx.t, f"{s} hand opened while moving", side=s,
                                          speed_rad_s=round(speed, 2))
                    else:
                        self._milestone(ctx, "release", "info", f"{s} gripper opened ({cmd:.2f})", emit=False)
                elif meas - cmd < p.hold_gap / 2:  # nothing stops the fingers any more
                    g["gap_closed"] = ctx.t if g["gap_closed"] is None else g["gap_closed"]
                    if ctx.t - g["gap_closed"] >= p.release_window_s:  # and the hand was never told to open
                        g["holding"] = False
                        self._milestone(ctx, "separated", "warn",
                                        f"{s} gripper lost its object (gripper at {cmd:.2f}, not opened)", side=s)
                        if p.live_verdicts and not self.verdict_given:
                            self._verdict(ctx, "dropped", g["gap_closed"], f"dropped the object ({s} gripper never "
                                          "opened)", side=s)
                else:
                    g["gap_closed"] = None
        ctx.ui.series("speed", ctx.t, speed, unit="rad/s")
        # the older Live names, kept for panels that still read them
        ctx.ui.series("grip", ctx.t, min(row["grip_left"], row["grip_right"]))
        ctx.ui.series("track", ctx.t, max(row["err_left"], row["err_right"]), unit="rad")
        self.rows.append(row)

    def _new_segment(self, ctx: PluginContext) -> None:
        """The control plugin restarted the task mid-attempt: judge what happens next on its own."""
        self.segment = ctx.store.get("segment", 0)
        self.verdict_given, self.let_go = False, False
        self.grip = _fresh_grip()
        self.still_since, self.stall, self.stall_warned = None, None, False

    # -- the end of the attempt --------------------------------------------------------------------------------------
    def diagnose(self, ctx: PluginContext) -> dict[str, Any] | None:
        """The first cause in time (the more specific one when two are close); None: nothing the arm showed."""
        p = self.params
        # only the last try counts: an earlier one ended in a recovery, and the attempt went on from there
        cands = [{"reason": v["reason"], "at": v["at"], "details": {k: v[k] for k in ("side",) if k in v}}
                 for v in self.verdicts if v.get("segment", 0) == self.segment]
        if self.stall and not any(c["reason"] == "stalled" for c in cands):
            cands.append({"reason": "stalled", "at": self.stall["at"], "details": {}})
        for v in ctx.store.get("verdicts", []):  # what other plugins (camera-check) called, in this try
            if v.get("by") != "failure-monitor" and v.get("segment", 0) == self.segment:
                cands.append({"reason": v["reason"], "at": v["at_s"], "details": {
                    "by": v.get("by"), "message": v.get("message"), "source": v.get("source", "robot")}})
        if not cands:
            return None
        earliest = min(c["at"] for c in cands)
        best = min([c for c in cands if c["at"] - earliest <= p.tie_s],
                   key=lambda c: PRIORITY.index(c["reason"]) if c["reason"] in PRIORITY else 99)
        best["details"].setdefault("source", "robot")
        best["message"] = MESSAGE.get(best["reason"], reasons.label(best["reason"]))
        best["candidates"] = [{"reason": c["reason"], "at_s": c["at"]} for c in sorted(cands, key=lambda c: c["at"])]
        return best

    def _slow(self) -> dict[str, Any] | None:
        """slow_model when the mean round trip is slow: the reason if the attempt failed for nothing else."""
        if not self.rtts:
            return None
        mean = float(np.mean([r for _, r in self.rtts]))
        if mean <= self.params.slow_ms:
            return None
        first = next(t for t, r in self.rtts if r > self.params.slow_ms)
        return {"reason": "slow_model", "at_s": round(first, 2), "mean_rtt_ms": round(mean)}

    def on_episode_end(self, ctx: PluginContext) -> None:
        if not self.active:
            ctx.result["failure-monitor"] = {"latency": self._latency()}
            return
        self._save_trace(ctx)
        ctx.result["timeline"] = self.milestones
        if ctx.result.get("success") is True:
            return
        d = self.diagnose(ctx)
        evidence = self._evidence(ctx)
        if d is None and evidence.get("empty"):  # only the gripper saw it; the verdict is still this one
            first = evidence["empty"][0]
            d = {"reason": "grasped_air", "at": first["t_s"], "message": MESSAGE["grasped_air"],
                 "details": {"source": "robot", "side": first.get("side")},
                 "candidates": [{"reason": "grasped_air", "at_s": first["t_s"]}]}
        if d is None:  # nothing wrong seen: the outcome stays unknown (vision-judge or a person decides)
            return
        matching = {"grasped_air": "empty", "dropped": "loose"}.get(d["reason"])
        if evidence:  # §7: other plugins' signals back the one verdict; they never add a second one
            d["details"]["evidence"] = [{"by": "gripper-check", **e} for kind, es in evidence.items() for e in es
                                        if matching is None or kind == matching] or None
        if ctx.result.get("reason") in ("model_error", "error", "stopped"):
            return  # cut short by something else; that stays the reason
        ctx.result.update(reason=d["reason"], at_s=d["at"], details=d["details"], message=d["message"],
                          candidates=d["candidates"])
        ctx.emit("error", f"{d['message']} at {d['at']:.1f} s", reason=d["reason"], at_s=float(d["at"]), source="robot")

    @staticmethod
    def _evidence(ctx: PluginContext) -> dict[str, list[dict[str, Any]]]:
        """gripper-check's live signals of this attempt, by kind (empty: closed on nothing, loose: grip loosening)."""
        out: dict[str, list[dict[str, Any]]] = {}
        for e in (ctx.result.get("gripper-check") or {}).get("signals") or []:
            out.setdefault(str(e.get("kind")), []).append(e)
        return out

    def _save_trace(self, ctx: PluginContext) -> None:
        every = self.params.trace_every
        out = []
        for r in self.rows[::every]:
            out.append({"t": r["t"], "speed": r["speed"],
                        **{f"{k}_{s}": round(r[f"{k}_{s}"], 3) for s in SIDES for k in ("cmd", "grip", "err")}})
        path = ctx.episode_dir / "trace.json"  # §8: beside the attempt's videos
        payload = {"rows": out, "milestones": self.milestones, "source": self.source}
        ctx.defer(path.write_text, json.dumps(payload))
        ctx.result["failure-monitor"] = {"trace": ctx.files.relative(path), "latency": self._latency(),
                                         "source": self.source, **({"if_failed": s} if (s := self._slow()) else {})}
