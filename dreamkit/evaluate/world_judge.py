"""world-judge: why an attempt failed, from the simulator's positions and contacts — an evaluation tool (§6).

The right answer dreamkit doctor --sim measures failure-monitor against: failure-monitor sees only what a real arm
reports; this one sees the world (where the object is, what touched what) and says dropped, released_off_target,
grasped_air, not_reached, collision, fell_off_table, stalled. It changes nothing in the run: no live verdicts, no
events; its answer goes to ctx.result["world-judge"] = {reason, at_s, message, success}."""

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
PRIORITY = ("fell_off_table", "camera_failure", "dropped", "released_off_target", "grasped_air",
            "collision", "stalled", "not_reached", "slow_model", "timeout")
LABEL = {
    "fell_off_table": "{obj} fell off the table",
    "camera_failure": "camera failure",
    "dropped": "dropped {obj} (the gripper never opened)",
    "released_off_target": "let go of {obj} outside the {target}",
    "grasped_air": "grasped air next to {obj}",
    "collision": "collision: {what}",
    "stalled": "arms stalled",
    "not_reached": "never reached {obj}",
    "slow_model": "model too slow",
    "timeout": "ran out of time",
}
BAD = {"separated", "fell", "collision", "stalled"}


def _robot(name: str) -> bool:
    return name.startswith("left_") or name.startswith("right_")


class Params(BaseModel):
    reach_m: float = Field(0.045, gt=0, le=0.3, description="A tool point this close to the object has reached it",
                           json_schema_extra={"unit": "m"})
    near_m: float = Field(0.08, gt=0, le=0.3, description="A gripper closing this close is a grasp on the object",
                          json_schema_extra={"unit": "m"})
    close_cmd: float = Field(0.3, ge=0, le=1, description="Gripper command below which the hand is closing",
                             json_schema_extra={"unit": ""})
    open_cmd: float = Field(0.6, ge=0, le=1, description="Gripper command above which the hand is opening",
                            json_schema_extra={"unit": ""})
    release_cmd: float = Field(0.85, ge=0, le=1, description="The hand let go only if its command reaches this",
                               json_schema_extra={"unit": ""})
    release_window_s: float = Field(1.0, gt=0, le=5, description="How long after the object left the hand to look "
                                    "for that release", json_schema_extra={"unit": "s"})
    lift_m: float = Field(0.03, gt=0, le=0.3, description="Raised this much above its start, the object is lifted",
                          json_schema_extra={"unit": "m"})
    grasp_lift_m: float = Field(0.02, gt=0, le=0.3, description="A grasp that never raises the object this much "
                                "grasped air", json_schema_extra={"unit": "m"})
    slip_m: float = Field(0.07, gt=0, le=0.3, description="A lifted object this far from every hand has left it",
                          json_schema_extra={"unit": "m"})
    miss_max_m: float = Field(0.15, gt=0, le=0.5, description="A release this close to the target is marked a near "
                              "miss in the details", json_schema_extra={"unit": "m"})
    fell_below_m: float = Field(0.05, gt=0, le=0.5, description="This far below the table top, it fell off",
                                json_schema_extra={"unit": "m"})
    contact_force_n: float = Field(2.0, gt=0, le=200, description="A robot contact this strong is a collision",
                                   json_schema_extra={"unit": "N"})
    finger_table_force_n: float = Field(25.0, gt=0, le=500, description="Fingertips may brush the table; only "
                                        "this much force counts", json_schema_extra={"unit": "N"})
    contact_min_s: float = Field(0.15, ge=0, le=5, description="A collision must last this long",
                                 json_schema_extra={"unit": "s"})
    stall_speed_rad_s: float = Field(0.03, gt=0, le=1, description="Every joint slower than this is not moving",
                                     json_schema_extra={"unit": "rad/s"})
    stall_s: float = Field(3.0, gt=0, le=60, description="Not moving this long is a stall",
                           json_schema_extra={"unit": "s"})
    slow_ms: float = Field(700.0, gt=0, le=10000, description="A model round trip above this is slow: warned at once, "
                           "and a failure if the attempt's mean is above it", json_schema_extra={"unit": "ms"})
    tie_s: float = Field(1.0, ge=0, le=10, description="Causes this close in time count as simultaneous",
                         json_schema_extra={"unit": "s"})
    settle_s: float = Field(1.0, gt=0, le=10, description="Wait this long after the object leaves the hand before "
                            "calling it", json_schema_extra={"unit": "s"})
    grasp_judge_s: float = Field(1.5, gt=0, le=10, description="A closed grasp that has not lifted the object after "
                                 "this long grasped air", json_schema_extra={"unit": "s"})
    not_reached_s: float = Field(15.0, gt=0, le=600, description="No reach and no grasp this long is not reached",
                                 json_schema_extra={"unit": "s"})
    live_verdicts: bool = Field(False, description="Never: an evaluation tool changes nothing in the run")
    empty_below: float = Field(0.05, ge=0, le=1, description="Robot mode: a closed gripper measuring less than this "
                               "holds nothing", json_schema_extra={"unit": ""})
    hold_gap: float = Field(0.15, gt=0, le=1, description="Robot mode: fingers stopped this far above their command "
                            "are holding something", json_schema_extra={"unit": ""})
    grip_settle_s: float = Field(0.6, gt=0, le=5, description="Robot mode: how long the fingers get to close",
                                 json_schema_extra={"unit": "s"})
    track_err_rad: float = Field(0.12, gt=0, le=3, description="Robot mode: an arm this far from a steady command is "
                                 "blocked (a collision)", json_schema_extra={"unit": "rad"})
    steady_cmd_rad: float = Field(0.02, gt=0, le=1, description="Robot mode: a command moving less than this per "
                                  "step is steady", json_schema_extra={"unit": "rad"})
    blocked_s: float = Field(0.3, gt=0, le=5, description="Robot mode: blocked this long is a collision",
                             json_schema_extra={"unit": "s"})
    trace_every: int = Field(2, ge=1, le=20, description="Keep every Nth step in the saved trace",
                             json_schema_extra={"unit": ""})


class WorldJudge(Plugin):
    """Judges an attempt from the simulator's world: the answer the monitor is measured against."""

    name = "world-judge"
    Params = Params
    manifest = Manifest(group="evaluation", icon="Scale", capability="observe",
                        hooks=["on_episode_start", "after_model", "on_step", "on_episode_end"], step_rate_hz=20,
                        needs=["robot", "world"])

    # -- recording ---------------------------------------------------------------------------------------------
    def on_episode_start(self, ctx: PluginContext) -> None:
        self.rtts: list[tuple[float, float]] = []  # latency works on any robot
        self.mock_calls = False
        self.world = ctx.world.available
        self.active = self.world or ctx.robot.available
        self.source = "world" if self.world else "robot"
        self.verdicts: list[dict[str, Any]] = []
        self.segment, self.seg_start_t, self.seg_grasp_from, self.verdict_given = 0, 0.0, 0, False
        self.still_since: float | None = 0.0
        self.stall: dict[str, Any] | None = None
        self.stall_warned = False
        self.milestones: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.last_q = {s: np.asarray(ctx.state[s][:6], dtype=float) for s in SIDES} if ctx.state else None
        if not self.active:
            return
        if not self.world:
            self.grip = {s: {"closed_since": None, "judged": False, "holding": False, "gap_closed": None,
                             "blocked_since": None} for s in SIDES}
            self.let_go = False  # the hand opened on purpose while holding something
            self.obj = self.target = None
            return
        self.obj, self.target = ctx.target["object"], ctx.target["target"]
        self.rows: list[dict[str, Any]] = []
        self.rtts: list[tuple[float, float]] = []
        self.mock_calls = False
        self.contacts: dict[tuple[str, str], dict[str, Any]] = {}
        self.collisions: list[dict[str, Any]] = []
        self.flags: dict[str, Any] = {"reached": False, "lifted": False, "grasping": {s: False for s in SIDES}}
        self.grasps: list[dict[str, Any]] = []
        if self.obj and ctx.world.has(self.obj):
            self.z0 = float(ctx.world.pos(self.obj)[2])
        else:
            self.obj = None
        self._record(ctx)

    def _milestone(self, ctx: PluginContext, kind: str, level: str, message: str, emit: bool = True,
                   **data: Any) -> None:
        self.milestones.append({"t": round(ctx.t, 2), "kind": kind, "message": message})  # kept, never shown

    def _record(self, ctx: PluginContext) -> None:
        w = ctx.world
        row: dict[str, Any] = {"t": round(ctx.t, 3)}
        cmd = ctx.command
        for i, s in enumerate(SIDES):
            row[f"cmd_{s}"] = float(cmd[i][6]) if cmd is not None else float(ctx.state[s][6]) if ctx.state else w.grip(s)
            row[f"tcp_{s}"] = w.tcp(s)
        if self.obj:
            pos = w.pos(self.obj)
            row["obj"] = pos
            for s in SIDES:
                row[f"d_{s}"] = float(np.linalg.norm(row[f"tcp_{s}"] - pos))
            row["d"] = min(row["d_left"], row["d_right"])
            if self.target:
                row["target"] = w.pos(self.target)
        self.rows.append(row)

    def after_model(self, ctx: PluginContext) -> None:
        """Model latency (what latency-monitor used to do): every round trip, a warning when slow."""
        timing = ctx.timing or {}
        if timing.get("mock"):
            self.mock_calls = True
            return
        if timing.get("rtt_ms") is None:
            return
        rtt = float(timing["rtt_ms"])
        self.rtts.append((ctx.t, rtt))


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
        if not self.world:
            self._stall(ctx)
            self._robot_step(ctx)
            return
        self._record(ctx)
        row = self.rows[-1]
        p = self.params
        self._contacts(ctx)
        self._stall(ctx)
        if not self.obj:
            return
        z = float(row["obj"][2])
        if not self.flags["reached"] and row["d"] < p.reach_m:
            self.flags["reached"] = True
            side = "left" if row["d_left"] < row["d_right"] else "right"
            self._milestone(ctx, "reached", "info", f"{side} hand reached {self.obj} ({row['d'] * 100:.1f} cm)",
                            side=side)
        for s in SIDES:
            if row[f"cmd_{s}"] < p.close_cmd and not self.flags["grasping"][s]:
                self.flags["grasping"][s] = True
                g = {"side": s, "t": round(ctx.t, 2), "dist": row[f"d_{s}"], "z_start": z, "rise": 0.0}
                self.grasps.append(g)
                if g["dist"] < p.near_m:
                    self._milestone(ctx, "grasp", "info",
                                    f"{s} gripper closing on {self.obj} ({g['dist'] * 100:.1f} cm away)", side=s)
            elif self.flags["grasping"][s] and row[f"cmd_{s}"] > p.open_cmd:
                self.flags["grasping"][s] = False
                self._milestone(ctx, "release", "info", f"{s} gripper opening ({row[f'cmd_{s}']:.2f})", emit=False)
            if self.flags["grasping"][s]:
                g = next(g for g in reversed(self.grasps) if g["side"] == s)
                g["rise"] = max(g["rise"], z - g["z_start"])
        if not self.flags["lifted"] and z - self.z0 > p.lift_m and row["d"] < p.slip_m:
            self.flags["lifted"], self.flags["separated"] = True, None
            self._milestone(ctx, "lifted", "info", f"lifted {self.obj} {(z - self.z0) * 100:.0f} cm")
        if self.flags["lifted"] and self.flags.get("separated") is None and row["d"] <= p.slip_m:
            self.flags["peak"] = max(self.flags.get("peak", z), z)
            self.flags["holder"] = "left" if row["d_left"] < row["d_right"] else "right"
        if self.flags["lifted"] and self.flags.get("separated") is None and row["d"] > p.slip_m:
            side = self.flags.get("holder") or ("left" if row["d_left"] < row["d_right"] else "right")
            let_go = row[f"cmd_{side}"] >= p.release_cmd
            height = self.flags.get("peak", z) - self.z0
            self.flags["separated"] = {"t": round(ctx.t, 2), "height_m": round(height, 3), "side": side,
                                       "tcp": [round(float(v), 3) for v in row[f"tcp_{side}"]]}
            how = "was let go" if let_go else f"slipped out of the hand (gripper at {row[f'cmd_{side}']:.2f}, not opened)"
            self._milestone(ctx, "separated", "warn", f"{self.obj} {how} at {height * 100:.0f} cm")
        if self.flags["lifted"] and self.flags.get("separated") and row["d"] < p.reach_m and z - self.z0 > p.lift_m:
            self.flags["separated"] = None  # picked up again
        if not self.flags.get("fell") and z < ctx.world.table_z - p.fell_below_m:
            self.flags["fell"] = round(ctx.t, 2)
            self._milestone(ctx, "fell", "error", f"{self.obj} fell off the table")
        if p.live_verdicts and not self.verdict_given:
            self._live_verdict(ctx, row)

    def _stall(self, ctx: PluginContext) -> None:
        p = self.params
        if not ctx.state or self.last_q is None:
            return
        q = {s: np.asarray(ctx.state[s][:6], dtype=float) for s in SIDES}
        speed = max(float(np.max(np.abs(q[s] - self.last_q[s]))) for s in SIDES) / 0.05
        self.last_q = q
        if speed < p.stall_speed_rad_s:
            if self.still_since is None:
                self.still_since = ctx.t
            elif ctx.t - self.still_since >= p.stall_s and not self.stall_warned:
                self.stall_warned = True
                self.stall = {"at": round(self.still_since, 2)}
                self._milestone(ctx, "stalled", "warn", f"arms have not moved for {p.stall_s:.0f} s")
                if not self.world and self.let_go:
                    self.stall = None  # it let go of the object and rests: finished, not stuck (robot mode)
                elif not self.world and p.live_verdicts and not self.verdict_given:
                    self._robot_verdict(ctx, "stalled", self.stall["at"], "arms stalled")
        else:
            self.still_since, self.stall_warned = None, False

    # -- robot mode: the gripper's and the arm's own signals --------------------------------------------------
    def _robot_verdict(self, ctx: PluginContext, reason: str, at: float, message: str, **details: Any) -> None:
        self.verdict_given = True
        v = {"reason": reason, "at": round(at, 2), "segment": self.segment, "source": "robot", **details}
        self.verdicts.append(v)
        ctx.verdict(reason, at, message, source="robot", **details)

    def _robot_step(self, ctx: PluginContext) -> None:
        p, r = self.params, ctx.robot
        row: dict[str, Any] = {"t": round(ctx.t, 3)}
        for s in SIDES:
            g = self.grip[s]
            cmd, meas = r.grip_command(s), r.grip(s)
            err, steady = r.tracking_error(s), r.command_speed(s) < p.steady_cmd_rad
            row[f"cmd_{s}"] = cmd if cmd is not None else meas
            row[f"grip_{s}"] = meas
            row[f"err_{s}"] = err
            # collision: the arm cannot follow a command that is not moving
            if cmd is not None and err > p.track_err_rad and steady:
                g["blocked_since"] = ctx.t if g["blocked_since"] is None else g["blocked_since"]
                if ctx.t - g["blocked_since"] >= p.blocked_s and not g.get("blocked_told"):
                    g["blocked_told"] = True
                    self._milestone(ctx, "collision", "warn", f"{s} arm blocked: {err:.2f} rad from a steady command",
                                    side=s)
                    if p.live_verdicts and not self.verdict_given:
                        self._robot_verdict(ctx, "collision", g["blocked_since"], f"{s} arm blocked (collision)", side=s)
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
                        self._robot_verdict(ctx, "grasped_air", g["closed_since"], f"{s} gripper closed on nothing",
                                            side=s)
                elif meas - cmd >= p.hold_gap:
                    g["holding"], g["hold_width"] = True, meas
                    self._milestone(ctx, "grasp", "info", f"{s} gripper holding something ({meas:.2f} wide)", side=s)
            if not closed:
                g["closed_since"] = None
            if g["holding"]:
                if cmd >= p.release_cmd:  # the policy opened the hand: a release, not a drop
                    g["holding"], g["gap_closed"] = False, None
                    self.let_go = True
                    self._milestone(ctx, "release", "info", f"{s} gripper opened ({cmd:.2f})", emit=False)
                elif meas - cmd < p.hold_gap / 2:  # nothing stops the fingers any more
                    g["gap_closed"] = ctx.t if g["gap_closed"] is None else g["gap_closed"]
                    if ctx.t - g["gap_closed"] >= p.release_window_s:  # and the hand was never told to open
                        g["holding"] = False
                        self._milestone(ctx, "separated", "warn",
                                        f"{s} gripper lost its object (gripper at {cmd:.2f}, not opened)", side=s)
                        if p.live_verdicts and not self.verdict_given:
                            self._robot_verdict(ctx, "dropped", g["gap_closed"], f"dropped the object ({s} gripper "
                                                "never opened)", side=s)
                else:
                    g["gap_closed"] = None
        self.rows.append(row)

    def _contacts(self, ctx: PluginContext) -> None:
        p = self.params
        now: dict[tuple[str, str], float] = {}
        for a, b, f in ctx.world.contacts(with_force=True, prefixes=("left_", "right_")):
            if not (_robot(a) or _robot(b)):
                continue
            if _robot(a) and _robot(b) and a.split("_")[0] == b.split("_")[0]:
                continue  # an arm touching itself
            robot, other = (a, b) if _robot(a) else (b, a)
            if other == self.obj:
                continue  # touching the object is the task
            finger = "_tip_" in robot
            limit = p.finger_table_force_n if finger and other in ("world", self.target) else p.contact_force_n
            if f >= limit:
                now[(robot, other)] = max(now.get((robot, other), 0.0), f)
        for key, f in now.items():
            entry = self.contacts.get(key)
            if entry is None:
                self.contacts[key] = {"since": ctx.t, "force": f, "reported": False}
                continue
            entry["force"] = max(entry["force"], f)
            if not entry["reported"] and ctx.t - entry["since"] >= p.contact_min_s:
                entry["reported"] = True
                robot, other = key
                other_name = "table" if other == "world" else other.replace("_", " ")
                col = {"at": round(entry["since"], 2), "robot": robot, "other": other_name,
                       "force_n": round(entry["force"], 1)}
                self.collisions.append(col)
                self._milestone(ctx, "collision", "warn", f"contact {robot} with {other_name} ({entry['force']:.0f} N)",
                                force_n=round(entry["force"], 1))
        for key in [k for k in self.contacts if k not in now]:
            del self.contacts[key]

    # -- live verdicts and segments -------------------------------------------------------------------------------
    def _new_segment(self, ctx: PluginContext) -> None:
        """The control plugin restarted the task mid-attempt: judge what happens next on its own."""
        self.segment = ctx.store.get("segment", 0)
        if not self.world:
            self.seg_start_t, self.verdict_given, self.let_go = ctx.t, False, False
            self.grip = {s: {"closed_since": None, "judged": False, "holding": False, "gap_closed": None,
                             "blocked_since": None} for s in SIDES}
            self.still_since, self.stall, self.stall_warned = None, None, False
            return
        self.seg_start_t, self.seg_grasp_from, self.verdict_given = ctx.t, len(self.grasps), False
        fell = self.flags.get("fell")
        self.flags = {"reached": False, "lifted": False, "grasping": {s: False for s in SIDES}}
        if fell:
            self.flags["fell"] = fell
        self.still_since, self.stall, self.stall_warned = None, None, False
        if self.obj:
            self.z0 = float(ctx.world.pos(self.obj)[2])

    def _in_target(self, row: dict[str, Any]) -> bool:
        if "target" not in row:
            return False
        return (float(np.linalg.norm(row["obj"][:2] - row["target"][:2])) < 0.09
                and row["target"][2] - 0.01 < row["obj"][2] < row["target"][2] + 0.12)

    def _separation_reason(self, sep: dict[str, Any], target, final_dist) -> str:
        p = self.params
        window = [r for r in self.rows if sep["t"] - 0.3 <= r["t"] <= sep["t"] + p.release_window_s]
        side = sep.get("side", "left")
        peak = max((r[f"cmd_{side}"] for r in window), default=0.0)
        sep["released"] = bool(peak >= p.release_cmd)
        sep["gripper_cmd_peak"] = round(peak, 2)
        release = float(np.linalg.norm(np.asarray(sep["tcp"][:2]) - target[:2])) if target is not None else None
        sep["release_distance_m"] = release and round(release, 3)
        sep["final_distance_m"] = final_dist and round(final_dist, 3)
        if not sep["released"]:
            return "dropped"
        sep["near_target"] = bool(release is not None and release <= p.miss_max_m)  # close miss, or far
        return "released_off_target"

    def _live_verdict(self, ctx: PluginContext, row: dict[str, Any]) -> None:
        p = self.params
        if not self.obj or self._in_target(row):
            return
        verdict = None
        sep = self.flags.get("separated")
        if self.flags.get("fell") and self.flags["fell"] >= self.seg_start_t:
            verdict = {"reason": "fell_off_table", "at": self.flags["fell"]}
        elif self.flags["lifted"] and sep and ctx.t - sep["t"] >= p.settle_s:
            target = row.get("target")
            final = float(np.linalg.norm(row["obj"][:2] - target[:2])) if target is not None else None
            verdict = {"reason": self._separation_reason(dict(sep), target, final), "at": sep["t"]}
        elif self.stall is not None:
            verdict = {"reason": "stalled", "at": self.stall["at"]}
        else:
            near = [g for g in self.grasps[self.seg_grasp_from:] if g["dist"] < p.near_m]
            last = near[-1] if near else None
            if (last and not self.flags["lifted"] and self.flags["grasping"][last["side"]]
                    and ctx.t - last["t"] >= p.grasp_judge_s and last["rise"] < p.grasp_lift_m):
                verdict = {"reason": "grasped_air", "at": last["t"], "side": last["side"]}
            elif not self.flags["reached"] and not near and ctx.t - self.seg_start_t >= p.not_reached_s:
                verdict = {"reason": "not_reached", "at": round(ctx.t, 2)}
        if verdict is None:
            return
        self.verdict_given = True
        message = LABEL[verdict["reason"]].format(obj=self.obj, target=self.target or "target", what="contact")
        self.verdicts.append({**verdict, "segment": self.segment})
        details = {"side": verdict["side"]} if "side" in verdict else {}
        ctx.verdict(verdict["reason"], verdict["at"], message, source="world", **details)

    # -- the attempt's verdict ------------------------------------------------------------------------------------
    def diagnose(self, ctx: PluginContext) -> dict[str, Any] | None:
        p = self.params
        cands: list[dict[str, Any]] = []
        if not self.world:
            return self._diagnose_robot(ctx)
        rows = self.rows
        end_t = rows[-1]["t"] if rows else ctx.t
        if self.obj and rows:
            final = rows[-1]["obj"]
            target = rows[-1].get("target")
            final_dist = float(np.linalg.norm(final[:2] - target[:2])) if target is not None else None
            if self.flags.get("fell"):
                cands.append({"reason": "fell_off_table", "at": self.flags["fell"]})
            sep = self.flags.get("separated")
            if self.flags["lifted"] and sep:
                cands.append({"reason": self._separation_reason(sep, target, final_dist), "at": sep["t"], "details": sep})
            near = [g for g in self.grasps if g["dist"] < p.near_m]
            air = [g for g in near if g["rise"] < p.grasp_lift_m]
            if air and not self.flags["lifted"]:
                g = air[0]
                cands.append({"reason": "grasped_air", "at": g["t"], "details": {
                    "side": g["side"], "distance_m": round(g["dist"], 3), "rise_m": round(g["rise"], 3)}})
            dmin = min(r["d"] for r in rows)
            if dmin > p.reach_m and not near:
                at = next(r["t"] for r in rows if r["d"] <= dmin + 0.002)
                cands.append({"reason": "not_reached", "at": at, "details": {"closest_m": round(dmin, 3)}})
        if self.collisions:
            c = self.collisions[0]
            cands.append({"reason": "collision", "at": c["at"], "details": c,
                          "what": f"{c['robot'].replace('_', ' ')} hit the {c['other']}"})
        if self.stall:
            cands.append({"reason": "stalled", "at": self.stall["at"], "details": self.stall})
        if self.rtts:
            mean = float(np.mean([r for _, r in self.rtts]))
            if mean > p.slow_ms:
                first = next(t for t, r in self.rtts if r > p.slow_ms)
                cands.append({"reason": "slow_model", "at": first, "details": {"mean_rtt_ms": round(mean)}})
        for v in ctx.store.get("verdicts", []):  # what other plugins (camera-check) or earlier segments called
            if not any(c["reason"] == v["reason"] and abs(c["at"] - v["at_s"]) < 0.5 for c in cands):
                cands.append({"reason": v["reason"], "at": v["at_s"],
                              "details": {"by": v.get("by"), "message": v.get("message"), "source": v.get("source", "robot")}})
        if not cands:
            cands.append({"reason": "timeout", "at": round(end_t, 2)})
        earliest = min(c["at"] for c in cands)
        tied = [c for c in cands if c["at"] - earliest <= p.tie_s]
        best = min(tied, key=lambda c: PRIORITY.index(c["reason"]) if c["reason"] in PRIORITY else 99)
        best.setdefault("details", {})
        best["message"] = LABEL.get(best["reason"], reasons.label(best["reason"])).format(
            obj=self.obj or "object", target=self.target or "target", what=best.get("what", "contact"))
        best["candidates"] = [{"reason": c["reason"], "at_s": c["at"]} for c in sorted(cands, key=lambda c: c["at"])]
        return best

    def _diagnose_robot(self, ctx: PluginContext) -> dict[str, Any] | None:
        """Robot mode: only what the robot's own signals show. Nothing found = the outcome is unknown, not success."""
        p = self.params
        cands = [{"reason": v["reason"], "at": v["at"], "details": {k: v[k] for k in ("side",) if k in v}}
                 for v in self.verdicts]
        if self.stall and not any(c["reason"] == "stalled" for c in cands):
            cands.append({"reason": "stalled", "at": self.stall["at"], "details": {}})
        # slow_model is warned live, but without a world it does not turn an unknown outcome into a failure
        for v in ctx.store.get("verdicts", []):
            if v.get("by") != "failure-monitor":
                cands.append({"reason": v["reason"], "at": v["at_s"], "details": {
                    "by": v.get("by"), "message": v.get("message"), "source": v.get("source", "robot")}})
        if not cands:
            return None
        earliest = min(c["at"] for c in cands)
        best = min([c for c in cands if c["at"] - earliest <= p.tie_s],
                   key=lambda c: PRIORITY.index(c["reason"]) if c["reason"] in PRIORITY else 99)
        best["details"].setdefault("source", "robot")
        best["message"] = {"grasped_air": "grasped air (the fingers closed on nothing)",
                           "dropped": "dropped the object (the gripper never opened)",
                           "collision": "collision (the arm could not follow its command)"}.get(
            best["reason"], LABEL.get(best["reason"], reasons.label(best["reason"])).format(
                obj="object", target="target", what="contact"))
        best["candidates"] = [{"reason": c["reason"], "at_s": c["at"]} for c in sorted(cands, key=lambda c: c["at"])]
        return best

    def on_episode_end(self, ctx: PluginContext) -> None:
        """The right answer for this attempt, under its own key: success (success-check, when on) and the reason."""
        if not self.active or not self.world:
            return
        truth = ctx.result.get("success-check-truth")
        ok = truth if isinstance(truth, bool) else ctx.result.get("success") is True
        out: dict[str, Any] = {"success": ok}
        if not ok:
            d = self.diagnose(ctx)
            if d is not None:
                out.update(reason=d["reason"], at_s=d["at"], message=d["message"])
        ctx.result["world-judge"] = out

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
            item = {"t": r["t"], "cmd_left": round(r["cmd_left"], 3), "cmd_right": round(r["cmd_right"], 3)}
            if "grip_left" in r:
                item.update(grip=round(min(r["grip_left"], r["grip_right"]), 3),
                            err=round(max(r["err_left"], r["err_right"]), 3))
            if "obj" in r:
                item.update(z=round(float(r["obj"][2]) - self.z0, 4), d=round(r["d"], 4))
                if "target" in r:
                    item["dt"] = round(float(np.linalg.norm(r["obj"][:2] - r["target"][:2])), 4)
            out.append(item)
        path = ctx.episode_dir / "trace.json"  # §8: beside the attempt's videos
        payload = {"object": self.obj, "target": self.target, "rows": out, "milestones": self.milestones,
                   "source": self.source}
        ctx.defer(path.write_text, json.dumps(payload))
        ctx.result["failure-monitor"] = {"trace": ctx.files.relative(path), "latency": self._latency(),
                                         "source": self.source}
