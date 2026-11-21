"""failure-monitor: why an attempt failed and when, from what ctx.world measures every step (PLUGIN_STANDARD.md).

Live: milestones in the feed and on the timeline, a verdict as soon as a failure is certain (recovery acts on it).
At the end: the first cause in time, written to ctx.result (reason, at_s, details, message, candidates, timeline)."""

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
PRIORITY = ("fell_off_table", "camera_failure", "dropped", "released_off_target", "missed_target", "grasped_air",
            "collision", "stalled", "not_reached", "slow_model", "timeout")
LABEL = {
    "fell_off_table": "{obj} fell off the table",
    "camera_failure": "camera failure",
    "dropped": "dropped {obj} (the gripper never opened)",
    "released_off_target": "let go of {obj} away from the {target}",
    "missed_target": "missed the {target} with {obj}",
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
    miss_max_m: float = Field(0.15, gt=0, le=0.5, description="Released this close to the target, it missed it "
                              "(farther: released off target)", json_schema_extra={"unit": "m"})
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
    slow_ms: float = Field(700.0, gt=0, le=10000, description="A mean model round trip above this is too slow",
                           json_schema_extra={"unit": "ms"})
    tie_s: float = Field(1.0, ge=0, le=10, description="Causes this close in time count as simultaneous",
                         json_schema_extra={"unit": "s"})
    settle_s: float = Field(1.0, gt=0, le=10, description="Wait this long after the object leaves the hand before "
                            "calling it", json_schema_extra={"unit": "s"})
    grasp_judge_s: float = Field(1.5, gt=0, le=10, description="A closed grasp that has not lifted the object after "
                                 "this long grasped air", json_schema_extra={"unit": "s"})
    not_reached_s: float = Field(15.0, gt=0, le=600, description="No reach and no grasp this long is not reached",
                                 json_schema_extra={"unit": "s"})
    live_verdicts: bool = Field(True, description="Call failures mid-attempt (recovery acts on them)")
    trace_every: int = Field(2, ge=1, le=20, description="Keep every Nth step in the saved trace",
                             json_schema_extra={"unit": ""})


class FailureMonitor(Plugin):
    """Explains failures: why an attempt failed and at which second."""

    name = "failure-monitor"
    Params = Params
    manifest = Manifest(group="core", icon="ScanSearch", capability="observe",
                        hooks=["on_episode_start", "after_model", "on_step", "on_episode_end"], step_rate_hz=20)

    # -- recording ---------------------------------------------------------------------------------------------
    def on_episode_start(self, ctx: PluginContext) -> None:
        self.active = ctx.world.available
        if not self.active:
            ctx.emit("info", "no simulator world in episode() or step(): physics diagnosis is off")
            return
        self.obj, self.target = ctx.target["object"], ctx.target["target"]
        self.rows: list[dict[str, Any]] = []
        self.rtts: list[tuple[float, float]] = []
        self.contacts: dict[tuple[str, str], dict[str, Any]] = {}
        self.collisions: list[dict[str, Any]] = []
        self.milestones: list[dict[str, Any]] = []
        self.verdicts: list[dict[str, Any]] = []
        self.flags: dict[str, Any] = {"reached": False, "lifted": False, "grasping": {s: False for s in SIDES}}
        self.last_q = {s: np.asarray(ctx.state[s][:6], dtype=float) for s in SIDES} if ctx.state else None
        self.still_since: float | None = 0.0
        self.stall: dict[str, Any] | None = None
        self.stall_warned = False
        self.grasps: list[dict[str, Any]] = []
        self.segment, self.seg_start_t, self.seg_grasp_from, self.verdict_given = 0, 0.0, 0, False
        if self.obj and ctx.world.has(self.obj):
            self.z0 = float(ctx.world.pos(self.obj)[2])
        else:
            self.obj = None
        self._record(ctx)

    def _milestone(self, ctx: PluginContext, kind: str, level: str, message: str, emit: bool = True,
                   **data: Any) -> None:
        self.milestones.append({"t": round(ctx.t, 2), "kind": kind, "message": message})
        ctx.ui.mark(ctx.t, message, "warn" if kind in BAD else "info")
        if emit:  # the feed gets at most one event a second from on_step (§7); the timeline gets every mark
            ctx.emit(level, message, kind=kind, **data)

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
        if self.active and ctx.timing and ctx.timing.get("rtt_ms") is not None:
            self.rtts.append((ctx.t, float(ctx.timing["rtt_ms"])))

    def on_step(self, ctx: PluginContext) -> None:
        if not self.active:
            return
        if ctx.store.get("segment", 0) != self.segment:
            self._new_segment(ctx)
        self._record(ctx)
        row = self.rows[-1]
        p = self.params
        self._contacts(ctx)
        if ctx.state and self.last_q is not None:
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
            else:
                self.still_since, self.stall_warned = None, False
        if not self.obj:
            return
        z = float(row["obj"][2])
        ctx.ui.series("d", ctx.t, row["d"], unit="m")
        ctx.ui.series("z", ctx.t, z - self.z0, unit="m")
        if "target" in row:
            ctx.ui.series("dt", ctx.t, float(np.linalg.norm(row["obj"][:2] - row["target"][:2])), unit="m")
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
        if release is not None and release <= p.miss_max_m:
            return "missed_target"
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
        ctx.verdict(verdict["reason"], verdict["at"], message, **details)

    # -- the attempt's verdict ------------------------------------------------------------------------------------
    def diagnose(self, ctx: PluginContext) -> dict[str, Any]:
        p = self.params
        cands: list[dict[str, Any]] = []
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
                              "details": {"by": v.get("by"), "message": v.get("message")}})
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

    def on_episode_end(self, ctx: PluginContext) -> None:
        if not self.active:
            return
        self._save_trace(ctx)
        ctx.result["timeline"] = self.milestones
        if ctx.result.get("success") is True:
            return
        d = self.diagnose(ctx)
        if ctx.result.get("reason") in ("stopped", "model_error", "error", "camera_failure") and d["reason"] == "timeout":
            return  # cut short by something else; that stays the reason
        if ctx.result.get("reason") in ("model_error", "error"):
            return
        ctx.result.update(reason=d["reason"], at_s=d["at"], details=d["details"], message=d["message"],
                          candidates=d["candidates"])
        ctx.emit("error", f"{d['message']} at {d['at']:.1f} s", reason=d["reason"], at_s=float(d["at"]))

    def _save_trace(self, ctx: PluginContext) -> None:
        every = self.params.trace_every
        out = []
        for r in self.rows[::every]:
            item = {"t": r["t"], "cmd_left": round(r["cmd_left"], 3), "cmd_right": round(r["cmd_right"], 3)}
            if "obj" in r:
                item.update(z=round(float(r["obj"][2]) - self.z0, 4), d=round(r["d"], 4))
                if "target" in r:
                    item["dt"] = round(float(np.linalg.norm(r["obj"][:2] - r["target"][:2])), 4)
            out.append(item)
        path = ctx.files.path(ctx.files.episode_name("trace.json"))
        payload = {"object": self.obj, "target": self.target, "rows": out, "milestones": self.milestones}
        ctx.defer(path.write_text, json.dumps(payload))
        ctx.result["failure-monitor"] = {"trace": ctx.files.relative(path)}
