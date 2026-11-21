"""failure-monitor: says *why* an attempt failed and *when*, from a per-step trace of the physics."""

from __future__ import annotations

import json
from typing import Any

import mujoco
import numpy as np

from ..plugin import Context, Plugin
from ..sim import TABLE_TOP_Z
from ..task import parse

SIDES = ("left", "right")
## Tie-break when two causes happen within tie_s of each other: the more specific one wins.
PRIORITY = ("fell_off_table", "dropped", "released_off_target", "missed_target", "grasped_air", "collision",
            "stalled", "not_reached", "slow_model", "timeout")
LABEL = {
    "fell_off_table": "{obj} fell off the table",
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


def _robot(name: str) -> bool:
    return name.startswith("left_") or name.startswith("right_")


def _finger(name: str) -> bool:
    return "_tip_" in name


class FailureMonitor(Plugin):
    """Traces distances, heights, grips, contacts and joint motion; on failure picks the first cause in time."""

    name = "failure-monitor"
    group = "diagnosis"
    description = "Explains failures: why the attempt failed and at which second (dropped, grasped air, ...)"
    defaults = {
        "reach_m": 0.045,  # a tool point this close to the object counts as reaching it
        "near_m": 0.08,  # a gripper closing within this distance is a grasp attempt on the object
        "close_cmd": 0.3,  # commanded opening below this = gripper closing
        "open_cmd": 0.6,  # commanded opening above this = gripper releasing
        "release_cmd": 0.85,  # the policy opened the gripper (released) only if the command reaches this...
        "release_window_s": 1.0,  # ...within this long after the object left the hand; otherwise it was dropped
        "lift_m": 0.03,  # object raised this much above its start = lifted
        "grasp_lift_m": 0.02,  # a grasp that never raises the object this much grasped air
        "slip_m": 0.07,  # lifted object farther than this from every tool point = separated from the hand
        "miss_max_m": 0.15,  # released over, or resting, this close to the target (horizontally) = missed it
        "fell_below_m": 0.05,  # object this far below the table top = off the table
        "contact_force_n": 2.0,  # robot contact force that counts as a collision
        "finger_table_force_n": 25.0,  # fingertips may brush the table while grasping; only hard presses count
        "contact_min_s": 0.15,  # a collision must last this long
        "stall_speed": 0.03,  # rad/s: every joint slower than this = not moving
        "stall_s": 3.0,
        "slow_ms": 700.0,
        "tie_s": 1.0,
        "trace_every": 2,  # keep every Nth step in the saved trace
        # live verdicts (mid-attempt, what recovery acts on)
        "live_verdicts": True,
        "settle_s": 1.0,  # after the object leaves the hand, wait this long before calling it
        "grasp_judge_s": 1.5,  # a closed grasp that has not lifted the object after this long grasped air
        "not_reached_s": 15.0,  # no reach and no grasp near the object this long into a segment = not reached
    }

    # -- recording ---------------------------------------------------------------------------------------------
    def on_episode_start(self, ctx: Context) -> None:
        self.active = hasattr(ctx.env, "data") and hasattr(ctx.env, "model")
        if not self.active:
            ctx.emit("info", "no simulator env passed to episode()/step(): physics diagnosis is off")
            return
        self.obj, self.target = parse(ctx.task)
        m = ctx.env.model
        self.body_names = [m.body(i).name for i in range(m.nbody)]
        self.rows: list[dict[str, Any]] = []
        self.rtts: list[tuple[float, float]] = []
        self.contacts: dict[tuple[str, str], dict[str, Any]] = {}  # active contact pairs -> since, max force
        self.collisions: list[dict[str, Any]] = []
        self.milestones: list[dict[str, Any]] = []
        self.flags: dict[str, Any] = {"reached": False, "lifted": False, "grasping": {s: False for s in SIDES}}
        self.last_q = {s: ctx.state[s][:6].copy() for s in SIDES}
        self.still_since: float | None = 0.0
        self.stall: dict[str, Any] | None = None
        self.stall_warned = False
        self.grasps: list[dict[str, Any]] = []  # each closure: side, t, dist, max rise while closed
        self.verdicts: list[dict[str, Any]] = []  # live verdicts, one per segment at most
        self.segment = ctx.store.get("segment", 0)
        self.seg_start_t = 0.0
        self.seg_grasp_from = 0
        self.verdict_given = False
        if self.obj:
            self.z0 = float(ctx.env.data.body(self.obj).xpos[2])
            self.start = ctx.env.data.body(self.obj).xpos.copy()
        self._record(ctx)

    def _milestone(self, ctx: Context, kind: str, level: str, message: str, **data: Any) -> None:
        self.milestones.append({"t": round(ctx.t, 2), "kind": kind, "message": message, **data})
        ctx.emit(level, message, kind=kind, **data)

    def _record(self, ctx: Context) -> None:
        data = ctx.env.data
        row: dict[str, Any] = {"t": round(ctx.t, 3)}
        cmd = ctx.command
        for i, s in enumerate(SIDES):
            row[f"open_{s}"] = float(ctx.state[s][6])
            row[f"cmd_{s}"] = float(cmd[i][6]) if cmd is not None else float(ctx.state[s][6])
            row[f"tcp_{s}"] = data.site(f"{s}_tcp").xpos.copy()
        if self.obj:
            pos = data.body(self.obj).xpos.copy()
            row["obj"] = pos
            for s in SIDES:
                row[f"d_{s}"] = float(np.linalg.norm(row[f"tcp_{s}"] - pos))
            row["d"] = min(row["d_left"], row["d_right"])
        if self.target:
            row["target"] = data.body(self.target).xpos.copy()
        self.rows.append(row)

    def after_model(self, ctx: Context) -> None:
        if not getattr(self, "active", False):
            return
        if ctx.timing and ctx.timing.get("rtt_ms") is not None:
            self.rtts.append((ctx.t, float(ctx.timing["rtt_ms"])))

    def on_step(self, ctx: Context) -> None:
        if not getattr(self, "active", False):
            return
        if ctx.store.get("segment", 0) != self.segment:
            self._new_segment(ctx)
        self._record(ctx)
        row = self.rows[-1]
        p = self.params
        self._contacts(ctx)
        # joint motion -> stall
        speed = max(float(np.max(np.abs(ctx.state[s][:6] - self.last_q[s]))) for s in SIDES) / 0.05
        self.last_q = {s: ctx.state[s][:6].copy() for s in SIDES}
        if speed < p["stall_speed"]:
            if self.still_since is None:
                self.still_since = ctx.t
            elif ctx.t - self.still_since >= p["stall_s"] and not self.stall_warned:
                self.stall_warned = True
                self.stall = {"at": round(self.still_since, 2), "speed": speed}
                self._milestone(ctx, "stalled", "warn", f"arms have not moved for {p['stall_s']:.0f} s")
        else:
            if self.stall_warned and self.stall is not None:
                self.stall["until"] = round(ctx.t, 2)
            self.still_since = None
            self.stall_warned = False
        values = {"grip_l": round(row["cmd_left"], 3), "grip_r": round(row["cmd_right"], 3)}
        if self.obj:
            values.update(d=round(row["d"], 4), z=round(float(row["obj"][2]) - self.z0, 4))
            if "target" in row:
                values["dt"] = round(float(np.linalg.norm(row["obj"][:2] - row["target"][:2])), 4)
        ctx.telemetry(**values)
        if not self.obj:
            return
        # reach
        if not self.flags["reached"] and row["d"] < p["reach_m"]:
            self.flags["reached"] = True
            side = "left" if row["d_left"] < row["d_right"] else "right"
            self._milestone(ctx, "reached", "info", f"{side} hand reached {self.obj} ({row['d'] * 100:.1f} cm)",
                            side=side)
        # grasps: commanded closure near the object
        for s in SIDES:
            closing = row[f"cmd_{s}"] < p["close_cmd"]
            if closing and not self.flags["grasping"][s]:
                self.flags["grasping"][s] = True
                g = {"side": s, "t": round(ctx.t, 2), "dist": row[f"d_{s}"], "z_start": float(row["obj"][2]),
                     "rise": 0.0, "open": None}
                self.grasps.append(g)
                if g["dist"] < p["near_m"]:
                    self._milestone(ctx, "grasp", "info",
                                    f"{s} gripper closing on {self.obj} ({g['dist'] * 100:.1f} cm away)", side=s)
            elif self.flags["grasping"][s] and row[f"cmd_{s}"] > p["open_cmd"]:
                self.flags["grasping"][s] = False
                g = next(g for g in reversed(self.grasps) if g["side"] == s)
                g["open"] = round(ctx.t, 2)
                self._milestone(ctx, "release", "info", f"{s} gripper opening ({row[f'cmd_{s}']:.2f})", side=s)
            if self.flags["grasping"][s]:
                g = next(g for g in reversed(self.grasps) if g["side"] == s)
                g["rise"] = max(g["rise"], float(row["obj"][2]) - g["z_start"])
        # lifted / separated
        z = float(row["obj"][2])
        if not self.flags["lifted"] and z - self.z0 > p["lift_m"] and row["d"] < p["slip_m"]:
            self.flags["lifted"] = True
            self.flags["separated"] = None
            self._milestone(ctx, "lifted", "info", f"lifted {self.obj} {(z - self.z0) * 100:.0f} cm")
        if self.flags["lifted"] and self.flags.get("separated") is None and row["d"] <= p["slip_m"]:
            self.flags["peak"] = max(self.flags.get("peak", z), z)
            self.flags["holder"] = "left" if row["d_left"] < row["d_right"] else "right"
        if self.flags["lifted"] and self.flags.get("separated") is None and row["d"] > p["slip_m"]:
            side = self.flags.get("holder") or ("left" if row["d_left"] < row["d_right"] else "right")
            open_cmd = row[f"cmd_{side}"] >= p["release_cmd"]
            height = self.flags.get("peak", z) - self.z0
            self.flags["separated"] = {
                "t": round(ctx.t, 2), "height_m": round(height, 3), "released": bool(open_cmd), "side": side,
                "tcp": [round(float(v), 3) for v in row[f"tcp_{side}"]],
            }
            how = "was let go" if open_cmd else f"slipped out of the hand (gripper at {row[f'cmd_{side}']:.2f}, not opened)"
            self._milestone(ctx, "separated", "warn" if not self._near_target(row) else "info",
                            f"{self.obj} {how} at {height * 100:.0f} cm")
        if self.flags["lifted"] and self.flags.get("separated") and row["d"] < p["reach_m"] and z - self.z0 > p["lift_m"]:
            # picked up again: the next separation counts
            self.flags["separated"] = None
        if not self.flags.get("fell") and z < TABLE_TOP_Z - p["fell_below_m"]:
            self.flags["fell"] = round(ctx.t, 2)
            self._milestone(ctx, "fell", "error", f"{self.obj} fell off the table")
        if p["live_verdicts"] and not self.verdict_given:
            self._live_verdict(ctx, row)

    # -- live verdicts and segments -------------------------------------------------------------------------------
    def _new_segment(self, ctx: Context) -> None:
        """A control plugin (recovery) restarted the task mid-attempt: judge what happens next on its own."""
        self.segment = ctx.store.get("segment", 0)
        self.seg_start_t = ctx.t
        self.seg_grasp_from = len(self.grasps)
        self.verdict_given = False
        fell = self.flags.get("fell")
        self.flags = {"reached": False, "lifted": False, "grasping": {s: False for s in SIDES}}
        if fell:
            self.flags["fell"] = fell
        self.still_since, self.stall, self.stall_warned = None, None, False
        if self.obj:
            self.z0 = float(ctx.env.data.body(self.obj).xpos[2])

    def _in_target(self, row) -> bool:
        if "target" not in row:
            return False
        return (float(np.linalg.norm(row["obj"][:2] - row["target"][:2])) < 0.09
                and row["target"][2] - 0.01 < row["obj"][2] < row["target"][2] + 0.12)

    def _separation_reason(self, sep: dict[str, Any], target, final_dist) -> str:
        p = self.params
        window = [r for r in self.rows if sep["t"] - 0.3 <= r["t"] <= sep["t"] + p["release_window_s"]]
        side = sep.get("side", "left")
        peak = max((r[f"cmd_{side}"] for r in window), default=0.0)
        sep["released"] = bool(peak >= p["release_cmd"])
        sep["gripper_cmd_peak"] = round(peak, 2)
        release_dist = float(np.linalg.norm(np.asarray(sep["tcp"][:2]) - target[:2])) if target is not None else None
        sep["release_distance_m"] = release_dist and round(release_dist, 3)
        sep["final_distance_m"] = final_dist and round(final_dist, 3)
        if not sep["released"]:
            return "dropped"
        if release_dist is not None and release_dist <= p["miss_max_m"]:
            return "missed_target"
        return "released_off_target"

    def _live_verdict(self, ctx: Context, row: dict[str, Any]) -> None:
        p = self.params
        if not self.obj or self._in_target(row):
            return
        verdict = None
        sep = self.flags.get("separated")
        if self.flags.get("fell") and self.flags["fell"] >= self.seg_start_t:
            verdict = {"reason": "fell_off_table", "at": self.flags["fell"]}
        elif self.flags["lifted"] and sep and ctx.t - sep["t"] >= p["settle_s"]:
            target = row.get("target")
            final = float(np.linalg.norm(row["obj"][:2] - target[:2])) if target is not None else None
            verdict = {"reason": self._separation_reason(dict(sep), target, final), "at": sep["t"]}
        elif self.stall is not None:
            verdict = {"reason": "stalled", "at": self.stall["at"]}
        else:
            seg_grasps = self.grasps[self.seg_grasp_from:]
            near = [g for g in seg_grasps if g["dist"] < p["near_m"]]
            last = near[-1] if near else None
            if (last and not self.flags["lifted"] and self.flags["grasping"][last["side"]]
                    and ctx.t - last["t"] >= p["grasp_judge_s"] and last["rise"] < p["grasp_lift_m"]):
                verdict = {"reason": "grasped_air", "at": last["t"], "side": last["side"]}
            elif not self.flags["reached"] and not near and ctx.t - self.seg_start_t >= p["not_reached_s"]:
                verdict = {"reason": "not_reached", "at": round(ctx.t, 2)}
        if verdict is None:
            return
        self.verdict_given = True
        verdict.update(segment=self.segment, message=LABEL[verdict["reason"]].format(
            obj=self.obj, target=self.target or "target", what="contact"))
        self.verdicts.append(verdict)
        ctx.store["verdict"] = verdict
        ctx.emit("warn", f"verdict: {verdict['message']} at {verdict['at']:.1f} s", kind="verdict",
                 **{k: v for k, v in verdict.items() if k != "message"})

    def _near_target(self, row) -> bool:
        if "target" not in row:
            return False
        return float(np.linalg.norm(row["obj"][:2] - row["target"][:2])) < self.params["miss_max_m"] / 2

    def _contacts(self, ctx: Context) -> None:
        m, d = ctx.env.model, ctx.env.data
        p = self.params
        force = np.zeros(6)
        now: dict[tuple[str, str], float] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            a = self.body_names[m.geom_bodyid[c.geom1]]
            b = self.body_names[m.geom_bodyid[c.geom2]]
            if not (_robot(a) or _robot(b)):
                continue
            if _robot(a) and _robot(b) and a.split("_")[0] == b.split("_")[0]:
                continue  # self-contact inside one arm (adjacent parts) is the model's business
            robot, other = (a, b) if _robot(a) else (b, a)
            if other == self.obj and _finger(robot):
                continue  # fingers on the object: that is grasping
            if other == self.obj:
                continue  # pushing the object with the hand is not a collision by itself
            mujoco.mj_contactForce(m, d, i, force)
            f = abs(float(force[0]))
            limit = p["finger_table_force_n"] if _finger(robot) and other in ("world", self.target) else p["contact_force_n"]
            if f < limit:
                continue
            key = (robot, other)
            now[key] = max(now.get(key, 0.0), f)
        for key, f in now.items():
            entry = self.contacts.get(key)
            if entry is None:
                self.contacts[key] = {"since": ctx.t, "force": f, "reported": False}
            else:
                entry["force"] = max(entry["force"], f)
                if not entry["reported"] and ctx.t - entry["since"] >= p["contact_min_s"]:
                    entry["reported"] = True
                    robot, other = key
                    other_name = "table" if other == "world" else other.replace("_", " ")
                    col = {"at": round(entry["since"], 2), "robot": robot, "other": other_name,
                           "force_n": round(entry["force"], 1)}
                    self.collisions.append(col)
                    self._milestone(ctx, "collision", "warn",
                                    f"contact {robot} ↔ {other_name} ({entry['force']:.0f} N)", **col)
        for key in list(self.contacts):
            if key not in now:
                del self.contacts[key]

    # -- diagnosis -----------------------------------------------------------------------------------------------
    def diagnose(self, ctx: Context) -> dict[str, Any]:
        """All candidate causes, then the first in time (ties broken by PRIORITY)."""
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
            near_grasps = [g for g in self.grasps if g["dist"] < p["near_m"]]
            air = [g for g in near_grasps if g["rise"] < p["grasp_lift_m"]]
            if air and not self.flags["lifted"]:
                g = air[0]
                cands.append({"reason": "grasped_air", "at": g["t"],
                              "details": {"side": g["side"], "distance_m": round(g["dist"], 3),
                                          "rise_m": round(g["rise"], 3)}})
            dmin = min(r["d"] for r in rows)
            if dmin > p["reach_m"] and not near_grasps:
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
            if mean > p["slow_ms"]:
                first = next(t for t, r in self.rtts if r > p["slow_ms"])
                cands.append({"reason": "slow_model", "at": first, "details": {"mean_rtt_ms": round(mean)}})
        for v in self.verdicts:  # what went wrong before a recovery restarted the task
            if not any(c["reason"] == v["reason"] and abs(c["at"] - v["at"]) < 0.5 for c in cands):
                cands.append({"reason": v["reason"], "at": v["at"], "details": {"segment": v["segment"]}})
        if not cands:
            cands.append({"reason": "timeout", "at": round(end_t, 2)})
        earliest = min(c["at"] for c in cands)
        tied = [c for c in cands if c["at"] - earliest <= p["tie_s"]]
        best = min(tied, key=lambda c: PRIORITY.index(c["reason"]))
        best.setdefault("details", {})
        best["message"] = LABEL[best["reason"]].format(obj=self.obj or "object", target=self.target or "target",
                                                       what=best.get("what", "contact"))
        best["candidates"] = [{"reason": c["reason"], "at": c["at"]} for c in sorted(cands, key=lambda c: c["at"])]
        return best

    def on_episode_end(self, ctx: Context) -> None:
        if not getattr(self, "active", False):
            return
        self._save_trace(ctx)
        if ctx.result.get("success") is True:
            return
        diagnosis = self.diagnose(ctx)
        ctx.result["timeline"] = self.milestones
        if ctx.result.get("reason") == "stopped" and diagnosis["reason"] != "timeout":
            # stopped by the user after something had already gone wrong: that is the story
            ctx.result["details"] = {**diagnosis["details"], "stopped_by_user_at": ctx.result.get("at")}
        elif ctx.result.get("reason") in ("stopped", "stopped_by_plugin", "model_error", "error"):
            ctx.result["diagnosis"] = diagnosis  # the attempt was cut short; that stays the reason
            return
        else:
            ctx.result["details"] = diagnosis["details"]
        ctx.result.update(success=False, reason=diagnosis["reason"], at=diagnosis["at"],
                          message=diagnosis["message"],
                          candidates=diagnosis["candidates"])
        ctx.emit("error", f"{diagnosis['message']} at {diagnosis['at']:.1f} s", reason=diagnosis["reason"],
                 at=diagnosis["at"], video=f"ep{ctx.episode}/top.mp4#t={max(0.0, diagnosis['at'] - 1):.1f}")

    def _save_trace(self, ctx: Context) -> None:
        if ctx.episode_dir is None or not self.rows:
            return
        every = max(1, int(self.params["trace_every"]))
        out = []
        for r in self.rows[::every]:
            item = {"t": r["t"], "cmd_left": round(r["cmd_left"], 3), "cmd_right": round(r["cmd_right"], 3)}
            if "obj" in r:
                item.update(z=round(float(r["obj"][2]) - self.z0, 4), d=round(r["d"], 4))
                if "target" in r:
                    item["dt"] = round(float(np.linalg.norm(r["obj"][:2] - r["target"][:2])), 4)
            out.append(item)
        (ctx.episode_dir / "trace.json").write_text(json.dumps({"object": self.obj, "target": self.target,
                                                                "rows": out, "milestones": self.milestones}))
