"""Tell the arms what to do, one task at a time, with every dreamkit plugin running.

The simulator opens with the arms at rest and nothing planned. Type a task at the `task>` prompt; Dream-1 (or the
scripted mock) carries it out through `dreamkit.wrap`, so failure-monitor, recovery, harvester and the rest work and
the attempt shows up in the panel (`dreamkit ui`: Live while it runs, History afterwards). The scene is not reset
between tasks, as on a real table: the next task starts from where the last one left the fruit and the arms.

    python examples/interactive.py --viewer          # Dream-1, MuJoCo window (uses credits)
    python examples/interactive.py --mock --viewer   # scripted arm, no API calls, no credits

At the prompt:
    put banana on the plate      any task: one attempt (plus a retry if recovery asks for one)
    /home                        drive the arms home over 2 s
    /reset                       new scene: objects shuffled (seed + 1), arms home
    /seconds 60                  time limit per attempt
    /model <id>                  switch model, e.g. a fine-tune (not with --mock)
    /status  /help  /quit
Ctrl-C during an attempt stops that attempt and returns to the prompt; Ctrl-C at the prompt quits.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout without installing

import dreamkit  # noqa: E402
from dreamkit import config as dk_config, scene, sim  # noqa: E402

try:
    import readline  # noqa: F401  (arrow-key history at the prompt where available)
except ImportError:
    pass

SUGGESTED = ("put apple into the bowl", "put orange into the bowl", "put banana on the plate")
HELP = __doc__.split("At the prompt:")[1].strip()


class Session:
    def __init__(self, args, env, yam_env, dc, client, viewer, sync, gateway, poses):
        self.args, self.env, self.yam_env, self.dc = args, env, yam_env, dc
        self.client, self.viewer, self.sync = client, viewer, sync
        self.gateway, self.poses = gateway, poses
        self.model = args.model
        self.seconds = args.seconds
        self.seed = args.seed
        self.stop = False

    # -- viewer -----------------------------------------------------------------------------------------------
    def show(self) -> None:
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit("MuJoCo window closed")
            self.sync(self.viewer)

    def hold(self, steps: int) -> None:
        state = self.env.state()
        for _ in range(steps):
            self.env.step(state["left"], state["right"])
            self.show()

    # -- commands ---------------------------------------------------------------------------------------------
    def home(self, seconds: float = 2.0) -> None:
        for left, right in self.yam_env.home_path(self.env.state(), seconds=seconds):
            tick = time.monotonic()
            self.env.step(left, right)
            self.show()
            if not self.args.fast:
                time.sleep(max(0.0, 0.05 - (time.monotonic() - tick)))
        print("arms home", flush=True)

    def new_scene(self) -> None:
        self.env.reset()
        positions = scene.randomize(self.env, list(sim.FRUITS), self.args.jitter, self.seed)
        self.show()
        where = "  ".join(f"{k} ({positions[k][0]:+.3f}, {positions[k][1]:+.3f})" for k in sim.FRUITS)
        print(f"scene seed {self.seed}: {where}", flush=True)
        self.positions = positions

    def status(self) -> None:
        eps = self.client.episodes
        ok = sum(1 for e in eps if e.get("success"))
        print(f"model {'mock' if self.args.mock else self.model} · {self.seconds:g} s per attempt · scene seed {self.seed}"
              f" · {ok}/{len(eps)} attempts succeeded · run {os.path.relpath(self.client.run_dir)}", flush=True)

    def set_model(self, model: str) -> None:
        if self.args.mock:
            print("the mock has no models; restart without --mock to use Dream-1", flush=True)
            return
        self.model = model
        self.poses = model in self.dc.POSE_MODELS
        self.client.policy(self.dc.policy_config(model, self.gateway, self.poses))
        print(f"model: {model}", flush=True)

    # -- one task ---------------------------------------------------------------------------------------------
    def run(self, task: str) -> None:
        dc, env, client = self.dc, self.env, self.client
        self.stop = False
        previous = signal.signal(signal.SIGINT, self._interrupt)
        try:
            while True:
                self._attempt(task)
                last = client.episodes[-1] if client.episodes else {}
                verdict = "succeeded" if last.get("success") else (
                    "outcome unknown" if last.get("success") is None else f"failed: {last.get('reason')}"
                    + (f" at {last['at_s']} s" if last.get("at_s") is not None else ""))
                print(f"attempt {last.get('episode', '?')} {verdict}", flush=True)
                if self.stop or not client.retry_requested:
                    break
                print("recovery asked for a retry from here", flush=True)
        finally:
            signal.signal(signal.SIGINT, previous)

    def _interrupt(self, *_):
        if self.stop:  # second Ctrl-C during an attempt: leave for real
            raise KeyboardInterrupt
        self.stop = True
        print("\nstopping this attempt (Ctrl-C again to quit)", flush=True)

    def _attempt(self, task: str) -> None:
        dc, env, client, args = self.dc, self.env, self.client, self.args
        windows = {name: deque(maxlen=dc.FRAMES) for name in dc.VIDEO_ROLES}
        steps = 0
        images = None
        while any(len(w) < dc.FRAMES for w in windows.values()):  # the model's camera windows, as dream_control does
            state = env.state()
            env.step(state["left"], state["right"])
            images = dc.capture(env, windows, keep=steps % dc.FRAME_EVERY_STEPS == 0)
            steps += 1
        self.show()
        world = {} if args.no_world else {"env": env}
        with client.episode(task, images=images, state=env.state(), seed=self.seed, interactive=True, **world):
            dc.feed(client, env, windows, task, self.model, steps, every=0, station=self.poses)
            while client.t < self.seconds and not self.stop:
                reply = client.get_action()  # physics holds while the model thinks
                if self.stop:
                    break
                left, right = dc.validate_chunk(env, reply, args.actions, verbose=False)
                for l, r in zip(left, right):
                    tick = time.monotonic()
                    env.step(l, r)
                    kept = steps % dc.FRAME_EVERY_STEPS == 0
                    wanted = steps % client.config.render_every == 0 and client.wants_frames
                    images = dc.capture(env, windows, keep=kept) if kept or wanted else None
                    if kept:
                        dc.feed(client, env, windows, task, self.model, steps, station=self.poses)
                    steps += 1
                    client.step(state=env.state(), images=images, command=(l, r), **world)
                    self.show()
                    if self.stop or client.t >= self.seconds:
                        break
                    if not args.fast:
                        time.sleep(max(0.0, 0.05 - (time.monotonic() - tick)))
            if self.stop:
                client.ctx.result.setdefault("success", False)
                client.ctx.result.setdefault("reason", "stopped")

    # -- prompt -----------------------------------------------------------------------------------------------
    def loop(self) -> None:
        print("\narms ready. Type a task, e.g. " + " · ".join(SUGGESTED) + "   (/help for commands)", flush=True)
        while True:
            try:
                line = input("task> ").strip()
            except EOFError:
                return
            if not line:
                continue
            word, _, rest = line.partition(" ")
            if word in ("/quit", "/exit", "/q"):
                return
            if word == "/help":
                print(HELP, flush=True)
            elif word == "/home":
                self.home()
            elif word == "/reset":
                self.seed += 1
                self.new_scene()
            elif word == "/seconds":
                try:
                    self.seconds = max(1.0, float(rest))
                    print(f"{self.seconds:g} s per attempt", flush=True)
                except ValueError:
                    print("usage: /seconds 60", flush=True)
            elif word == "/model":
                self.set_model(rest.strip()) if rest.strip() else print(f"model: {self.model}", flush=True)
            elif word == "/status":
                self.status()
            elif word.startswith("/"):
                print(f"unknown command {word}; /help lists them", flush=True)
            elif not any(c.isalpha() for c in line):
                print("type a task in words, e.g. put apple into the bowl", flush=True)
            else:
                self.run(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dreamlabs/dream-1-beta")
    parser.add_argument("--seconds", type=float, default=40, help="simulated seconds per attempt")
    parser.add_argument("--actions", type=int, default=24, help="rows executed per model reply, 1-48")
    parser.add_argument("--config", default="dreamkit.yaml", help="dreamkit config (plugins, runs folder)")
    parser.add_argument("--sim", help="Dream Labs' yam_mujoco folder (default: sim_path from the config)")
    parser.add_argument("--seed", type=int, default=42, help="first scene; /reset moves to the next")
    parser.add_argument("--jitter", type=float, default=0.03, help="metres of random shift for the fruit")
    parser.add_argument("--mock", action="store_true", help="scripted arm instead of Dream-1: no API calls")
    parser.add_argument("--behaviors", default="dropped,success,released_off_target,success",
                        help="mock only: what goes wrong, one per attempt, cycled")
    parser.add_argument("--viewer", action="store_true", help="open the simulator's MuJoCo window "
                        "(macOS: re-runs itself under mjpython)")
    parser.add_argument("--no-world", action="store_true", help="give dreamkit only what a real robot reports")
    parser.add_argument("--fast", action="store_true", help="no real-time pacing")
    args = parser.parse_args(argv)

    if args.viewer and sys.platform == "darwin" and os.environ.get("DREAMKIT_UNDER_MJPYTHON") != "1":
        mjpython = Path(sys.executable).with_name("mjpython")
        os.environ["DREAMKIT_UNDER_MJPYTHON"] = "1"
        os.execv(str(mjpython), [str(mjpython), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])

    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    yam_env, dc = sim.use(Path(args.sim).expanduser() if args.sim else cfg.sim_dir(base))
    env = yam_env.YamEnv()

    if args.mock:
        from dreamkit.mock import MockClient

        base_client = MockClient(env, behaviors=args.behaviors.split(","), seed=args.seed)
        gateway, poses = "mock", False
    else:
        import dream

        settings = dc.settings()
        key = settings.get("DREAM_API_KEY")
        if not key:
            parser.error("no API key: set DREAM_API_KEY or ~/.dreamlabs/credentials (or use --mock)")
        gateway = settings.get("DREAM_GATEWAY_URL", "https://api.dreamlabs.ai")
        poses = args.model in dc.POSE_MODELS
        base_client = dream.DreamClient(api_key=key)

    viewer = sync = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
        yam_env.aim_viewer(viewer)
        sync = dc.RunViewerSync(env)

    client = dreamkit.wrap(base_client, config=args.config)
    client.policy(dc.policy_config(args.model, gateway, poses))
    session = Session(args, env, yam_env, dc, client, viewer, sync, gateway, poses)
    status = "finished"
    try:
        session.new_scene()
        session.loop()
    except KeyboardInterrupt:
        status = "stopped"
        print()
    finally:
        client.close(status=status)
        if viewer is not None:
            viewer.close()
        env.close()
    eps = client.episodes
    print(f"{sum(1 for e in eps if e.get('success'))}/{len(eps)} attempts succeeded; "
          f"run saved in {os.path.relpath(client.run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
