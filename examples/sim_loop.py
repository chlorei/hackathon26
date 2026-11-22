"""A user's control loop on Dream Labs' simulator, with dreamkit plugged in.

This is dream_control.main's loop written the way a Dream-1 user writes theirs: Dream Labs' own functions
(policy_config, feed, capture, validate_chunk) drive the simulator unchanged; dreamkit adds one line,
`client = dreamkit.wrap(...)`, plus `with client.episode(...)` and `client.step(...)` for the plugins that read the
robot. Watch it in the panel (`dreamkit ui`, Live page) while it runs.

The simulator plays a real arm: dreamkit gets only what a real arm reports (joint and gripper state, the commands that
ran, the cameras, the model's timing), never positions or contacts. `--with-world` hands it the simulator's world
too, for evaluation only (dreamkit doctor --sim measures the monitor against it).

    python examples/sim_loop.py --mock                    # scripted arm, no API calls, no credits
    python examples/sim_loop.py --episodes 5              # Dream-1 (uses credits)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a checkout without installing

import dreamkit  # noqa: E402
from dreamkit import config as dk_config, scene, sim  # noqa: E402

TASKS = ("put apple into the bowl", "put orange into the bowl", "put banana on the plate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="put apple into the bowl")
    parser.add_argument("--episodes", type=int, default=3, help="scenes to try (retries come on top)")
    parser.add_argument("--seconds", type=float, default=40, help="simulated seconds per attempt")
    parser.add_argument("--actions", type=int, default=24, help="rows executed per model reply, 1-48")
    parser.add_argument("--model", default="dreamlabs/dream-1-beta")
    parser.add_argument("--config", default="dreamkit.yaml", help="dreamkit config (plugins, runs folder)")
    parser.add_argument("--sim", help="Dream Labs' yam_mujoco folder (default: sim_path from the config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=0.03, help="metres of random shift for the fruit")
    parser.add_argument("--mock", action="store_true", help="scripted arm instead of Dream-1: no API calls")
    parser.add_argument("--behaviors", default="dropped,success,released_off_target,success",
                        help="mock only: what goes wrong, one per attempt, cycled")
    parser.add_argument("--fast", action="store_true", help="no real-time pacing (tests)")
    parser.add_argument("--record", help="save the first attempt as a replay fixture in this folder")
    parser.add_argument("--fault-camera", choices=["top", "left", "right"], help="demo: this camera fails")
    parser.add_argument("--fault-kind", choices=["black", "frozen"], default="black")
    parser.add_argument("--fault-at", type=float, default=5.0, help="seconds into the attempt")
    parser.add_argument("--fault-episode", type=int, help="only in this attempt (1-based)")
    parser.add_argument("--light", action="store_true", help="fewer camera frames, less CPU")
    parser.add_argument("--viewer", action="store_true", help="open the simulator's own MuJoCo window: watch the arm "
                        "like a real one on the table (macOS: re-runs itself under mjpython)")
    parser.add_argument("--with-world", action="store_true", help="evaluation only: also give dreamkit the simulator's "
                        "world (positions, contacts); normally it gets what a real arm reports")
    parser.add_argument("--no-world", action="store_true", help=argparse.SUPPRESS)  # the default now; kept for old scripts
    parser.add_argument("--fake-rtt-ms", type=float, help="mock only: report this model round trip")
    parser.add_argument("--without", action="append", default=[], metavar="PLUGIN",
                        help="leave this plugin of the config out of this run (repeatable)")
    args = parser.parse_args(argv)

    if args.viewer and sys.platform == "darwin" and os.environ.get("DREAMKIT_UNDER_MJPYTHON") != "1":
        # MuJoCo's window needs mjpython on macOS (as Dream Labs' yam_mujoco.py does it)
        mjpython = Path(sys.executable).with_name("mjpython")
        os.environ["DREAMKIT_UNDER_MJPYTHON"] = "1"
        os.execv(str(mjpython), [str(mjpython), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])])

    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    yam_env, dc = sim.use(Path(args.sim).expanduser() if args.sim else cfg.sim_dir(base))
    env = yam_env.YamEnv()

    if args.mock:
        from dreamkit.mock import MockClient

        base_client = MockClient(env, behaviors=args.behaviors.split(","), seed=args.seed, fake_rtt_ms=args.fake_rtt_ms)
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
        yam_env.aim_viewer(viewer)  # Dream Labs' own starting view
        sync = dc.RunViewerSync(env)  # their sync: the viewer's reset key cannot rewind a run

    # ---- the one line dreamkit asks for ----------------------------------------------------------------------
    overrides = {"light": True} if args.light else {}
    if args.without:  # e.g. the panel's free demo: no plugin that calls an outside model
        overrides["plugins"] = dk_config.dump_plugins([s for s in cfg.plugins if s.name not in args.without])
    if args.with_world:
        overrides["evaluation"] = True  # the evaluation tools (success-check, world-judge) may load and judge
    if args.fault_camera:
        overrides["faults"] = {f"camera_{args.fault_kind}": args.fault_camera, "at_s": args.fault_at,
                               "episode": args.fault_episode}
    client = dreamkit.wrap(base_client, config=args.config, record=args.record, **overrides)
    client.policy(dc.policy_config(args.model, gateway, poses))
    render_every = client.config.render_every

    scene_index = 0
    try:
        while scene_index < args.episodes:
            env.reset()
            seed = args.seed + scene_index
            positions = scene.randomize(env, ["apple", "orange", "banana"], args.jitter, seed)
            if viewer is not None:
                sync(viewer)
            where = "  ".join(f"{k} ({v[0]:+.3f}, {v[1]:+.3f})" for k, v in positions.items() if k in ("apple", "orange", "banana"))
            print(f"scene {scene_index + 1}{' (retry)' if client.retry_requested else ''}, seed {seed}: {where}", flush=True)

            # fill the model's camera windows the way dream_control.main does
            windows = {name: deque(maxlen=dc.FRAMES) for name in dc.VIDEO_ROLES}
            steps = 0
            while any(len(w) < dc.FRAMES for w in windows.values()):
                state = env.state()
                env.step(state["left"], state["right"])
                images = dc.capture(env, windows, keep=steps % dc.FRAME_EVERY_STEPS == 0)
                steps += 1

            # a real arm reports only its state, cameras and the commands it ran; the world is for evaluation only
            world = {"env": env} if args.with_world else {}
            with client.episode(args.task, images=images, state=env.state(), seed=seed, positions=positions, **world):
                dc.feed(client, env, windows, args.task, args.model, steps, every=0, station=poses)
                while client.t < args.seconds:
                    reply = client.get_action()  # physics holds while the model thinks
                    left, right = dc.validate_chunk(env, reply, args.actions, verbose=False)
                    for l, r in zip(left, right):
                        tick = time.monotonic()
                        env.step(l, r)
                        kept = steps % dc.FRAME_EVERY_STEPS == 0
                        # the model needs a frame every 4th step; plugins and the Live page only when they ask
                        wanted = steps % render_every == 0 and client.wants_frames
                        images = dc.capture(env, windows, keep=kept) if kept or wanted else None
                        if kept:
                            dc.feed(client, env, windows, args.task, args.model, steps, station=poses)
                        steps += 1
                        client.step(state=env.state(), images=images, command=(l, r), **world)
                        if viewer is not None:
                            if not viewer.is_running():
                                raise KeyboardInterrupt  # the window was closed: stop like Ctrl-C
                            sync(viewer)
                        if client.t >= args.seconds:
                            break
                        if not args.fast:
                            time.sleep(max(0.0, 0.05 - (time.monotonic() - tick)))
            if client.stop_requested:  # the control plugin ended the run (e.g. budget-guard's budget used up)
                print("run stopped by a plugin: no new attempts", flush=True)
                break
            if not client.retry_requested:
                scene_index += 1
    except KeyboardInterrupt:
        client.close(status="stopped")
        print("stopped", flush=True)
        return 130
    finally:
        client.close()
        if viewer is not None:
            if viewer.is_running():
                print("run finished: close the MuJoCo window to exit", flush=True)
            while viewer.is_running():  # keep the last scene on screen until the window is closed
                sync(viewer)
                time.sleep(0.05)
            viewer.close()
        env.close()
    ok = sum(1 for e in client.episodes if e["success"])
    print(f"{ok}/{len(client.episodes)} attempts succeeded; run saved in {os.path.relpath(client.run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
