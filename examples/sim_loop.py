"""A user's control loop on Dream Labs' simulator, with dreamkit plugged in.

This is dream_control.main's loop written the way a Dream-1 user writes theirs: Dream Labs' own functions
(policy_config, feed, capture, validate_chunk) drive the simulator unchanged; dreamkit adds one line,
`client = dreamkit.wrap(...)`, plus `with client.episode(...)` and `client.step(...)` for the plugins that read the
robot. Watch it in the panel (`dreamkit ui`, Live page) while it runs.

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

import numpy as np

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
    parser.add_argument("--fake-rtt-ms", type=float, help="mock only: report this model round trip")
    args = parser.parse_args(argv)

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

    # ---- the one line dreamkit asks for ----------------------------------------------------------------------
    client = dreamkit.wrap(base_client, config=args.config, record=args.record)
    client.policy(dc.policy_config(args.model, gateway, poses))
    render_every = client.config.render_every

    scene_index = 0
    try:
        while scene_index < args.episodes:
            env.reset()
            seed = args.seed + scene_index
            positions = scene.randomize(env, ["apple", "orange", "banana"], args.jitter, seed)
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

            with client.episode(args.task, env=env, images=images, seed=seed, positions=positions):
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
                        client.step(env, images=images, command=(l, r))
                        if client.t >= args.seconds:
                            break
                        if not args.fast:
                            time.sleep(max(0.0, 0.05 - (time.monotonic() - tick)))
            if not client.retry_requested:
                scene_index += 1
    except KeyboardInterrupt:
        client.close(status="stopped")
        print("stopped", flush=True)
        return 130
    finally:
        client.close()
        env.close()
    ok = sum(1 for e in client.episodes if e["success"])
    print(f"{ok}/{len(client.episodes)} attempts succeeded; run saved in {os.path.relpath(client.run_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
