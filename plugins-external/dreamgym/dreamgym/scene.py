"""Reading and arranging the YAM sim scene: where things are, whether a fruit is in the bowl, moving objects.

Positions are read live from MuJoCo (`data.body(name).xpos`), never hard-coded, so a changed scene still works.
World frame: Z up, table top at about 0.76 m, arms mounted at y = 0.24.
"""

from __future__ import annotations

import numpy as np

## Free-moving objects of the scene and their free joints (yam_env.build_xml).
FREE_JOINTS = {
    "apple": "apple_free",
    "banana": "banana_free",
    "orange": "orange_free",
    "water_bottle": "bottle_free",
}
FRUITS = ("apple", "banana", "orange")
TABLE_Z = 0.76
## Table extent in x and y (yam_env: box at (0, -0.12) with half sizes 0.65 x 0.55).
TABLE_X = (-0.65, 0.65)
TABLE_Y = (-0.67, 0.43)


def pos(env, body: str) -> np.ndarray:
    return env.data.body(body).xpos.copy()


def speed(env, obj: str) -> float:
    """Linear speed of a free object, m/s."""
    return float(np.linalg.norm(env.data.joint(FREE_JOINTS[obj]).qvel[:3]))


def tcp_distance(env, obj: str) -> float:
    """Distance from the object to the nearer gripper's tool centre point, m."""
    p = pos(env, obj)
    return min(float(np.linalg.norm(env.data.site(f"{side}_tcp").xpos - p)) for side in ("left", "right"))


def horizontal(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def in_bowl(env, obj: str) -> bool:
    """Object centre inside the bowl's rim and below its top (bowl rim radius ~0.08 m, height ~0.058 m)."""
    b, p = pos(env, "bowl"), pos(env, obj)
    return horizontal(b, p) < 0.072 and b[2] < p[2] < b[2] + 0.09


def on_plate(env, obj: str) -> bool:
    """Object centre over the plate (radius ~0.069 m) and resting low on it."""
    plate, p = pos(env, "plate"), pos(env, obj)
    return horizontal(plate, p) < 0.068 and plate[2] < p[2] < plate[2] + 0.06


def off_table(env, obj: str) -> bool:
    p = pos(env, obj)
    return p[2] < TABLE_Z - 0.05 or not (TABLE_X[0] < p[0] < TABLE_X[1] and TABLE_Y[0] < p[1] < TABLE_Y[1])


def at_rest(env, obj: str, tolerance: float = 0.03) -> bool:
    return speed(env, obj) < tolerance


def released(env, obj: str, distance: float = 0.05) -> bool:
    """Neither gripper is still holding the object."""
    return tcp_distance(env, obj) > distance


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def place(env, obj: str, xy: np.ndarray, yaw: float = 0.0, lift: float = 0.004) -> None:
    """Moves a free object to `xy` (keeping its start height plus `lift`) turned by `yaw` about Z from its start.

    The applied yaw is kept in `env.dreamgym_yaw[obj]` so a scripted policy can grasp along it.
    """
    import mujoco

    joint = env.data.joint(FREE_JOINTS[obj])
    start = env.model.joint(FREE_JOINTS[obj]).qpos0
    joint.qpos[:3] = [xy[0], xy[1], start[2] + lift]
    turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    joint.qpos[3:7] = _quat_mul(turn, np.asarray(start[3:7]))
    joint.qvel[:] = 0.0
    if not hasattr(env, "dreamgym_yaw"):
        env.dreamgym_yaw = {}
    env.dreamgym_yaw[obj] = float(yaw)
    mujoco.mj_forward(env.model, env.data)


def start_xy(env, obj: str) -> np.ndarray:
    return np.asarray(env.model.joint(FREE_JOINTS[obj]).qpos0[:2], dtype=float)


def scale_lights(env, factor: float) -> None:
    """Scales every light's diffuse colour from its original value (domain randomisation of lighting)."""
    base = getattr(env, "dreamgym_light_base", None)
    if base is None:
        base = env.model.light_diffuse.copy()
        env.dreamgym_light_base = base
    env.model.light_diffuse[:] = np.clip(base * factor, 0.0, 1.0)
