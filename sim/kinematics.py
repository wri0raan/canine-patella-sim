from __future__ import annotations

import mujoco
import numpy as np

from sim.dog_env import body_id, joint_id


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def hind_limb_points(m, d, side: str = "L") -> dict[str, np.ndarray]:
    a = lambda name: d.xanchor[joint_id(m, name)].copy()
    return dict(
        hip=a(f"hip_{side}_extend"),
        stifle=a(f"knee_{side}"),
        hock=a(f"ankle_{side}"),
        toe=a(f"toe_{side}"),
        pelvis=d.xpos[body_id(m, "pelvis")].copy(),
        torso=d.xpos[body_id(m, "torso")].copy(),
    )


def hind_limb_angles(m, d, side: str = "L") -> dict[str, float]:
    p = hind_limb_points(m, d, side)
    return dict(
        stifle=_angle(p["hip"] - p["stifle"], p["hock"] - p["stifle"]),
        hock=_angle(p["stifle"] - p["hock"], p["toe"] - p["hock"]),
        hip=_angle(p["torso"] - p["pelvis"], p["stifle"] - p["hip"]),
    )
