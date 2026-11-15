from __future__ import annotations

import mujoco
import numpy as np

from sim.dog_env import joint_id
from sim.kinematics import hind_limb_angles

LEG_SIDE = {"LH": "L", "RH": "R"}


def calibrate_knee_map(m, side: str) -> tuple[float, float]:
    d = mujoco.MjData(m)
    j = joint_id(m, f"knee_{side}")
    adr = m.jnt_qposadr[j]
    qs = np.linspace(-110, 5, 12)
    th = []
    for q in qs:
        mujoco.mj_resetData(m, d)
        d.qpos[adr] = np.radians(q)
        mujoco.mj_kinematics(m, d)
        th.append(hind_limb_angles(m, d, side)["stifle"])
    a, b = np.polyfit(qs, th, 1)
    return float(a), float(b)


class KneeLimiter:
    def __init__(self, m, sv):
        self.m = m
        self.sv = sv
        self.maps = {s: calibrate_knee_map(m, s) for s in ("L", "R")}
        self.orig = {}

    def theta_to_q(self, side: str, theta_deg: float) -> float:
        a, b = self.maps[side]
        return float(np.radians((theta_deg - b) / a))

    def apply(self, legs, ext_deg: float, flex_deg: float):
        for leg in legs:
            side = LEG_SIDE[leg]
            j = joint_id(self.m, f"knee_{side}")
            k = int(np.where(self.sv["joint_ids"] == j)[0][0])
            act = int(self.sv["act_ids"][k])
            if j not in self.orig:
                self.orig[j] = (self.m.jnt_range[j].copy(), self.m.actuator_ctrlrange[act].copy(), act)
            lo, hi = self.theta_to_q(side, flex_deg), self.theta_to_q(side, ext_deg)
            self.m.jnt_range[j] = [lo, hi]
            self.m.actuator_ctrlrange[act] = [lo, hi]

    def restore(self):
        for j, (jr, cr, act) in self.orig.items():
            self.m.jnt_range[j] = jr
            self.m.actuator_ctrlrange[act] = cr
        self.orig = {}

    def set_grade(self, imp_cfg: dict, grade: int, mode: str, affected=("LH", "RH")):
        for leg in ("LH", "RH"):
            g = imp_cfg["grades"][grade if leg in affected else 0]
            ext = min(g["prom_ext"], g[imp_cfg["modes"][mode]])
            self.apply([leg], ext_deg=ext, flex_deg=g["prom_flex"])

    def clip_qpos(self, d):
        for side in ("L", "R"):
            j = joint_id(self.m, f"knee_{side}")
            adr = self.m.jnt_qposadr[j]
            d.qpos[adr] = np.clip(d.qpos[adr], *self.m.jnt_range[j])
