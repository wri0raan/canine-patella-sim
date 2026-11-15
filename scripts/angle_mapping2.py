import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.dog_env import body_id, joint_id, load_dog
from sim.kinematics import hind_limb_points

m, d = load_dog()
mujoco.mj_kinematics(m, d)
print("torso xmat diag:", np.round(np.diag(d.xmat[body_id(m, 'torso')].reshape(3, 3)), 2))

jk = joint_id(m, "knee_L")
adr = m.jnt_qposadr[jk]
print("knee axis (local):", m.jnt_axis[jk], " range deg:", np.degrees(m.jnt_range[jk]))


def proj(v):
    return np.array([v[0], v[2]])


def signed_deg(a, b):
    return np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))


print("\n q(deg) | hip->stifle (x,z) | stifle->hock (x,z) | |stifle angle| (proj) signed | stifle z-hip z | hock z")
for q in [10, 0, -30, -60, -90, -120]:
    d.qpos[adr] = np.radians(q)
    mujoco.mj_kinematics(m, d)
    p = hind_limb_points(m, d, "L")
    fem = proj(p["hip"] - p["stifle"])
    tib = proj(p["hock"] - p["stifle"])
    ang = signed_deg(fem, tib)
    print(f"{q:7.1f} | {np.round(proj(p['stifle']-p['hip']),3)} | {np.round(proj(p['hock']-p['stifle']),3)} | "
          f"{abs(ang):6.1f} ({ang:+7.1f}) | dz_stifle={p['stifle'][2]-p['hip'][2]:+.3f} hock_z={p['hock'][2]:.3f}")
