import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.dog_env import joint_id, load_dog
from sim.kinematics import hind_limb_angles

m, d = load_dog()
mujoco.mj_kinematics(m, d)
print("default pose (side L):", {k: round(v, 1) for k, v in hind_limb_angles(m, d, "L").items()})
print("default pose (side R):", {k: round(v, 1) for k, v in hind_limb_angles(m, d, "R").items()})


def sweep(joint: str, key: str, values_deg):
    j = joint_id(m, joint)
    adr = m.jnt_qposadr[j]
    q0 = d.qpos[adr]
    rows = []
    for v in values_deg:
        d.qpos[adr] = np.radians(v)
        mujoco.mj_kinematics(m, d)
        rows.append((v, hind_limb_angles(m, d, "L")[key]))
    d.qpos[adr] = q0
    mujoco.mj_kinematics(m, d)
    return rows


print("\nknee_L joint (deg) -> stifle included angle (deg)")
for v, a in sweep("knee_L", "stifle", [10, 0, -20, -40, -60, -80, -100, -120]):
    print(f"  q={v:6.1f}  stifle={a:6.1f}")
print("\nankle_L joint (deg) -> hock included angle (deg)")
for v, a in sweep("ankle_L", "hock", [20, 0, -20, -40, -60, -80, -100]):
    print(f"  q={v:6.1f}  hock={a:6.1f}")
print("\nhip_L_extend joint (deg) -> hip angle (deg)")
for v, a in sweep("hip_L_extend", "hip", [60, 30, 0, -30, -60, -90, -130]):
    print(f"  q={v:6.1f}  hip={a:6.1f}")
