import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.dog_env import body_id, load_dog, make_position_servos

kp_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
kd_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
DURATION = 5.0

m, d = load_dog(timestep=dt)
sv = make_position_servos(m, kp_scale=kp_scale, kd_s=kd_s)
q_hold = d.qpos[sv["qadr"]].copy()
torso = body_id(m, "torso")
print(f"kp_scale={kp_scale} kd_s={kd_s} dt={m.opt.timestep} integrator={m.opt.integrator} servos={len(sv['act_ids'])}")
print(f"init torso z={d.xpos[torso][2]:.3f}")

z_hist, up_hist = [], []
n = int(DURATION / m.opt.timestep)
for step in range(n):
    d.ctrl[sv["act_ids"]] = q_hold
    mujoco.mj_step(m, d)
    up = d.xmat[torso].reshape(3, 3)[2, 2]
    z_hist.append(d.xpos[torso][2])
    up_hist.append(up)
    if step % int(1.0 / m.opt.timestep) == 0:
        print(f"t={d.time:4.1f}s torso z={z_hist[-1]:.3f} up={up:.2f} |v|={np.linalg.norm(d.qvel[:3]):.3f} ncon={d.ncon}")
    if not np.isfinite(d.qpos).all():
        print("NaN - simulation unstable")
        break
z = np.array(z_hist)
tail = z[-int(1.0 / m.opt.timestep):]
print(f"final z={z[-1]:.3f}  last-1s mean={tail.mean():.3f} std={tail.std():.4f}  min up={min(up_hist):.2f}")
print("PASS (stands)" if (tail.mean() > 0.30 and min(up_hist) > 0.8) else "FAIL (collapsed or tipped)")
