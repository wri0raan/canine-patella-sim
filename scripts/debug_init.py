import mujoco
import numpy as np
from dm_control import suite

env = suite.load("dog", "stand", task_kwargs=dict(random=0))
env.reset()
m = env.physics.model.ptr
d = env.physics.data.ptr
torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
print("qpos root:", np.round(d.qpos[:7], 3), "qvel root:", np.round(d.qvel[:6], 3))
print("n contacts:", d.ncon, "time:", d.time)
foot_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
lowest = min(range(m.nbody), key=lambda b: d.xpos[b][2])
print("lowest body:", foot_names[lowest], round(float(d.xpos[lowest][2]), 3))
for name in ["foot_L", "foot_R", "toe_L", "toe_R", "hand_L", "hand_R", "finger_L", "upper_leg_L", "lower_leg_L"]:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    print(f"{name:12s} xyz={np.round(d.xpos[b], 3)}")
for i in range(200):
    mujoco.mj_step(m, d)
    if i % 20 == 0:
        print(f"t={d.time:.2f} z={d.xpos[torso][2]:.3f} vz={d.qvel[2]:.3f} ncon={d.ncon}")
