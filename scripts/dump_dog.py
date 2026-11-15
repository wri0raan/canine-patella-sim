import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

variant = sys.argv[1] if len(sys.argv) > 1 else "suite"
out = Path(__file__).resolve().parents[1] / "results"
out.mkdir(exist_ok=True)

if variant == "suite":
    from dm_control import suite

    env = suite.load("dog", "stand")
    model = env.physics.model.ptr
else:
    from dm_control.locomotion.walkers import dog as dog_walker

    walker = dog_walker.Dog() if hasattr(dog_walker, "Dog") else dog_walker.DogV2()
    from dm_control import mjcf

    model = mujoco.MjModel.from_xml_string(
        walker.mjcf_model.to_xml_string(), walker.mjcf_model.get_assets()
    )

JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
name = lambda t, i: mujoco.mj_id2name(model, t, i) or f"<{i}>"

joints = []
for j in range(model.njnt):
    bid = model.jnt_bodyid[j]
    lo, hi = model.jnt_range[j]
    joints.append(
        dict(
            joint=name(mujoco.mjtObj.mjOBJ_JOINT, j),
            type=JT[int(model.jnt_type[j])],
            body=name(mujoco.mjtObj.mjOBJ_BODY, bid),
            axis=np.round(model.jnt_axis[j], 2).tolist(),
            limited=bool(model.jnt_limited[j]),
            range_deg=[round(float(np.degrees(lo)), 1), round(float(np.degrees(hi)), 1)],
            stiffness=float(model.jnt_stiffness[j]),
            damping=float(model.dof_damping[model.jnt_dofadr[j]]) if model.jnt_type[j] != 1 else np.nan,
        )
    )
jdf = pd.DataFrame(joints)

acts = []
for a in range(model.nu):
    trn_id = model.actuator_trnid[a][0]
    is_tendon = model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON
    acts.append(
        dict(
            actuator=name(mujoco.mjtObj.mjOBJ_ACTUATOR, a),
            joint=name(mujoco.mjtObj.mjOBJ_TENDON if is_tendon else mujoco.mjtObj.mjOBJ_JOINT, trn_id),
            transmission="tendon" if is_tendon else "joint",
            gain=model.actuator_gainprm[a][:3].tolist(),
            bias=model.actuator_biasprm[a][:3].tolist(),
            ctrlrange_deg=np.round(np.degrees(model.actuator_ctrlrange[a]), 1).tolist(),
            forcerange=model.actuator_forcerange[a].tolist(),
        )
    )
adf = pd.DataFrame(acts)

bodies = []
for b in range(model.nbody):
    bodies.append(
        dict(
            body=name(mujoco.mjtObj.mjOBJ_BODY, b),
            parent=name(mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b]),
            mass=float(model.body_mass[b]),
            pos=np.round(model.body_pos[b], 4).tolist(),
        )
    )
bdf = pd.DataFrame(bodies)

jdf.to_csv(out / f"dog_dump_{variant}_joints.csv", index=False)
adf.to_csv(out / f"dog_dump_{variant}_actuators.csv", index=False)
bdf.to_csv(out / f"dog_dump_{variant}_bodies.csv", index=False)

pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_colwidth", 60)
print(f"[{variant}] nbody={model.nbody} njnt={model.njnt} nu={model.nu} nq={model.nq} nv={model.nv}")
print(f"total mass = {model.body_mass.sum():.2f} kg, timestep = {model.opt.timestep}")
print("\n--- joints ---")
print(jdf[["joint", "type", "body", "axis", "range_deg"]].to_string())
print("\n--- actuators ---")
print(adf[["actuator", "joint", "gain", "bias", "ctrlrange_deg"]].to_string())
