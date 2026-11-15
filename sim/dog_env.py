from __future__ import annotations

import mujoco
import numpy as np
from dm_control import suite


def load_dog(timestep: float | None = None, integrator: str = "implicitfast", leo: dict | None = None):
    if leo is not None:
        from sim.dog_scale import build_leo_model
        m = build_leo_model(leo)
    else:
        env = suite.load("dog", "stand", task_kwargs=dict(random=0))
        m = env.physics.model.ptr
    if timestep is not None:
        m.opt.timestep = timestep
    m.opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{integrator.upper()}")
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    return m, d


def joint_actuator_ids(m) -> np.ndarray:
    return np.array([a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT])


def make_position_servos(m, kp_scale: float = 1.0, kd_s: float = 0.05, digit_kp: float | None = None,
                         digit_kd: float = 0.3) -> dict:
    ids = joint_actuator_ids(m)
    gear = m.actuator_gainprm[ids, 0].copy()
    jid = m.actuator_trnid[ids, 0]
    for a, g, j in zip(ids, gear, jid):
        kp = kp_scale * g
        kd = kd_s * kp
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or ""
        if digit_kp is not None and jname.startswith(("toe_", "finger_")):
            kp, kd = digit_kp, digit_kd
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[a, :3] = [kp, 0.0, 0.0]
        m.actuator_biasprm[a, :3] = [0.0, -kp, -kd]
        lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
        m.actuator_ctrllimited[a] = 1
        m.actuator_ctrlrange[a] = [lo, hi]
        m.actuator_forcelimited[a] = 1
        m.actuator_forcerange[a] = [-g, g]
    return dict(act_ids=ids, joint_ids=jid, qadr=m.jnt_qposadr[jid], gear=gear)


def body_id(m, name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"no body named {name!r}")
    return i


def joint_id(m, name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise KeyError(f"no joint named {name!r}")
    return i
