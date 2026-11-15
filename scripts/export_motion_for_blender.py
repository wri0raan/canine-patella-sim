import argparse, sys
import mujoco, numpy as np, yaml
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--t0", type=float, default=2.0); a = ap.parse_args()
imp = load_yaml("configs/impairment/patella.yaml"); d0 = ROOT / "results" / a.run
cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv
sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q0 = d.qpos[sv["qadr"]].copy()
ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, q0); ctl.set_params(z["x"])
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
BODIES = ["pelvis", "L_5", "L_1", "torso", "C_7", "C_1", "skull", "Ca_1", "Ca_5", "Ca_10", "Ca_15", "Ca_21"]
for s in "LR":
    BODIES += [f"upper_arm_{s}", f"lower_arm_{s}", f"hand_anchor_{s}", f"finger_{s}", f"upper_leg_{s}", f"lower_leg_{s}", f"foot_{s}", f"foot_anchor_{s}"]
tipof = {}
for (b, lv), nm in zip(sim.dig, ("finger_L", "finger_R", "toe_L", "toe_R")): tipof[nm] = (b, lv)
NAMES = BODIES + list(tipof)
every = int(round(1.0 / (a.fps * m.opt.timestep))); P = []; R = []; t_s = []
for i in range(int((a.t0 + a.seconds) / m.opt.timestep)):
    t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if t >= a.t0 and i % every == 0:
        pos = [d.xpos[bid(n)].copy() for n in BODIES] + [d.xpos[b] + d.xmat[b].reshape(3, 3) @ lv for b, lv in tipof.values()]
        P.append(np.array(pos)); R.append(d.xmat[bid("pelvis")].reshape(3, 3).copy()); t_s.append(t)
np.savez(a.out, names=np.array(NAMES), pos=np.array(P), pelvis_R=np.array(R), t=np.array(t_s), fps=a.fps)
print("wrote", a.out, np.array(P).shape)
