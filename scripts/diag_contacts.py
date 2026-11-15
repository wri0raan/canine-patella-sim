import sys
from collections import defaultdict
from pathlib import Path
import mujoco, numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
imp = load_yaml("configs/impairment/patella.yaml")
for run in sys.argv[1:]:
    d0 = ROOT / "results" / run
    cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    m, d, sv = sim.m, sim.d, sim.sv
    spec = GaitSpec(cfg["gait"], list(z["free_legs"]))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d)
    ctl = TrotController(spec, sim.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
    force = defaultdict(float); f6 = np.zeros(6)
    for i in range(int(8.0 / m.opt.timestep)):
        t = i * m.opt.timestep
        d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if t < 2.0: continue
        for c in range(d.ncon):
            con = d.contact[c]; g1, g2 = con.geom1, con.geom2
            for g, o in ((g1, g2), (g2, g1)):
                if m.geom_bodyid[o] == 0 and m.geom_bodyid[g] != 0:
                    mujoco.mj_contactForce(m, d, c, f6)
                    b = m.body(m.geom_bodyid[g]).name
                    force[b] += abs(f6[0])
    tot = sum(force.values())
    print(f"== {run}  total contact-force share by body (>1%):")
    for b, f in sorted(force.items(), key=lambda kv: -kv[1]):
        if f / tot > 0.01: print(f"  {b:28s} {100*f/tot:5.1f}%")
