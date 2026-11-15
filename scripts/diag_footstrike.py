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
    side = {"L": 0, "R": 1}
    prev = {l: False for l in ("LF", "RF", "LH", "RH")}; tdstep = {}; first = defaultdict(list); early = defaultdict(lambda: defaultdict(float)); f6 = np.zeros(6)
    leg_of_body = {}
    for b in range(m.nbody):
        n = m.body(b).name
        for pre, leg in (("hand_", "F"), ("finger_", "F"), ("foot_", "H"), ("toe_", "H")):
            if n.startswith(pre) and n[-1] in "LR": leg_of_body[b] = (n[-1] + leg, pre[:-1])
    for i in range(int(8.0 / m.opt.timestep)):
        t = i * m.opt.timestep
        d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if t < 2.0: continue
        cont = defaultdict(lambda: defaultdict(float))
        for c in range(d.ncon):
            con = d.contact[c]
            for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                if m.geom_bodyid[o] == 0 and m.geom_bodyid[g] in leg_of_body:
                    mujoco.mj_contactForce(m, d, c, f6); leg, part = leg_of_body[m.geom_bodyid[g]]; cont[leg][part] += abs(f6[0])
        for leg in prev:
            on = sum(cont[leg].values()) > 2.0
            if on and not prev[leg]:
                tdstep[leg] = i; first[leg].append(max(cont[leg], key=cont[leg].get))
            if on and leg in tdstep and i - tdstep[leg] <= 15:
                for part, fv in cont[leg].items(): early[leg][part] += fv
            prev[leg] = on
    print(f"== {run}: part carrying most force at touchdown instant / first 30 ms force share")
    for leg in ("LF", "RF", "LH", "RH"):
        tot = sum(early[leg].values()) or 1.0
        print(f"  {leg}: first-contact {dict((p, first[leg].count(p)) for p in set(first[leg]))}  early share " + ", ".join(f"{p} {100*v/tot:.0f}%" for p, v in early[leg].items()))
