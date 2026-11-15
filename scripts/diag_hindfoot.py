import sys
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
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    rows = []
    for i in range(int(8.0 / m.opt.timestep)):
        t = i * m.opt.timestep
        d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if t < 2.0 or i % 10: continue
        for s in ("L", "R"):
            hock, mtp = d.xpos[bid(f"foot_{s}")], d.xpos[bid(f"toe_{s}")]
            on = any((m.geom_bodyid[c.geom1] == bid(f"toe_{s}") or m.geom_bodyid[c.geom2] == bid(f"toe_{s}")) for c in d.contact[:d.ncon])
            v = mtp - hock
            rows.append((on, hock[2], mtp[2], np.degrees(np.arctan2(-v[2], np.hypot(v[0], v[1]))), np.linalg.norm(v)))
    r = np.array(rows, float); st = r[:, 0] > 0
    print(f"== {run}: stance fraction {st.mean():.2f}")
    print(f"  hock(foot body) height stance mean {r[st,1].mean()*100:.1f} cm, min {r[st,1].min()*100:.1f} cm | MTP height {r[st,2].mean()*100:.1f} cm")
    print(f"  metatarsus incl. stance mean {r[st,3].mean():.0f} deg (min {r[st,3].min():.0f}, p10 {np.percentile(r[st,3],10):.0f}); length {r[:,4].mean()*100:.1f} cm")
