import sys
from pathlib import Path
import mujoco, numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
imp = load_yaml("configs/impairment/patella.yaml")
run0 = sys.argv[1]
for run in sys.argv[1:]:
    d0 = ROOT / "results" / run
    cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    m, d, sv = sim.m, sim.d, sim.sv
    spec = GaitSpec(cfg["gait"], list(z["free_legs"]))
    mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d)
    ctl = TrotController(spec, sim.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    legs = {"LH": ("toe_L", "foot_L"), "RH": ("toe_R", "foot_R"), "LF": ("finger_L", "hand_L"), "RF": ("finger_R", "hand_R")}
    info = {}
    for leg, (dig, prox) in legs.items():
        b = bid(dig); gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        info[leg] = (b, bid(prox), gs)
    def segs(leg):
        b, pb, gs = info[leg]; o = d.xpos[b]
        tip = max((d.geom_xpos[g] for g in gs), key=lambda p: np.linalg.norm(p - o))
        v = tip - o; return np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))), tip[2] - 0.0
    rec = {l: [] for l in legs}; prev = {l: False for l in legs}; stance_t = {l: None for l in legs}
    for i in range(int(8.0 / m.opt.timestep)):
        t = i * m.opt.timestep
        d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
        mujoco.mj_step(m, d)
        if t < 2.0: continue
        for leg in legs:
            b = info[leg][0]
            on = any(((m.geom_bodyid[c.geom1] == b) or (m.geom_bodyid[c.geom2] == b)) and (m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0) for c in d.contact[:d.ncon])
            pitch, _ = segs(leg)
            if on and not prev[leg]: stance_t[leg] = i; rec[leg].append(dict(td=pitch, mid=[], n=0))
            if on and rec[leg]: rec[leg][-1]["mid"].append(pitch)
            prev[leg] = on
    print(f"== {run}: digit pitch (deg, 0 flat, <0 tip down)")
    for leg in legs:
        r = rec[leg][2:]
        if not r: continue
        td = np.mean([x["td"] for x in r]); mid = np.mean([np.median(x["mid"]) for x in r if x["mid"]]); 
        print(f"  {leg}: touchdown {td:+.0f}  stance median {mid:+.0f}")
