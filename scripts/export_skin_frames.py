import argparse, sys
import mujoco, numpy as np, yaml
sys.path.insert(0, ".")
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("prefix"); ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--fps", type=int, default=30); ap.add_argument("--t0", type=float, default=2.0); a = ap.parse_args()
imp = load_yaml("configs/impairment/patella.yaml"); d0 = ROOT / "results" / a.run
cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
sim = Sim(cfg); m, d, sv = sim.m, sim.d, sim.sv
sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); q0 = d.qpos[sv["qadr"]].copy()
ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), sim.name_to_k, q0); ctl.set_params(z["x"])
r = mujoco.Renderer(m, height=64, width=64); o = mujoco.MjvOption(); o.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
V = m.skin_vert.shape[0]
def skin_now(dd=None):
    r.update_scene(d if dd is None else dd, scene_option=o); return np.array(r.scene.skinvert, dtype=np.float32).reshape(-1, 3)
d2 = mujoco.MjData(m); mujoco.mj_resetData(m, d2); sim.limiter.clip_qpos(d2)
for i in range(int(1.0 / m.opt.timestep)):
    d2.ctrl[sv["act_ids"]] = q0
    if sim.head: d2.ctrl[sim.head_act] = sim.head_ctrl(d2)
    mujoco.mj_step(m, d2)
rest = skin_now(d2)
w = np.zeros((V, m.nskinbone), np.float32)
for b in range(m.nskinbone):
    a0, n0 = int(m.skin_bonevertadr[b]), int(m.skin_bonevertnum[b]); ids = m.skin_bonevertid[a0:a0 + n0]; w[ids, b] = m.skin_bonevertweight[a0:a0 + n0]
dom = np.argmax(w, 1); bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.skin_bonebodyid[b])) for b in range(m.nskinbone)]
vb = np.array([bname[k] for k in dom])
BLACK = np.array([0.045, 0.045, 0.05]); WHITE = np.array([0.93, 0.92, 0.90]); col = np.tile(WHITE, (V, 1))
X, Y, Zc = rest[:, 0], rest[:, 1], rest[:, 2]
def has(prefixes): return np.array([any(n.startswith(p) for p in prefixes) for n in vb])
head = has(("skull", "jaw", "eye")); neck = has(("C_",)); trunk = has(("torso", "L_", "T_", "pelvis", "rib", "scapula"))
tail = has(("Ca_")); fl = has(("upper_arm", "lower_arm", "hand", "finger")); hl = has(("upper_leg", "lower_leg", "foot", "toe"))
sk = rest[head]; xs = sk[:, 0]
x_head0, x_head1 = xs.min(), xs.max()
muzzle = head & (X > x_head0 + 0.62 * (x_head1 - x_head0)) & (Zc < rest[head][:, 2].mean() + 0.005)
col[head] = BLACK; col[muzzle | has(("jaw",))] = WHITE
col[head & (X > x_head0 + 0.55 * (x_head1 - x_head0)) & (np.abs(Y) < 0.006)] = WHITE
col[neck] = WHITE; ztop = np.percentile(Zc[neck], 70); col[neck & (Zc > ztop) & (X < np.percentile(X[neck], 55))] = BLACK
tz = Zc[trunk]; zlo, zhi = tz.min(), tz.max(); zr = (Zc - zlo) / (zhi - zlo + 1e-9)
xp, xt = rest[vb == "pelvis"][:, 0].mean(), rest[vb == "torso"][:, 0].mean()
col[trunk] = WHITE
saddle = trunk & (zr > 0.42) & (X < xt + 0.03); col[saddle] = BLACK
rump = trunk & (X < xp + 0.03) & (zr > 0.25); col[rump] = BLACK
col[trunk & (X >= xt + 0.03)] = np.where((zr[trunk & (X >= xt + 0.03)] > 0.8)[:, None], BLACK, WHITE)
col[tail] = BLACK; tail_x = rest[tail][:, 0]; col[tail & (X < np.percentile(tail_x, 55))] = WHITE
col[fl] = WHITE; col[hl] = WHITE; hlz = Zc[hl]; col[hl & (rest[:, 0] < xp - 0.0) & (Zc > np.percentile(hlz, 70)) & (vb == "upper_leg_L") | (hl & (Zc > np.percentile(hlz, 78)) & (rest[:, 0] < xp) & np.char.startswith(vb.astype(str), "upper_leg"))] = BLACK
FUR = {"torso": .013, "L_": .013, "T_": .013, "rib": .013, "pelvis": .011, "C_": .018, "scapula": .011, "upper_arm": .011, "lower_arm": .007,
       "hand": .004, "finger": .004, "upper_leg": .013, "lower_leg": .007, "foot": .004, "toe": .004, "Ca_": .024, "skull": .004, "jaw": .003}
off = np.zeros(V, np.float32)
for k, vv in FUR.items(): off[has((k,))] = vv
chest = trunk & (X > xt) & (zr < 0.75); off[chest] += 0.010; off[neck & (Zc < np.percentile(Zc[neck], 60))] += 0.012
adj = [[] for _ in range(V)]
for f_ in m.skin_face:
    for i_, j_ in ((0, 1), (1, 2), (2, 0)): adj[f_[i_]].append(f_[j_]); adj[f_[j_]].append(f_[i_])
for _ in range(10): off = np.array([0.5 * off[i] + 0.5 * np.mean(off[adj[i]]) if adj[i] else off[i] for i in range(V)], np.float32)
every = int(round(1.0 / (a.fps * m.opt.timestep))); F = []; txs = []
for i in range(int((a.t0 + a.seconds) / m.opt.timestep)):
    t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if t >= a.t0 and i % every == 0:
        F.append(skin_now()); txs.append(d.xpos[sim.torso][:2].copy())
np.save(a.prefix + "_verts.npy", np.array(F, dtype=np.float32))
np.savez(a.prefix + "_mesh.npz", faces=m.skin_face.astype(np.int32), rest=rest, colors=col.astype(np.float32), fur=off, bodies=vb, torso_x=np.array(txs), fps=a.fps, t0=a.t0)
print("wrote", a.prefix, np.array(F).shape, "faces", m.skin_face.shape)
