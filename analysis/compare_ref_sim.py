import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, yaml, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
from ref_video_gait import analyse, FPS
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
out_png, run = sys.argv[1], sys.argv[2]; clip = sys.argv[3] if len(sys.argv) > 3 else "data/ref/youtube/work/gsd_trot_a_kp.npz"
imp = load_yaml("configs/impairment/patella.yaml")
def ang(a, b, c):
    v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)), -1, 1)))
ref, f, L = analyse(clip); per = int(np.median([np.diff(o["td"]).mean() for o in ref.values() if len(o["td"]) > 4]))
def fold(sig, td, n=60):
    cyc = [np.interp(np.linspace(td[i], td[i + 1], n), np.arange(len(sig)), sig) for i in range(len(td) - 1) if 0.3 < (td[i + 1] - td[i]) / FPS < 0.8]
    return np.array(cyc)
real = {"carpus (fore)": [fold(ref[l]["distal"], ref[l]["td"]) for l in ("LF", "RF")], "hock (hind)": [fold(ref[l]["distal"], ref[l]["td"]) for l in ("LH", "RH")]}
d0 = ROOT / "results" / run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
m, d, sv = sim.m, sim.d, sim.sv; spec = GaitSpec(cfg["gait"], list(z["free_legs"]))
mujoco.mj_resetData(m, d); sim.limiter.clip_qpos(d); ctl = TrotController(spec, sim.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
J = {"LF": ("elbow_L", "wrist_L", "finger_L"), "RF": ("elbow_R", "wrist_R", "finger_R"), "LH": ("knee_L", "ankle_L", "toe_L"), "RH": ("knee_R", "ankle_R", "toe_R")}
gidx = {"LF": 0, "RF": 1, "LH": 2, "RH": 3}
A = {l: [] for l in J}; G = {l: [] for l in J}; tt = []
for i in range(int(10.0 / m.opt.timestep)):
    t = i * m.opt.timestep
    d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
    if sim.head: d.ctrl[sim.head_act] = sim.head_ctrl(d)
    mujoco.mj_step(m, d)
    if i % 5 == 0 and t > 2:
        tt.append(t)
        Rt = d.xmat[sim.torso].reshape(3, 3); fwd = Rt[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
        P2 = lambda p: np.array([p @ fwd, p[2]])
        for l, (a, b, c) in J.items():
            bd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c); gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bd]
            tip = max((d.geom_xpos[g] for g in gs), key=lambda q_: np.linalg.norm(q_ - d.xanchor[jid(c)]))
            A[l].append(ang(P2(d.xanchor[jid(a)]), P2(d.xanchor[jid(b)]), P2(tip)))
        F = np.zeros(4); f6 = np.zeros(6)
        for ci in range(d.ncon):
            con = d.contact[ci]
            for g, o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                if m.geom_bodyid[o] == 0 and sim.geom_leg[g] >= 0:
                    mujoco.mj_contactForce(m, d, ci, f6); F[sim.geom_leg[g]] += f6[0]
        for l in J: G[l].append(F[gidx[l]])
tt = np.array(tt); simc = {}; f0 = float(ctl.f)
from scipy.signal import medfilt
for l in J:
    st = medfilt((np.array(G[l]) > 2.0).astype(float), 7) > 0.5; tdi = np.nonzero(~st[:-1] & st[1:])[0] + 1
    ph_all = (tt * f0) % 1.0; tph = np.angle(np.mean(np.exp(2j * np.pi * ph_all[tdi]))) / (2 * np.pi) % 1.0
    rel = (ph_all - tph) % 1.0; sig = np.array(A[l]); bins = np.minimum((rel * 60).astype(int), 59)
    simc[l] = np.array([[sig[(bins == k) & (np.arange(len(sig)) // int(1 / f0 / 0.01) == c)].mean() if ((bins == k) & (np.arange(len(sig)) // int(1 / f0 / 0.01) == c)).any() else np.nan for k in range(60)] for c in range(int(len(sig) / (1 / f0 / 0.01)))])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); ph = np.linspace(0, 1, 60)
for ax, key, legs in zip(axs, real, (("LF", "RF"), ("LH", "RH"))):
    R = np.vstack(real[key]); ax.plot(ph, R.mean(0), "k-", label="real GSD (video)"); ax.fill_between(ph, R.mean(0) - R.std(0), R.mean(0) + R.std(0), color="k", alpha=0.15)
    S = np.vstack([simc[l] for l in legs]); S = S[~np.isnan(S).any(1)]; ax.plot(ph, S.mean(0), "r-", label=f"sim {run}"); ax.fill_between(ph, S.mean(0) - S.std(0), S.mean(0) + S.std(0), color="r", alpha=0.15)
    ax.set_title(f"{key} angle (deg), cycle from touchdown", fontsize=9); ax.set_xlabel("stride phase (0 = touchdown)")
    print(key, f"real min/max {R.mean(0).min():.0f}/{R.mean(0).max():.0f}  at phase {ph[R.mean(0).argmin()]:.2f}/{ph[R.mean(0).argmax()]:.2f} | sim min/max {S.mean(0).min():.0f}/{S.mean(0).max():.0f} at {ph[S.mean(0).argmin()]:.2f}/{ph[S.mean(0).argmax()]:.2f}")
axs[0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(out_png, dpi=150); print("saved", out_png)
