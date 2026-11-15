from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))


def ang2(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def sim(run):
    import mujoco, yaml
    from sim.config import ROOT, load_yaml
    from sim.gait import GaitSpec, TrotController
    from sim.rollout import Sim
    imp = load_yaml("configs/impairment/patella.yaml")
    d0 = ROOT / "results" / run
    cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    s = Sim(cfg); s.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    m, d, sv = s.m, s.d, s.sv
    mujoco.mj_resetData(m, d); s.limiter.clip_qpos(d)
    ctl = TrotController(GaitSpec(cfg["gait"], list(z["free_legs"])), s.name_to_k, d.qpos[sv["qadr"]].copy()); ctl.set_params(z["x"])
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    B = {n: bid(n) for n in ("torso", "pelvis", "C_7", "skull", "Ca_1", "Ca_21")}
    rec = {k: [] for k in B}; rec["fwd"] = []; rec["t"] = []
    for i in range(int(10.0 / m.opt.timestep)):
        t = i * m.opt.timestep; d.ctrl[sv["act_ids"]] = ctl.targets(t, min(1.0, t / cfg["gait"]["ramp_in"]))
        if s.head: d.ctrl[s.head_act] = s.head_ctrl(d)
        mujoco.mj_step(m, d)
        if t >= 2.0 and i % 10 == 0:
            for k, b in B.items(): rec[k].append(d.xpos[b].copy())
            rec["fwd"].append(d.xmat[B["torso"]].reshape(3, 3)[:, 0].copy()); rec["t"].append(t)
    P = {k: np.array(v) for k, v in rec.items()}
    t = P["t"]; tor = P["torso"]
    vel = np.polyfit(t, tor[:, :2], 1)[0]; travel = np.degrees(np.arctan2(vel[1], vel[0]))
    heading = np.degrees(np.arctan2(P["fwd"][:, 1], P["fwd"][:, 0])); head_mean = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(heading))))))
    yaw_drift = float(np.polyfit(t, np.unwrap(np.radians(heading)), 1)[0] * 180 / np.pi)
    side = (travel - head_mean + 180) % 360 - 180
    fh = P["fwd"][:, :2].copy(); fh /= np.linalg.norm(fh, axis=1, keepdims=True)
    S = lambda v: np.stack([(v[:, :2] * fh).sum(1), v[:, 2]], 1)
    back = S(P["torso"] - P["pelvis"]); head = S(P["skull"] - P["C_7"]); tail = S(P["Ca_21"] - P["Ca_1"])
    lat = np.cross(np.c_[fh, np.zeros(len(fh))], np.c_[P["Ca_21"][:, :2] - P["Ca_1"][:, :2], np.zeros(len(fh))])[:, 2]
    print(f"== {run}: speed {np.linalg.norm(vel):.2f} m/s | travel dir {travel:+.1f} deg, heading {head_mean:+.1f} deg -> sideslip {side:+.1f} deg, yaw drift {yaw_drift:+.1f} deg/s, lateral drift {vel[1]:+.3f} m/s")
    print("   head: abs elevation %.0f +- %.0f deg, rel back %.0f +- %.0f deg ; head height bob (skull z sd) %.1f mm"
          % (ang2(head).mean(), ang2(head).std(), (ang2(head) - ang2(back)).mean(), (ang2(head) - ang2(back)).std(), 1000 * P["skull"][:, 2].std()))
    print("   tail: abs elevation %.0f +- %.0f deg, rel back %.0f +- %.0f deg ; lateral wag (sd of tip offset) %.1f mm, tail length %.2f m"
          % (ang2(tail).mean(), ang2(tail).std(), (ang2(tail) - ang2(back)).mean(), (ang2(tail) - ang2(back)).std(), 1000 * lat.std(), np.linalg.norm(P["Ca_21"][0] - P["Ca_1"][0])))


def leo(path, fps):
    from leo_segments import presence, smooth_series
    z = np.load(path); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}
    segs, _, _ = presence(kp, ix, fps)
    print("Leo: head = neck_base->nose, tail = tail_base->tail_end, back = back_end->back_base (image side view; positive = above)")
    for a, b in segs:
        seg = kp[a:b + 1]; P = lambda nm: smooth_series(seg[:, ix[nm], :2], seg[:, ix[nm], 2])
        nose, nb, tb, te, bb, be = P("nose"), P("neck_base"), P("tail_base"), P("tail_end"), P("back_base"), P("back_end")
        face = np.sign((nose - tb).mean(0)[0]) or 1.0
        up = lambda v: np.stack([v[:, 0] * face, -v[:, 1]], 1)
        back, head, tail = up(bb - be), up(nose - nb), up(te - tb)
        t = np.arange(len(seg)) / fps; ctr = 0.5 * (bb + be)
        travel = np.polyfit(t, ctr[:, 0], 1)[0] * face
        L = float(np.median(np.linalg.norm(bb - be, axis=1)))
        hh = (-(nose[:, 1]) / L); hh = hh - hh.mean()
        print("  %5.1f-%5.1f s facing %-5s moves %s | head abs %.0f+-%.0f rel back %.0f+-%.0f | tail abs %.0f+-%.0f rel back %.0f+-%.0f | head bob sd %.3f BL"
              % (a / fps, b / fps, "right" if face > 0 else "left", "forward" if travel > 0 else "BACKWARD", ang2(head).mean(), ang2(head).std(),
                 (ang2(head) - ang2(back)).mean(), (ang2(head) - ang2(back)).std(), ang2(tail).mean(), ang2(tail).std(),
                 (ang2(tail) - ang2(back)).mean(), (ang2(tail) - ang2(back)).std(), hh.std()))


if __name__ == "__main__":
    if sys.argv[1] == "sim":
        for r in sys.argv[2:]: sim(r)
    else:
        leo(sys.argv[2], float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 120.0)
