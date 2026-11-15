import argparse, sys
import mujoco, numpy as np
sys.path.insert(0, ".")
from sim.config import load_yaml
from sim.dog_scale import apply_leo
from sim.rollout import Sim

ap = argparse.ArgumentParser(); ap.add_argument("--kp", nargs="+", default=["data/real/leo2/work/photo9_kp.npz", "data/real/leo2/work/photos_kp.npz"]); ap.add_argument("--stifle", type=float, default=16.0); ap.add_argument("--hock", type=float, default=8.5)
a = ap.parse_args()
ap.set_defaults(); 
def photo_lengths(kpf, names, min_conf=0.45):
    ix = {n: i for i, n in enumerate(names)}
    g = lambda n: kpf[ix[n], :2]; c = lambda n: kpf[ix[n], 2]
    best = None
    for hs, fs in (("back_left", "front_left"), ("back_right", "front_right")):
        cs = min(c(hs + "_paw"), c(hs + "_knee"), c(hs + "_thai"), c(fs + "_paw"), c(fs + "_knee"), c(fs + "_thai"))
        if best is None or cs > best[0]: best = (cs, hs, fs)
    cs, hs, fs = best
    if cs < min_conf or min(c("back_base"), c("back_end")) < min_conf: return None
    ground = max(g(hs + "_paw")[1], g(fs + "_paw")[1])
    if ground - g(hs + "_thai")[1] <= 0 or ground - g(hs + "_knee")[1] <= 0: return None
    cm_px = 0.5 * (a.stifle / (ground - g(hs + "_thai")[1]) + a.hock / (ground - g(hs + "_knee")[1]))
    L = lambda p, q: float(np.linalg.norm(g(p) - g(q))) * cm_px / 100.0
    return {
        "tibia (stifle-hock)": L(hs + "_thai", hs + "_knee"), "metatarsus+foot (hock-paw)": L(hs + "_knee", hs + "_paw"),
        "forearm (elbow-carpus)": L(fs + "_thai", fs + "_knee"), "fore paw (carpus-paw)": L(fs + "_knee", fs + "_paw"),
        "trunk (back base-back end)": L("back_base", "back_end"), "neck+head (neck base-nose)": L("neck_base", "nose"),
    }
rows = []
for f in a.kp:
    z = np.load(f); nm = list(z["names"])
    for kf in z["kp"]:
        r = photo_lengths(kf, nm)
        if r: rows.append(r)
photo = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
spread = {k: float(np.percentile([r[k] for r in rows], 75) - np.percentile([r[k] for r in rows], 25)) for k in rows[0]}
cm_px = float("nan"); hs = fs = "-"
def model_lengths(model_cfg):
    cfg = load_yaml("configs/gait/trot.yaml")
    if model_cfg == "leo": cfg = apply_leo(cfg)
    else: cfg["model"] = None
    s = Sim(cfg); m, d = s.m, s.d; mujoco.mj_forward(m, d)
    P = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
    tip = {nm: d.xpos[b] + d.xmat[b].reshape(3, 3) @ lv for (b, lv), nm in zip(s.dig, ("finger_L", "finger_R", "toe_L", "toe_R"))}
    n = lambda p, q: float(np.linalg.norm(p - q))
    return {
        "tibia (stifle-hock)": n(P("lower_leg_L"), P("foot_L")), "metatarsus+foot (hock-paw)": n(P("foot_L"), tip["toe_L"]),
        "forearm (elbow-carpus)": n(P("lower_arm_L"), P("hand_anchor_L")), "fore paw (carpus-paw)": n(P("hand_anchor_L"), tip["finger_L"]),
        "trunk (back base-back end)": n(P("C_7"), P("pelvis")), "neck+head (neck base-nose)": n(P("C_7"), P("skull")) + 0.55 * n(P("skull"), P("jaw")) * 2
    }
orig, leo = model_lengths("orig"), model_lengths("leo")
cfgm = load_yaml("configs/model/leo.yaml"); man = {"tibia (stifle-hock)": cfgm["hind"], "metatarsus+foot (hock-paw)": cfgm["hindfoot"], "forearm (elbow-carpus)": cfgm["fore"],
    "fore paw (carpus-paw)": cfgm["digit"], "trunk (back base-back end)": cfgm["trunk_x"], "neck+head (neck base-nose)": cfgm["neck"], }
print(f"{len(rows)} usable photo frames (median over frames; IQR shows the spread)")
print(f"{'segment':30s} {'photo m':>8s} {'orig m':>8s} {'auto scale':>10s} {'manual':>7s} {'leo model m':>11s} {'rel.err':>8s}")
errs = []
for k in photo:
    auto = photo[k] / orig[k]; e = auto / man[k] - 1; errs.append(abs(e))
    print(f"{k:30s} {photo[k]:8.3f} {orig[k]:8.3f} {auto:10.2f} {man[k]:7.2f} {leo[k]:11.3f} {100*e:7.0f}%   (IQR {100*spread[k]/photo[k]:.0f}%)")
print("mean |relative difference| auto vs hand-tuned: %.0f%%" % (100 * np.mean(errs)))
