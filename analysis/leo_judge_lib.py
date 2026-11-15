import sys, os, glob, json
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import norm
sys.path.insert(0, ".")
from analysis.virtual_markers import sim_full
from analysis.identify_angles import lp

CAUD, DORS, TRACKER_SD, TIB_MARK_PX = 0.38, 0.22, 3.0, 126.8
FPSV = 120.0; _b, _a = butter(3, 6.0 / (FPSV / 2)); lpv = lambda x: filtfilt(_b, _a, x, axis=0)
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
ang = lambda p, q, r: np.degrees(np.arccos(np.clip((unit(p - q) * unit(r - q)).sum(-1), -1, 1)))
CONST = "results/judge_constants.json"


def sim_stifle(run, caud=CAUD, dors=DORS):
    path = f"results/ident/{run}.npz"; L, ax, fps = sim_full(path); z = np.load(path, allow_pickle=True); names = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; ix = {k: i for i, k in enumerate(names)}
    xz = lambda k: P[:, ix[k]][:, [0, 2]]; hip, st, hk = xz("upper_leg_L"), xz("lower_leg_L"), xz("foot_L"); dd = np.stack([-ax[:, 1], ax[:, 0]], 1); dd = dd if dd[:, 1].mean() > 0 else -dd
    T = np.median(np.linalg.norm(st - hk, axis=1)); m = slice(int(2 * fps), None)
    return float(ang(lp(hip + T * (caud * ax + dors * dd), fps)[m], lp(st, fps)[m], lp(hk, fps)[m]).mean())


def rump_offset(clip="a"):
    L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
    z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); ix = {n: i for i, n in enumerate(z["names"])}; v = []
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); tl = L[key].get("tail")
        if tl: be, bb = z["kp"][f, ix["back_end"], :2], z["kp"][f, ix["back_base"], :2]; v.append(float((np.array(tl) - be) @ unit(be - bb)) / TIB_MARK_PX)
    v = np.array(v); return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def video_stifle(tag, leg, c):
    fz = np.load(f"data/real/leo5/work/fused/{tag}_tracks.npz"); ix = {n: i for i, n in enumerate(fz["names"])}; ins = np.nonzero(fz["inside"])[0]
    if len(ins) < 30: return float("nan"), 0.0
    a0, b0 = ins[0], ins[-1] + 1; body = fz["body"][a0:b0]; be, bb = lpv(body[:, ix["back_end"], :2]), lpv(body[:, ix["back_base"], :2]); X = fz[leg][a0:b0]; st, hk = lpv(X[:, 0, :2]), lpv(X[:, 1, :2])
    Tv = np.median(np.linalg.norm(st - hk, axis=1)); return float(ang(be + c * Tv * unit(be - bb), st, hk).mean()), (b0 - a0) / FPSV


def clip_stifle(tag, c):
    v = [video_stifle(tag, leg, c) for leg in ("hindA", "hindB")]; return float(np.nanmean([x[0] for x in v])), v[0][1], [x[0] for x in v]


RUNSETS = {"fit": dict(normal=["fitA", "fitB"], g2=["fitA_G2", "fitB_G2"], g3=["fitB_G3"], pair2=["fitA", "fitB"], pair3=["fitB"]),
           "R": dict(normal=["fitR2"], g2=["fitR_G2"], g3=["fitR_G3"], pair2=["fitR2"], pair3=["fitR2"]),
           "R3": dict(normal=["fitR3"], g2=["fitR3_G2"], g3=["fitR3_G3"], pair2=["fitR3"], pair3=["fitR3"])}


def constants(refresh=False, runset="fit"):
    path = CONST if runset == "fit" else CONST.replace(".json", f"_{runset}.json"); rs = RUNSETS[runset]
    if os.path.exists(path) and not refresh: return json.load(open(path))
    base = {r: sim_stifle(r) for r in rs["normal"]}; mu = float(np.mean(list(base.values())))
    d2 = float(np.mean([sim_stifle(g) - sim_stifle(b) for g, b in zip(rs["g2"], rs["pair2"])])); d3 = float(np.mean([sim_stifle(g) - sim_stifle(b) for g, b in zip(rs["g3"], rs["pair3"])]))
    sd_sim = float(np.std([sim_stifle(rs["normal"][0], CAUD + a, DORS + b) for a in (-0.05, 0, 0.05) for b in (-0.05, 0, 0.05)])); c, se = rump_offset("a")
    sd_c = float(np.std([video_stifle("a", "hindA", c + d)[0] for d in (-se, 0, se)]))
    C = dict(mu_normal=mu, d_abnormal=d2, d_severe=d3, sd_sim=sd_sim, c_rump=c, c_se=se, sd_c=sd_c, sd_tracker=TRACKER_SD, sim_base=base, runset=runset)
    json.dump(C, open(path, "w"), indent=1); return C


def posterior(x, mus, sd):
    lik = {k: norm.pdf(x, m, sd) for k, m in mus.items()}; s = sum(lik.values()) + 1e-300; return {k: v / s for k, v in lik.items()}
