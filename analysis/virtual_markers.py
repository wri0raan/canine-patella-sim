import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.identify_angles import sim_L, video_L, lp, angle_series
from analysis.video_style_features import angle2

SEG3 = {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}


def sim_full(path):
    z = np.load(path, allow_pickle=True); n = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; R = z["pelvis_R"]; fps = float(z["fps"]); ix = {k: i for i, k in enumerate(n)}
    xz = lambda k: P[:, ix[k]][:, [0, 2]]
    L = sim_L(path)[0]
    t = xz("pelvis") - xz("C_7"); t = lp(t, fps); t /= np.linalg.norm(t, axis=1, keepdims=True)
    loc = np.einsum("tji,tj->ti", R, np.concatenate([t, np.zeros((len(t), 1))], 1)); loc = loc.mean(0); loc /= np.linalg.norm(loc)
    ax = np.einsum("tij,j->ti", R, loc)[:, [0, 2]]; ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    return L, ax, fps


def with_rump(L, ax, lam, scale):
    L2 = dict(L); L2["rump"] = L["rump"] + lam * scale * ax; return L2


def stifle_stats(L, fps):
    A = angle_series(L, fps); d = np.linalg.norm(L["rump"] - L["stifle"], axis=1); s = np.linalg.norm(L["stifle"] - L["hock"], axis=1)
    return A, float(np.median(d / s))


def video_ratio(kp, names, a, b):
    L = video_L(kp, names, a, b); L = {k: lp(v, 120) for k, v in L.items()}
    d = np.linalg.norm(L["rump"] - L["stifle"], axis=1); s = np.linalg.norm(L["stifle"] - L["hock"], axis=1); return float(np.median(d / s))


if __name__ == "__main__":
    from analysis.video_style_features import angle2 as _a
    vr = {}
    for f, fpp, tag in (("data/real/leo2/work/photos_kp.npz", 2, "photos"), ("data/real/leo2/work/photo9_kp.npz", 1, "photo9")):
        z = np.load(f, allow_pickle=True); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}; vals = []
        for k in range(0, kp.shape[0], fpp):
            S = kp[k]; g = lambda n: S[ix[n], :2]; c = lambda n: S[ix[n], 2]
            for hs in ("back_left", "back_right"):
                if min(c(n) for n in (f"{hs}_thai", f"{hs}_knee", "back_end")) < 0.5: continue
                vals.append(np.linalg.norm(g("back_end") - g(f"{hs}_thai")) / np.linalg.norm(g(f"{hs}_thai") - g(f"{hs}_knee")))
        vr[tag] = (float(np.median(vals)), float(np.std(vals)), len(vals))
    z2 = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True)
    vr["leo2 A walk"] = (video_ratio(z2["kp"], list(z2["names"]), int(43.7 * 120), int(46.8 * 120) - 1), 0, 1); vr["leo2 B walk"] = (video_ratio(z2["kp"], list(z2["names"]), int(33.8 * 120), int(36.8 * 120) - 1), 0, 1)
    for tag, (a, b) in SEG3.items():
        z = np.load(f"data/real/leo3/work/{tag}_kp.npz"); vr[f"leo3 {tag} trot"] = (video_ratio(z["kp"], list(z["names"]), int(a * 120), int(b * 120) - 1), 0, 1)
    print("video rump-stifle / stifle-hock length ratio:"); [print("  %-14s %.2f (sd %.2f, n %d)" % (k, *v)) for k, v in vr.items()]
    target = vr["photos"][0]
    rows = []
    for r in ("fitA", "fitB", "fitA_G2", "fitB_G2", "fitB_G3"):
        L, ax, fps = sim_full(f"results/ident/{r}.npz"); scale = float(np.median(np.linalg.norm(L["stifle"] - L["hock"], axis=1)))
        for lam in np.arange(0.0, 1.61, 0.1):
            A, ratio = stifle_stats(with_rump(L, ax, lam, scale), fps); n = len(A["stif"]); m = slice(int(2 * fps), n)
            rows.append(dict(run=r, lam=round(lam, 2), ratio=ratio, stif_mean=float(A["stif"][m].mean()), stif_rom=float(np.percentile(A["stif"][m], 97) - np.percentile(A["stif"][m], 3)), hock_mean=float(A["hock"][m].mean())))
    R = pd.DataFrame(rows); R.to_csv("results/ident_results/O_vmarker.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nsim: ratio and stifle mean vs lam (fitA / fitB normal)"); print(R[R.run.isin(["fitA", "fitB"])].pivot(index="lam", columns="run", values=["ratio", "stif_mean", "stif_rom"]).round(2).to_string())
    lam_star = {}
    for r in ("fitA", "fitB"):
        d = R[R.run == r]; lam_star[r] = float(np.interp(target, d.ratio.values, d.lam.values))
    print("\nlam from the standing-photo ratio %.2f: " % target, {k: round(v, 2) for k, v in lam_star.items()})
    lam = float(np.mean(list(lam_star.values()))); print("lam used:", round(lam, 2))
    out = []
    for r in ("fitA", "fitB", "fitA_G2", "fitB_G2", "fitB_G3"):
        d = R[R.run == r]; out.append(dict(run=r, stif_mean_Ca1=float(d[d.lam == 0].stif_mean.iloc[0]), stif_mean_vm=float(np.interp(lam, d.lam, d.stif_mean)), ratio_vm=float(np.interp(lam, d.lam, d.ratio))))
    print(pd.DataFrame(out).round(1).to_string())
    pd.DataFrame(out).to_csv("results/ident_results/O_vmarker_used.csv", index=False)
