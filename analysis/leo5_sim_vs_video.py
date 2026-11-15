import sys, glob, json
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
sys.path.insert(0, ".")
from analysis.virtual_markers import sim_full

CAUD, DORS = 0.38, 0.22
FPSV = 120.0; b6, a6 = butter(3, 6.0 / (FPSV / 2)); lpv = lambda x: filtfilt(b6, a6, x, axis=0)
unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
def ang(p, q, r):
    u, v = unit(p - q), unit(r - q); return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))
stat = lambda A: (float(A.mean()), float(np.percentile(A, 97) - np.percentile(A, 3)))

rows = []
for run in ("fitA", "fitB"):
    path = f"results/ident/{run}.npz"; L, ax, fps = sim_full(path); z = np.load(path, allow_pickle=True); names = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; ix = {k: i for i, k in enumerate(names)}
    xz = lambda k: P[:, ix[k]][:, [0, 2]]; toe = P[:, P.shape[1] - 4 + 2][:, [0, 2]]
    hip, st, hk, mtp, ca1 = xz("upper_leg_L"), xz("lower_leg_L"), xz("foot_L"), xz("foot_anchor_L"), xz("Ca_1")
    dors = np.stack([-ax[:, 1], ax[:, 0]], 1); dors = dors if dors[:, 1].mean() > 0 else -dors
    T = np.median(np.linalg.norm(st - hk, axis=1)); v = ca1 - hip
    print(f"{run}: sim Ca_1 relative to the hip: caudal {np.median((v * ax).sum(1)) / T:+.2f} T, dorsal {np.median((v * dors).sum(1)) / T:+.2f} T  (X-ray tail root: +{CAUD} / +{DORS}); T_sim {T:.3f} m")
    rump_v = hip + T * (CAUD * ax + DORS * dors); m = slice(int(2 * fps), None)
    lp_ = lambda x: x
    from analysis.identify_angles import lp
    F = {k: lp(x, fps)[m] for k, x in dict(rv=rump_v, ca=ca1, st=st, hk=hk, mtp=mtp, toe=toe).items()}
    for rk, rn in (("rv", "xray_rump"), ("ca", "Ca_1")):
        sm, sr = stat(ang(F[rk], F["st"], F["hk"]))
        for pk, pn in (("mtp", "mtp_end"), ("toe", "toe_tip")):
            hm, hr = stat(ang(F["st"], F["hk"], F[pk])); rows.append(dict(source=f"sim {run}", rump=rn, paw=pn, stif_mean=sm, stif_rom=sr, hock_mean=hm, hock_rom=hr))
L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
off = {}
for clip in ("a", "b"):
    cg = np.load(f"data/real/leo5/work/c{clip}_crop.npz"); scale = float(cg["size"]) / cg["side"]; z = np.load(f"data/real/leo5/work/hrnet_w32/{clip}_kp.npz"); ix = {n: i for i, n in enumerate(z["names"])}; vals = []
    for key in sorted(k for k in L if k.startswith(clip + "_")):
        f = int(key.split("_")[1]); tl = L[key].get("tail")
        if not tl: continue
        be, bb = z["kp"][f, ix["back_end"], :2], z["kp"][f, ix["back_base"], :2]; u = unit(be - bb); vals.append(float((np.array(tl) - be) @ u) / 126.8)
    off[clip] = vals; print(f"video: tail-root mark along the back axis beyond back_end, clip {clip}: {np.mean(vals):+.2f} T (sd {np.std(vals):.2f}, n {len(vals)})")
c_fit = float(np.mean(off["a"]))
for clip in ("a", "b"):
    fz = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz"); names = list(fz["names"]); ix = {n: i for i, n in enumerate(names)}; ins = np.nonzero(fz["inside"])[0]; a0, b0 = ins[0], ins[-1] + 1
    body = fz["body"][a0:b0]; be, bb = lpv(body[:, ix["back_end"], :2]), lpv(body[:, ix["back_base"], :2]); ub = unit(be - bb)
    for leg in ("hindA", "hindB"):
        X = fz[leg][a0:b0]; st, hk, pw = (lpv(X[:, i, :2]) for i in range(3)); Tv = np.median(np.linalg.norm(st - hk, axis=1))
        for rn, rump in (("tail_est(c_fit)", be + c_fit * Tv * ub), ("back_end", be)):
            sm, sr = stat(ang(rump, st, hk)); hm, hr = stat(ang(st, hk, pw)); rows.append(dict(source=f"video {clip} {leg}", rump=rn, paw="tracked", stif_mean=sm, stif_rom=sr, hock_mean=hm, hock_rom=hr))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/W_sim_vs_video.csv", index=False); pd.set_option("display.width", 200); print(); print(R.round(1).to_string(index=False))
