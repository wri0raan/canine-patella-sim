import sys
import numpy as np, pandas as pd
from scipy.signal import find_peaks, savgol_filter
sys.path.insert(0, ".")
from analysis.virtual_markers import sim_full
from analysis.identify_angles import lp
from analysis.leo_judge_lib import ang, unit, lpv, CAUD, DORS, constants

stat = lambda A: (float(A.mean()), float(np.percentile(A, 97) - np.percentile(A, 3)))
rows = []


def stride_hz(hp, ctr_x, fps):
    uf = savgol_filter(hp - ctr_x, max(5, int(0.1 * fps) | 1), 2); pk, _ = find_peaks(uf, distance=int(0.14 * fps), prominence=0.12 * np.ptp(uf))
    return (len(pk) - 1) / ((pk[-1] - pk[0]) / fps) if len(pk) > 2 else float("nan")


for run in sys.argv[1:]:
    path = f"results/ident/{run}.npz"
    try: L, ax, fps = sim_full(path)
    except Exception as e: print("missing", run, e); continue
    z = np.load(path, allow_pickle=True); names = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; ix = {k: i for i, k in enumerate(names)}; xz = lambda k: P[:, ix[k]][:, [0, 2]]; toe = P[:, P.shape[1] - 4 + 2][:, [0, 2]]
    hip, st, hk, mtp, ca1 = xz("upper_leg_L"), xz("lower_leg_L"), xz("foot_L"), xz("foot_anchor_L"), xz("Ca_1"); dd = np.stack([-ax[:, 1], ax[:, 0]], 1); dd = dd if dd[:, 1].mean() > 0 else -dd
    T = np.median(np.linalg.norm(st - hk, axis=1)); m = slice(int(2 * fps), None); rv = hip + T * (CAUD * ax + DORS * dd)
    F = {k: lp(x, fps)[m] for k, x in dict(rv=rv, st=st, hk=hk, mtp=mtp, toe=toe).items()}; n = len(F["st"]); speed = (L["rump"][-1, 0] - L["rump"][int(2 * fps), 0]) / (n / fps)
    sm, sr = stat(ang(F["rv"], F["st"], F["hk"])); hm, hr = stat(ang(F["st"], F["hk"], F["mtp"])); tm, tr = stat(ang(F["st"], F["hk"], F["toe"]))
    rows.append(dict(source=f"sim {run}", speed=speed, stride_hz=stride_hz(lp(mtp, fps)[m][:, 0], lp(0.5 * (L["withers"] + L["rump"]), fps)[m][:, 0], fps), stif_mean=sm, stif_rom=sr, hock_mean_mtp=hm, hock_rom_mtp=hr, hock_mean_toe=tm, hock_rom_toe=tr))
c = constants()["c_rump"]
for tag, role in (("a", "fit target (speed/stride)"), ("b", "fit target (speed/stride)"), ("t7725", "HELD OUT"), ("l2a", "HELD OUT"), ("l2b", "HELD OUT")):
    fz = np.load(f"data/real/leo5/work/fused/{tag}_tracks.npz"); ix = {n: i for i, n in enumerate(fz["names"])}; ins = np.nonzero(fz["inside"])[0]; a0, b0 = ins[0], ins[-1] + 1; body = fz["body"][a0:b0]
    be, bb, wit = lpv(body[:, ix["back_end"], :2]), lpv(body[:, ix["back_base"], :2]), lpv(body[:, ix["back_base"], :2]); ctr = 0.5 * (be + wit); dog = float(np.median(np.linalg.norm(be - bb, axis=1)))
    face = np.sign(np.median(body[:, ix["nose"], 0] - body[:, ix["tail_base"], 0])) or 1.0; t = np.arange(len(ctr)) / 120.0; spd = abs(np.polyfit(t, ctr[:, 0], 1)[0]) * 0.45 / dog
    r = []
    for leg in ("hindA", "hindB"):
        X = fz[leg][a0:b0]; s_, h_, p_ = (lpv(X[:, i, :2]) for i in range(3)); Tv = np.median(np.linalg.norm(s_ - h_, axis=1)); rump = be + c * Tv * unit(be - bb)
        sm, sr = stat(ang(rump, s_, h_)); hm, hr = stat(ang(s_, h_, p_)); r.append((sm, sr, hm, hr, stride_hz((p_[:, 0]) * face, ctr[:, 0] * face, 120.0)))
    r = np.array(r).mean(0); rows.append(dict(source=f"video {tag} [{role}]", speed=spd, stride_hz=r[4], stif_mean=r[0], stif_rom=r[1], hock_mean_mtp=r[2], hock_rom_mtp=r[3]))
A = pd.read_csv("results/ident_results/V_convert_angles.csv"); A = A[A.model == "fused"]
for cl in ("a", "b"):
    g = A[A["clip"] == cl]; rows.append(dict(source=f"marked truth {cl} (joint centres, incl. marking noise)", stif_mean=g.truth_stif_conv.mean(), stif_rom=np.percentile(g.truth_stif_conv, 97) - np.percentile(g.truth_stif_conv, 3), hock_mean_mtp=g.truth_hock_conv.mean(), hock_rom_mtp=np.percentile(g.truth_hock_conv, 97) - np.percentile(g.truth_hock_conv, 3)))
R = pd.DataFrame(rows); R.to_csv("results/ident_results/X_refit_eval.csv", index=False); pd.set_option("display.width", 220); print(R.round(2).to_string(index=False))
