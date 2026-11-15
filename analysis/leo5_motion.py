import sys
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
from analysis.video_style_features import angle2

FPS = 120.0; N = 100; b6, a6 = butter(3, 6.0 / (FPS / 2))
lp = lambda x: filtfilt(b6, a6, x, axis=0)
rows, cyc, curves = [], [], {}
for clip in ("a", "b"):
    z = np.load(f"data/real/leo5/work/fused/{clip}_tracks.npz"); names = list(z["names"]); ix = {n: i for i, n in enumerate(names)}; ins = np.nonzero(z["inside"])[0]; a0, b0 = ins[0], ins[-1] + 1
    body = z["body"][a0:b0]; P = lambda nm: body[:, ix[nm], :2]
    rump, wit = lp(P("back_end")), lp(P("back_base")); ctr = 0.5 * (rump + wit); dog = float(np.median(np.linalg.norm(wit - rump, axis=1))); mpp = 0.45 / dog
    face = np.sign(np.median(P("nose")[:, 0] - P("tail_base")[:, 0])) or 1.0; t = np.arange(len(ctr)) / FPS
    speed = abs(np.polyfit(t, ctr[:, 0], 1)[0]) * mpp
    for leg in ("hindA", "hindB"):
        X = z[leg][a0:b0]; st, hk, pw = (lp(X[:, i, :2]) for i in range(3))
        u = (pw[:, 0] - ctr[:, 0]) * face; uf = savgol_filter(u, 13, 2); pk, _ = find_peaks(uf, distance=int(0.14 * FPS), prominence=0.12 * np.ptp(uf))
        ang = {"stif": angle2(rump, st, hk), "hock": angle2(st, hk, pw)}; r = dict(clip=clip, leg=leg, seconds=len(t) / FPS, speed=speed, strides=len(pk) - 1, stride_hz=(len(pk) - 1) / ((pk[-1] - pk[0]) / FPS) if len(pk) > 2 else np.nan)
        for j, A in ang.items():
            r[f"{j}_mean"] = float(A.mean()); r[f"{j}_rom"] = float(np.percentile(A, 97) - np.percentile(A, 3))
            C = np.array([np.interp(np.linspace(0, pk[i + 1] - pk[i], N), np.arange(pk[i + 1] - pk[i] + 1), A[pk[i]:pk[i + 1] + 1]) for i in range(len(pk) - 1)])
            curves[(clip, leg, j)] = C; r[f"{j}_cycle_mean_sd"] = float(C.mean(1).std(ddof=1)) if len(C) > 1 else np.nan
            cc = [np.corrcoef(C[i] - C[i].mean(), C[k] - C[k].mean())[0, 1] for i in range(len(C)) for k in range(i + 1, len(C))]; r[f"{j}_cycle_corr"] = float(np.mean(cc)) if cc else np.nan
        rows.append(r)
R = pd.DataFrame(rows)
for j in ("stif", "hock"):
    for clip in ("a", "b"):
        ca, cb = curves[(clip, "hindA", j)].mean(0), curves[(clip, "hindB", j)].mean(0); R.loc[R["clip"] == clip, f"{j}_AB_corr"] = float(np.corrcoef(ca - ca.mean(), cb - cb.mean())[0, 1])
R.to_csv("results/ident_results/T_leo5_motion.csv", index=False); pd.set_option("display.width", 250); print(R.round(2).to_string())
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True); x = np.linspace(0, 100, N)
for r_, j in enumerate(("stif", "hock")):
    for c_, (clip, leg) in enumerate((("a", "hindA"), ("a", "hindB"), ("b", "hindA"), ("b", "hindB"))):
        C = curves[(clip, leg, j)]; a_ = ax[r_, c_]
        for row in C: a_.plot(x, row, color="0.75", lw=0.8)
        a_.plot(x, C.mean(0), "k", lw=2.2); a_.set_title(f"{'무릎' if j == 'stif' else '비절'}, 클립 {clip.upper()} 다리 {leg[-1]} ({len(C)}주기)", fontsize=10)
        if c_ == 0: a_.set_ylabel("각도 (도)")
        if r_ == 1: a_.set_xlabel("걸음 주기 (%)")
plt.tight_layout(); plt.savefig("results/figures/leo5_curves.png", dpi=140)
