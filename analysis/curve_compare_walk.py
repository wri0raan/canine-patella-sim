import sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, find_peaks
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
from analysis.video_style_features import video_landmarks
from analysis.identify_angles import sim_L, video_L, angle_series, lp

N = 100


def curves(A, hp, fps, a=0, b=None):
    b = len(hp) if b is None else b
    uf = savgol_filter(hp, max(5, int(0.1 * fps) | 1), 2); pk, _ = find_peaks(uf, distance=int(0.14 * fps), prominence=0.12 * np.ptp(uf))
    pk = pk[(pk >= a) & (pk < b)]; out = {"stif": [], "hock": []}
    for i in range(len(pk) - 1):
        for j in out:
            seg = A[j][pk[i]:pk[i + 1] + 1]; out[j].append(np.interp(np.linspace(0, len(seg) - 1, N), np.arange(len(seg)), seg))
    return {j: np.array(v) for j, v in out.items()}, len(pk) - 1


def video_pass(kp, names, t0, t1):
    a, b = int(t0 * 120), int(t1 * 120) - 1; L = video_L(kp, names, a, b); A = angle_series(L, 120)
    ctr = 0.5 * (L["withers"] + L["rump"]); return curves(A, L["hpaw"][:, 0] - ctr[:, 0], 120)


def sim_pass(path, skip=2.0):
    L, fps = sim_L(path); A = angle_series(L, fps); ctr = 0.5 * (L["withers"] + L["rump"])
    return curves(A, L["hpaw"][:, 0] - ctr[:, 0], fps, int(skip * fps))


k2 = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True); kp2, nm2 = k2["kp"], list(k2["names"])
VID = {"leo2@34s": video_pass(kp2, nm2, 33.78, 36.82), "leo2@44s": video_pass(kp2, nm2, 43.73, 46.70)}
for t, (t0, t1) in {"n2": (2.8, 5.8), "n4": (1.6, 5.2)}.items():
    z = np.load(f"data/real/leo4/work/{t}_kp.npz"); VID[f"leo4 {t}"] = video_pass(z["kp"], list(z["names"]), t0, t1)
SIM = {m: sim_pass(f"results/ident/{m}.npz") for m in ("fitA", "fitB")}
sess = lambda k: k.split()[0].split("@")[0]
rows = []
for m, (S, ns) in SIM.items():
    for p, (V, nv) in VID.items():
        for j in ("stif", "hock"):
            s, v = S[j].mean(0), V[j].mean(0)
            rows.append(dict(model=m, video=p, joint=j, n_cyc_sim=ns, n_cyc_vid=nv, corr=float(np.corrcoef(s - s.mean(), v - v.mean())[0, 1]), rom_ratio=float(np.ptp(s) / np.ptp(v)),
                             offset=float(s.mean() - v.mean()), vid_cycle_sd=float(V[j].mean(1).std(ddof=1)) if nv > 1 else np.nan))
R = pd.DataFrame(rows)
res = []
for m in SIM:
    for j in ("stif", "hock"):
        for held in ("leo2", "leo4"):
            fit = R[(R.model == m) & (R.joint == j) & (R.video.map(sess) != held)]; tst = R[(R.model == m) & (R.joint == j) & (R.video.map(sess) == held)]
            off = fit.offset.mean()
            for _, r in tst.iterrows(): res.append(dict(model=m, joint=j, held_out=r.video, offset_from_other_session=off, residual=r.offset - off, corr=r["corr"], rom_ratio=r.rom_ratio))
RS = pd.DataFrame(res); R.to_csv("results/ident_results/R_curve_walk.csv", index=False); RS.to_csv("results/ident_results/R_curve_walk_split.csv", index=False)
pd.set_option("display.width", 220); print(R.round(2).to_string()); print(); print(RS.round(2).to_string())
fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True); x = np.linspace(0, 100, N)
for i, j in enumerate(("stif", "hock")):
    for k, (m, (S, _)) in enumerate(SIM.items()):
        a_ = ax[i, k]
        for p, (V, _) in VID.items(): a_.plot(x, V[j].mean(0), lw=1.2, label=p)
        a_.plot(x, S[j].mean(0), "k", lw=2.4, label=f"모델 {m}"); a_.set_title(("무릎 " if j == "stif" else "비절 ") + f"각도, 모델 {m} vs 영상"); a_.set_ylabel("각도 (도)")
ax[1, 0].set_xlabel("걸음 주기 (%)"); ax[1, 1].set_xlabel("걸음 주기 (%)"); ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.savefig("results/figures/curve_walk.png", dpi=150)
