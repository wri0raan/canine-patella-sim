import sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
from analysis.video_style_features import video_landmarks
from analysis.identify_angles import video_L, angle_series, lp

SEG = {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}
J = ["hock", "stif", "carp", "elb"]


def cycles(tag, t0, t1):
    z = np.load(f"data/real/leo3/work/{tag}_kp.npz"); kp, names = z["kp"], list(z["names"]); a, b = int(t0 * 120), int(t1 * 120) - 1
    V = video_landmarks(kp, names, a, b); A = angle_series(video_L(kp, names, a, b), 120)
    ctr = 0.5 * (V["back_a"] + V["back_b"]); u = V["hpaw"][:, 0] - ctr[:, 0]
    uf = savgol_filter(u, 13, 2); pk, _ = find_peaks(uf, distance=int(0.14 * 120), prominence=0.12 * np.ptp(uf))
    rows = [dict(clip=tag, cycle=i, dur=(pk[i + 1] - pk[i]) / 120, **{f"{j}_{s}": f(A[j][pk[i]:pk[i + 1] + 1]) for j in J for s, f in (("mean", np.mean), ("rom", np.ptp))}) for i in range(len(pk) - 1)]
    whole = dict(clip=tag, **{f"{j}_{s}": f(A[j]) for j in J for s, f in (("mean", np.mean), ("rom", lambda v: np.percentile(v, 97) - np.percentile(v, 3)))})
    return rows, whole


C, Wh = [], []
for tag, (t0, t1) in SEG.items():
    r, w = cycles(tag, t0, t1); C += r; Wh.append(w)
C = pd.DataFrame(C); Wh = pd.DataFrame(Wh); C.to_csv("results/ident_results/N_baseline_cycles.csv", index=False)
pd.set_option("display.width", 250); print(C.round(1).to_string()); print(Wh.round(1).to_string())
feat = [f"{j}_{s}" for j in J for s in ("mean",)] + ["stif_rom", "hock_rom"]
base = Wh[feat]; mu, sd = base.mean(), base.std(ddof=1); n = len(base)
print("\nbaseline over %d clips:" % n); print(pd.DataFrame({"mean": mu, "sd": sd, "cycle_sd": C[feat].std(ddof=1)}).round(1).to_string())
tq = stats.t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n)
print("prediction-interval multiplier for one new clip (95 %%, n=%d): %.2f" % (n, tq))
fp = []
for i in range(n):
    tr = base.drop(base.index[i]); m, s = tr.mean(), tr.std(ddof=1); q = stats.t.ppf(0.975, n - 2) * np.sqrt(1 + 1 / (n - 1)); z = (base.iloc[i] - m) / s
    fp.append({**{k: float(z[k]) for k in feat}, "limit": float(q)})
FP = pd.DataFrame(fp, index=Wh["clip"]); print("\nleave-one-clip-out z (limit %.2f):" % FP.limit.iloc[0]); print(FP.round(2).to_string())
shifts = {"grade 2 (sim: mean -8 deg, range x0.55)": (-8, 0.55), "grade 3 (sim: mean -22 deg, range x0.30)": (-22, 0.30)}
out = []
for nm, (dm, rr) in shifts.items():
    for k, (dv, lab) in {"stif_mean": (dm, "stifle mean"), "stif_rom": (None, "stifle range")}.items():
        newv = mu[k] + dm if dv is not None else mu[k] * rr
        z = (newv - mu[k]) / sd[k]; out.append(dict(shift=nm, feature=lab, baseline_mean=mu[k], baseline_sd=sd[k], new_value=newv, z=z, detected=abs(z) > tq))
P = pd.DataFrame(out); P.to_csv("results/ident_results/N_baseline_power.csv", index=False); print("\n"); print(P.round(2).to_string())
z2 = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True); kp2, nm2 = z2["kp"], list(z2["names"])
w2 = {k: angle_series(video_L(kp2, nm2, int(a * 120), int(b * 120) - 1), 120) for k, (a, b) in {"leo2 A walk 0.73 m/s": (43.7, 46.8), "leo2 B walk 0.55 m/s": (33.8, 36.8)}.items()}
ctx = {k: dict(stif_mean=float(A["stif"].mean()), hock_mean=float(A["hock"].mean())) for k, A in w2.items()}
print(pd.DataFrame(ctx).T.round(0))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a_, (k, lab) in zip(ax, (("stif_mean", "무릎 각도 평균 (도)"), ("hock_mean", "비절 각도 평균 (도)"))):
    for i, t in enumerate(Wh["clip"]): a_.scatter(i, Wh[k][i], c="#1f77b4", s=70, zorder=3); a_.scatter(np.full((C["clip"] == t).sum(), i) + 0.12, C[C["clip"] == t][k], c="#9ecae1", s=22, zorder=2)
    a_.axhspan(mu[k] - tq * sd[k], mu[k] + tq * sd[k], color="#1f77b4", alpha=.12, label="기준선 예측 구간(95%)")
    for j, (nm, v) in enumerate(ctx.items()): a_.scatter(5 + j, v[k], c="#ff7f0e", s=70, marker="s", zorder=3)
    a_.set_xticks(list(range(4)) + [5, 6]); a_.set_xticklabels([f"{t}\n(속보)" for t in Wh["clip"]] + ["leo2 A\n걷기 .73", "leo2 B\n걷기 .55"], fontsize=8)
    a_.set_ylabel(lab); a_.grid(alpha=.3)
    if k == "stif_mean":
        for lv, c_, t_ in ((154, "g", "모델 정상 154"), (146, "orange", "모델 등급2 146"), (132, "r", "모델 등급3 132")): a_.axhline(lv, color=c_, ls=":", lw=1); a_.text(6.4, lv, t_, fontsize=7, va="bottom", ha="right", color=c_)
ax[0].legend(fontsize=7, loc="lower left"); fig.suptitle("leo 기준선: 영상 4개(속보), 점=영상, 연한 점=걸음 주기, 사각=leo 걷기 영상 (같은 개, 속도에 따라 12도 달라짐)", fontsize=9)
fig.tight_layout(); fig.savefig("results/figures/baseline_leo.png", dpi=150)
