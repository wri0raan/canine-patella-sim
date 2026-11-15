import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
S = pd.read_csv("results/ident_results/summary.csv"); D = pd.read_csv("results/ident_results/D_speedmatched.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
for fs, c in (("ALL", "#1f77b4"), ("ROBUST", "#d62728")):
    for tgt, ls in (("muscle", "-"), ("mass", "--")):
        d = S[(S.study == "A") & (S.feats == fs) & (S.target == tgt)]; ax[0].plot(d.noise, 1 - d.mae / d.baseline_mae, ls, color=c, marker="o", label=f"{ {'muscle': '근력', 'mass': '체중'}[tgt] } ({'모든 지표' if fs == 'ALL' else '속도, 걸음 주기, 보폭'})")
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("몸 조건(근력 실선, 체중 점선)"); ax[0].set_ylabel("예측 향상률 = 1 - 오차/평균예측오차 (0 이하면 쓸모 없음)"); ax[0].set_xlabel("측정 잡음 배수"); ax[0].legend(fontsize=7)
for fs, c in (("ALL", "#1f77b4"), ("ROBUST", "#d62728")):
    d = S[(S.study == "B") & (S.feats == fs)]; ax[1].plot(d.noise, d.acc, color=c, marker="o", label="모든 지표" if fs == "ALL" else "속도, 걸음 주기, 보폭")
ax[1].axhline(0.269, color="k", ls=":", label="무작위 기준"); ax[1].set_ylim(0, 0.8); ax[1].set_title("탈구 등급(0, 2, 3, 4) 정확도"); ax[1].set_xlabel("측정 잡음 배수"); ax[1].legend(fontsize=7)
for sp, c in (("S73", "#2ca02c"), ("S55", "#ff7f0e")):
    for tgt, ls in (("muscle", "-"), ("mass", "--")):
        d = D[(D.speed == sp) & (D.feats == "ALL") & (D.target == tgt)]; ax[2].plot(d.noise, 1 - d.mae / d.baseline_mae, ls, color=c, marker="o", label=f"{ {'S73': '0.73 m/s', 'S55': '0.55 m/s'}[sp] } {'근력' if tgt == 'muscle' else '체중'}")
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_title("속도를 맞춘 격자 (목표 속도별)"); ax[2].set_xlabel("측정 잡음 배수"); ax[2].legend(fontsize=7)
for a in ax: a.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("results/figures/ident_final.png", dpi=150); print("ok")
