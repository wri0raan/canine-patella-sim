import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
R = pd.read_csv("results/ident_results/O_vmarker.csv"); P = R.pivot(index="lam", columns="run", values="stif_mean"); x = P.index.values * 12.4
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for k, c, l in (("fitB", "#2ca02c", "정상"), ("fitB_G2", "#ff7f0e", "등급 2"), ("fitB_G3", "#d62728", "등급 3")): ax[0].plot(x, P[k], c=c, lw=2, label=f"모델 {l} (사슬 B)")
ax[0].axhspan(130.4, 142.4, color="#1f77b4", alpha=.15, label="leo 걷기 영상 (130-142도)"); ax[0].axvspan(0, 0.7 * 12.4, color="gray", alpha=.08)
ax[0].set_xlabel("엉덩이 표식을 꼬리 쪽으로 옮긴 거리 (cm)"); ax[0].set_ylabel("무릎 각도 평균 (도)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("절대 수준은 표식 위치에 따라 크게 움직인다", fontsize=10)
ax[1].plot(x, P.fitB_G2 - P.fitB, c="#ff7f0e", lw=2, label="등급 2 (B)"); ax[1].plot(x, P.fitA_G2 - P.fitA, c="#ff7f0e", lw=2, ls="--", label="등급 2 (A)"); ax[1].plot(x, P.fitB_G3 - P.fitB, c="#d62728", lw=2, label="등급 3 (B)")
ax[1].set_xlabel("엉덩이 표식을 꼬리 쪽으로 옮긴 거리 (cm)"); ax[1].set_ylabel("정상 대비 변화 (도)"); ax[1].set_ylim(-26, 0); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("등급에 따른 변화량은 거의 그대로", fontsize=10)
fig.tight_layout(); fig.savefig("results/figures/marker_sensitivity.png", dpi=150)
