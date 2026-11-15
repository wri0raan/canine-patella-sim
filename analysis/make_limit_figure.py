import yaml
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
G = yaml.safe_load(open("configs/impairment/patella.yaml", encoding="utf-8"))["grades"]
gr = [0, 1, 2, 3, 4]; mech = [G[g]["prom_ext"] for g in gr]; func = [min(G[g]["prom_ext"], G[g]["arom_ext"]) for g in gr]; meas = [G[g].get("measured", False) for g in gr]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = ax[0]
for y, col, lab in ((mech, "#4c78a8", "기계적 한계 (수동 최대 신전)"), (func, "#e45756", "기능적 한계 (걸을 때 쓰는 최대 신전)")):
    a.plot(gr, y, "-", color=col, lw=2, label=lab)
    for g, v, m in zip(gr, y, meas): a.plot(g, v, "o", ms=9, mfc=col if m else "white", mec=col, mew=2); a.text(g, v + 2.2, f"{v:.0f}", ha="center", fontsize=9, color=col)
a.set_xticks(gr); a.set_xticklabels(["정상", "1기", "2기", "3기", "4기"]); a.set_ylim(105, 175); a.set_ylabel("무릎 최대 신전 한계 (도, 180 = 완전히 폄)"); a.set_title("(a) 등급별 무릎 신전 한계", fontsize=11); a.legend(fontsize=8.5, loc="lower left")
a.text(0.02, 0.97, "● 임상 논문의 측정값(정상, 3기)   ○ 보간한 추정값", transform=a.transAxes, fontsize=8.5, va="top")
b = ax[1]; b.set_xlim(40, 185); b.set_ylim(-0.2, 6.2); b.set_yticks([]); b.spines[["top", "right", "left"]].set_visible(False); b.set_title("(b) 3기 개의 무릎 각도 눈금과 두 가지 한계", fontsize=11)
b.set_xticks(range(40, 181, 20)); b.set_xlabel("무릎 각도 (도)   굽힘 ← → 폄 (180 = 완전히 폄)")
g3 = G[3]; yy = 4.5
b.barh(yy, g3["prom_ext"] - g3["prom_flex"], left=g3["prom_flex"], height=0.6, color="#d9d9d9"); b.text(42, yy + 0.5, f"관절이 구조적으로 허용하는 범위(수동 가동범위): {g3['prom_flex']:.0f}~{g3['prom_ext']:.0f}도", ha="left", fontsize=9.5)
b.barh(yy - 1.5, g3["arom_ext"] - g3["knee_mfa"], left=g3["knee_mfa"], height=0.6, color="#f4a261"); b.text(42, yy - 1.5 + 0.5, f"걸을 때 실제로 쓰는 범위(능동 가동범위): {g3['knee_mfa']:.0f}~{g3['arom_ext']:.0f}도", ha="left", fontsize=9.5)
for x, col, lab, yv in ((g3["prom_ext"], "#4c78a8", f"기계적 한계 {g3['prom_ext']:.0f}도", 2.3), (g3["arom_ext"], "#e45756", f"기능적 한계 {g3['arom_ext']:.0f}도", 1.0)):
    b.plot([x, x], [0.4, 5.6], "--", color=col, lw=2); b.text(x + 1.5, yv, lab, color=col, fontsize=10, fontweight="bold")
b.plot([167.1, 167.1], [0.4, 5.6], ":", color="#555", lw=1.5); b.text(166, 5.8, "정상 개가 걸을 때 최대 신전 약 167도", ha="right", fontsize=8.5, color="#555")
b.annotate("", xy=(g3["arom_ext"], 0.45), xytext=(g3["prom_ext"], 0.45), arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5)); b.text(0.5 * (g3["prom_ext"] + g3["arom_ext"]), 0.0, "환견이 쓰지 않는 구간 약 25도", ha="center", fontsize=8.5, color="#555")
plt.tight_layout(); plt.savefig("results/figures/thesis/F22_limits.png", dpi=150); plt.close(); print("F22")
