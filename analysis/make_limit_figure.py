import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
BLUE, RED, ORANGE, GRAY = "#2f5f98", "#c8403f", "#e8923a", "#cfcfcf"
G = yaml.safe_load(open("configs/impairment/patella.yaml", encoding="utf-8"))["grades"]
gr = [0, 1, 2, 3, 4]; mech = [G[g]["prom_ext"] for g in gr]; func = [min(G[g]["prom_ext"], G[g]["arom_ext"]) for g in gr]; meas = [G[g].get("measured", False) for g in gr]
fig, (a, b) = plt.subplots(2, 1, figsize=(5.6, 6.3), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.55))
a.fill_between(gr, func, mech, color=RED, alpha=0.12, lw=0)
for y, col in ((mech, BLUE), (func, RED)):
    a.plot(gr, y, "-", color=col, lw=2)
    for g, v, m in zip(gr, y, meas):
        a.plot(g, v, "o", ms=8, mfc=col if m else "white", mec=col, mew=1.8, zorder=3)
for g, v in zip(gr, mech):
    if g >= 2: a.text(g, v + 3.2, f"{v:.0f}", ha="center", fontsize=9, color=BLUE)
for g, v in zip(gr, func):
    if g >= 2: a.text(g, v - 6.5, f"{v:.0f}", ha="center", fontsize=9, color=RED)
a.text(0, 163 + 3.2, "163", ha="center", fontsize=9, color="#444"); a.text(1, 160 + 3.2, "160", ha="center", fontsize=9, color="#444")
a.text(4.22, mech[4], "기계적 한계", color=BLUE, fontsize=9.5, fontweight="bold", va="center")
a.text(4.22, func[4], "기능적 한계", color=RED, fontsize=9.5, fontweight="bold", va="center")
a.text(3, 138.5, "쓰지 않는\n구간", ha="center", va="center", fontsize=8.5, color=RED)
a.set_xticks(gr); a.set_xticklabels(["정상", "1기", "2기", "3기", "4기"], fontsize=10)
a.set_xlim(-0.35, 5.0); a.set_ylim(105, 175); a.tick_params(axis="y", labelsize=9)
a.set_ylabel("무릎 최대 신전 한계(도)", fontsize=9.5); a.spines[["top", "right"]].set_visible(False)
a.set_title("(a) 등급별 신전 한계", fontsize=10.5, loc="left", fontweight="bold")
a.text(0.0, -0.2, "● 임상 논문 측정값(정상, 3기)   ○ 보간한 추정값   (180도 = 완전히 폄)", transform=a.transAxes, fontsize=8, color="#444")
g3 = G[3]
b.set_xlim(40, 180); b.set_ylim(-0.4, 4.2)
b.set_yticks([2.5, 1.2]); b.set_yticklabels(["수동 가동범위\n(관절이 허용)", "능동 가동범위\n(걸을 때 사용)"], fontsize=9)
b.spines[["top", "right", "left"]].set_visible(False); b.tick_params(axis="y", length=0); b.tick_params(axis="x", labelsize=9)
b.set_xticks(range(40, 181, 20))
b.set_xlabel("무릎 각도(도)     굽힘 ←   → 폄", fontsize=9)
b.set_title("(b) 3기 개의 무릎 각도 범위", fontsize=10.5, loc="left", fontweight="bold", pad=6)
b.barh(2.5, g3["prom_ext"] - g3["prom_flex"], left=g3["prom_flex"], height=0.8, color=GRAY)
b.text((g3["prom_flex"] + g3["prom_ext"]) / 2, 2.5, f"{g3['prom_flex']:.0f}~{g3['prom_ext']:.0f}도", ha="center", va="center", fontsize=9.5, color="#222")
b.barh(1.2, g3["arom_ext"] - g3["knee_mfa"], left=g3["knee_mfa"], height=0.8, color=ORANGE)
b.text((g3["knee_mfa"] + g3["arom_ext"]) / 2, 1.2, f"{g3['knee_mfa']:.0f}~{g3['arom_ext']:.0f}도", ha="center", va="center", fontsize=9.5, color="#222")
for x, col, lab, ha, dx in ((g3["prom_ext"], BLUE, f"기계적 한계 {g3['prom_ext']:.0f}도", "left", 1.5), (g3["arom_ext"], RED, f"기능적 한계 {g3['arom_ext']:.0f}도", "right", -1.5)):
    b.plot([x, x], [0.5, 3.5], "--", color=col, lw=1.8, clip_on=False); b.text(x + dx, 3.75, lab, color=col, fontsize=9.5, fontweight="bold", ha=ha, va="center")
b.annotate("", xy=(g3["arom_ext"], 0.75), xytext=(g3["prom_ext"], 0.75), arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
b.text((g3["prom_ext"] + g3["arom_ext"]) / 2, 0.1, "쓰지 않는 구간 약 25도", ha="center", va="center", fontsize=8.5, color="#444")
plt.savefig("results/figures/thesis/F22_limits.png", dpi=220, bbox_inches="tight"); plt.close(); print("F22")
