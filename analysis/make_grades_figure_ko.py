import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
d = pd.read_csv("results/final_table_leo.csv")
CODE = {"H1 energy-only, mechanical limit": ("H1 기계적", "#2f5f98", "s"), "H1 energy-only, functional limit": ("H1 기능적", "#e8923a", "^"), "H2 keep healthy excursions, functional limit": ("H2 기능적 (최종)", "#c8403f", "o")}
CLIN = {0: dict(smin=121.8, smax=167.1, hock=47.59, hip=19.02), 3: dict(smin=96.88, smax=125.1, hock=78.38, hip=35.53)}
GR = [0, 2, 3, 4]; X = {g: i for i, g in enumerate(GR)}; LAB = ["정상", "2기", "3기", "4기"]
fig = plt.figure(figsize=(5.6, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.4, wspace=0.4)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])
off = {"H1 기계적": -0.24, "H1 기능적": -0.08, "H2 기능적 (최종)": 0.08}
for key, (lab, col, mk) in CODE.items():
    sub = d[d.setting == key].set_index("grade")
    for g in GR:
        r = sub.loc[g]; x = X[g] + (off[lab] if g else -0.2)
        if g == 0 and lab != "H1 기계적": continue
        colr = "#8a8a8a" if g == 0 else col
        a.bar(x, r.stifle_max_mean - r.stifle_min_mean, bottom=r.stifle_min_mean, width=0.15, color=colr, zorder=2)
        if g:
            a.errorbar([x, x], [r.stifle_min_mean, r.stifle_max_mean], yerr=[r.stifle_min_sd, r.stifle_max_sd], fmt="none", ecolor="#222", lw=0.8, capsize=2, zorder=3)
for g, v in CLIN.items():
    x = X[g] + 0.3
    a.bar(x, v["smax"] - v["smin"], bottom=v["smin"], width=0.15, color="black", zorder=2)
a.axhspan(96.88, 125.1, xmin=0.455, xmax=0.78, color="black", alpha=0.07, zorder=0)
a.set_xticks(range(4)); a.set_xticklabels(LAB, fontsize=10); a.set_xlim(-0.55, 3.55); a.set_ylim(80, 175)
a.set_ylabel("무릎 각도(도)", fontsize=9.5); a.tick_params(axis="y", labelsize=9)
a.spines[["top", "right"]].set_visible(False)
a.set_title("(a) 걸을 때 무릎 각도 범위(180도 = 완전히 폄)", fontsize=10.5, loc="left", fontweight="bold")
handles = [Patch(color="#8a8a8a", label="정상 기준(모델)"), Patch(color="#2f5f98", label="H1 기계적"), Patch(color="#e8923a", label="H1 기능적"), Patch(color="#c8403f", label="H2 기능적(최종)"), Patch(color="black", label="임상 실측")]
a.legend(handles=handles, ncol=2, fontsize=8.2, loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False, handlelength=1.1, columnspacing=1.2, handletextpad=0.4)
for ax, key, clk, ttl in ((b, "hock_rom", "hock", "(b) 비절 가동범위"), (c, "hip_rom", "hip", "(c) 고관절 가동범위")):
    for k, (lab, col, mk) in CODE.items():
        sub = d[d.setting == k].set_index("grade")
        xs = [X[g] for g in GR]; ys = [sub.loc[g, key + "_mean"] for g in GR]; es = [sub.loc[g, key + "_sd"] for g in GR]
        ax.errorbar(xs, ys, yerr=es, fmt=mk + "-", color=col, ms=4.5, lw=1.6, capsize=2, elinewidth=0.9, zorder=2 if "H2" not in lab else 3)
    ax.plot([X[0], X[3 - 1 + 0] if False else X[3]], [CLIN[0][clk], CLIN[3][clk]], "o--", color="black", ms=6, lw=1.2, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(LAB, fontsize=9); ax.set_xlim(-0.35, 3.35)
    ax.set_title(ttl, fontsize=10.5, loc="left", fontweight="bold"); ax.set_ylabel("가동범위(도)", fontsize=9); ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
b.set_ylim(25, 85); c.set_ylim(15, 40)
b.text(2.08, 79.5, "임상 3기 78도", fontsize=8, ha="left", va="center"); c.text(2.08, 36.3, "임상 3기 36도", fontsize=8, ha="left", va="center")
fig.savefig("results/figures/thesis/F25_grades_ko.png", dpi=220, bbox_inches="tight"); plt.close(fig); print("F25")
