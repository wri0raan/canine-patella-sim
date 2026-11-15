import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
R = Path("results")
def get(run):
    f = R / run / "summary.json"
    return json.load(open(f)) if f.exists() else None
mus = [("leoH3", "leoH3", 1.0), ("cmp_M90", "cmp2_M90", .9), ("cmp_M80", "cmp2_M80", .8), ("cmp_M70", "cmp2_M70", .7), ("cmp_M60", "cmp2_M60", .6), ("cmp_M50", "cmp2_M50", .5)]
fat = [("leoH3", "leoH3", 0), ("cmp_F10", "cmp2_F10", 10), ("cmp_F20", "cmp2_F20", 20), ("cmp_F30", "cmp2_F30", 30), ("cmp_F40", "cmp2_F40", 40)]
M = [("speed", "속도 (m/s)", None), ("LH_stifle_rom", "무릎 가동범위 (도)", None), ("LH_hip_rom", "고관절 가동범위 (도)", (19.0, 35.5)), ("LH_hock_rom", "비절 가동범위 (도)", None), ("LH_duty", "뒷발 접지 비율", None), ("peak_grf_LH", "뒷다리 최대 지면반력 (N)", None)]
fig, axs = plt.subplots(2, 6, figsize=(19, 6.6))
for row, (series, xl) in enumerate(((mus, "뒷다리 근력 배수"), (fat, "추가 질량 (%)"))):
    for c, (k, lab, clin) in enumerate(M):
        ax = axs[row, c]
        x = [s[2] for s in series]
        for col, sel, ls, name in ((0, 0, "-", "시드 1"), (1, 1, "--", "시드 2")):
            y = [(get(s[sel]) or {}).get(k, np.nan) for s in series]
            ax.plot(x, y, ls, marker="o", ms=4, color=("tab:blue", "tab:orange")[col], label=name)
        if row == 0 and k == "speed":
            for s in ((1.2, "cmp_P120"), (1.4, "cmp_P140")):
                v = get(s[1]); 
                if v: ax.plot([s[0]], [v[k]], "s", color="tab:blue", ms=4)
        if row == 0 and k != "speed":
            for s in ((1.2, "cmp_P120"), (1.4, "cmp_P140")):
                v = get(s[1])
                if v: ax.plot([s[0]], [v[k]], "s", color="tab:blue", ms=4)
        if clin and row == 0: ax.axhline(clin[0], ls=":", color="k", lw=.8); ax.axhline(clin[1], ls=":", color="k", lw=.8)
        ax.set_title(lab, fontsize=10); ax.set_xlabel(xl, fontsize=9); ax.grid(alpha=.3)
        if row == 0: ax.invert_xaxis() if False else None
axs[0, 0].legend(fontsize=8)
fig.suptitle("근력(위)과 추가 질량(아래)에 따른 보행 지표, 시드 1(실선)과 시드 2(점선). 네모는 시드 1만 있는 근력 1.2, 1.4. 점선(가는 검은선)은 임상 고관절 가동범위 19.0, 35.5도", fontsize=10)
fig.tight_layout(); fig.savefig(R / "figures/composition_effects.png", dpi=150); print("ok")
