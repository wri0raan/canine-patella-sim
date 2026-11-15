import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
BONE = "#efe6d2"; EDGE = "#6b5b3e"; GROOVE = "#c9a86a"; PAT = "#e8b4a0"; LIG = "#b04a4a"; MUS = "#d98b8b"

def bone_poly(ax, pts, fc=BONE, z=2): ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=EDGE, lw=1.6, zorder=z, joinstyle="round"))
def lab(ax, xy, xytext, s, ha="left", color="k", fs=10): ax.annotate(s, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=fs, color=color, arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.0, shrinkA=0, shrinkB=2), zorder=10)

def cranial(ax, luxated):
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-0.3, 8.2); ax.axis("off"); ax.set_aspect("equal")
    femur = [(-0.75, 8.0), (0.75, 8.0), (0.85, 5.2), (1.25, 3.9), (1.55, 3.4), (1.6, 2.6), (1.25, 2.1), (0.6, 2.0), (0, 2.15), (-0.6, 2.0), (-1.25, 2.1), (-1.6, 2.6), (-1.55, 3.4), (-1.25, 3.9), (-0.85, 5.2)]
    bone_poly(ax, femur)
    bone_poly(ax, [(-1.55, 3.4), (-0.42, 3.75), (-0.38, 2.55), (-1.6, 2.6)], fc="#f7f0e0", z=3)
    bone_poly(ax, [(1.55, 3.4), (0.42, 3.75), (0.38, 2.55), (1.6, 2.6)], fc="#f7f0e0", z=3)
    ax.add_patch(Polygon([(-0.4, 3.75), (0.4, 3.75), (0.36, 2.5), (0, 2.38), (-0.36, 2.5)], closed=True, fc=GROOVE, ec=EDGE, lw=1.0, zorder=3))
    tibia = [(-0.95, 1.8), (0.95, 1.8), (1.0, 1.1), (0.7, 0.55), (0.55, -0.2), (-0.55, -0.2), (-0.7, 0.55), (-1.0, 1.1)]
    bone_poly(ax, tibia)
    tt = (-0.35, 0.95) if luxated else (0.0, 0.95)
    ax.add_patch(Ellipse(tt, 0.55, 0.5, fc="#f7f0e0", ec=EDGE, lw=1.4, zorder=4))
    px = -1.35 if luxated else 0.0
    ax.add_patch(Polygon([(-0.55, 8.0), (0.55, 8.0), (0.2 + px * 0.5, 4.5), (-0.2 + px * 0.5, 4.5)], closed=True, fc=MUS, ec="#a05555", lw=1.0, alpha=0.8, zorder=1))
    ax.add_patch(Ellipse((px, 3.55 if not luxated else 3.25), 0.8, 1.15, fc=PAT, ec="#8a4b3a", lw=1.6, zorder=6))
    ax.plot([px, tt[0]], [3.0 if not luxated else 2.7, tt[1] + 0.25], color=LIG, lw=3, zorder=5, solid_capstyle="round")
    ax.text(0, 8.15, "대퇴골(넙다리뼈)" if False else "", fontsize=1)

def lateral(ax):
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-0.3, 8.2); ax.axis("off"); ax.set_aspect("equal")
    femur = [(-1.0, 8.0), (0.5, 8.0), (0.75, 5.5), (1.35, 4.2), (1.45, 3.4), (1.2, 2.5), (0.6, 2.0), (-0.3, 1.95), (-1.0, 2.5), (-1.25, 3.5), (-1.1, 5.5)]
    bone_poly(ax, femur)
    ax.add_patch(Polygon([(1.45, 3.8), (1.05, 3.55), (1.0, 2.85), (1.25, 2.75)], closed=True, fc=GROOVE, ec=EDGE, lw=1.0, zorder=3))
    tibia = [(-0.8, 1.7), (0.95, 1.7), (1.25, 1.3), (1.35, 0.7), (0.9, -0.2), (-0.15, -0.2), (-0.35, 0.8)]
    bone_poly(ax, tibia)
    ax.add_patch(Ellipse((1.35, 0.95), 0.5, 0.65, fc="#f7f0e0", ec=EDGE, lw=1.4, zorder=4))
    ax.add_patch(Polygon([(-0.2, 8.0), (0.5, 8.0), (1.5, 4.6), (1.2, 4.3)], closed=True, fc=MUS, ec="#a05555", lw=1.0, alpha=0.8, zorder=1))
    ax.add_patch(Ellipse((1.5, 3.25), 0.75, 1.05, angle=-8, fc=PAT, ec="#8a4b3a", lw=1.6, zorder=6))
    ax.plot([1.55, 1.38], [2.7, 1.2], color=LIG, lw=3, zorder=5, solid_capstyle="round")

fig, axs = plt.subplots(1, 3, figsize=(15.5, 8.4))
cranial(axs[0], False); cranial(axs[1], True); lateral(axs[2])
axs[0].set_title("A. 정상: 왼쪽 뒷다리 무릎을 앞에서 본 모습", fontsize=12, weight="bold"); axs[1].set_title("B. 내측 슬개골 탈구 (같은 방향에서)", fontsize=12, weight="bold"); axs[2].set_title("C. 옆에서 본 무릎 (앞쪽이 오른쪽)", fontsize=12, weight="bold")
ax = axs[0]
lab(ax, (0.0, 6.5), (-3.45, 7.2), "대퇴골(넙다리뼈)", fs=10.5)
lab(ax, (0.0, 3.1), (1.85, 4.9), "활차구(trochlear groove)\n= 슬개골이 미끄러지는 홈", color="#7a5a1e", fs=10.5)
lab(ax, (-0.95, 3.2), (-3.45, 4.6), "내측 활차융기\n(몸 중앙쪽 둔덕)", fs=10)
lab(ax, (1.0, 3.2), (1.85, 3.5), "외측 활차융기\n(바깥쪽 둔덕)", fs=10)
lab(ax, (0.0, 3.55), (-3.45, 3.5), "슬개골(무릎뼈)", color="#8a4b3a", fs=10.5)
lab(ax, (0.0, 1.8), (1.85, 1.7), "슬개인대", color=LIG, fs=10.5)
lab(ax, (0.0, 0.95), (1.85, 0.7), "경골 조면(정강뼈 앞쪽 돌기)", fs=10)
lab(ax, (0.0, 0.2), (-3.45, 0.7), "경골(정강뼈)", fs=10.5)
ax.text(-3.45, 8.0, "← 내측(몸 중앙)   외측 →", fontsize=9.5, color="#555", va="center")
ax = axs[1]
lab(ax, (-1.35, 3.25), (-3.45, 4.7), "슬개골이 홈 밖\n(내측)으로 이탈", color="#8a4b3a", fs=10.5)
lab(ax, (0.0, 3.0), (1.85, 4.6), "비어 있는 활차구\n(얕거나 좁은 경우가 많음)", color="#7a5a1e", fs=10.5)
lab(ax, (-0.35, 0.95), (1.85, 0.7), "경골 조면이 안쪽으로\n치우치는 경우가 많음", fs=10)
lab(ax, (-0.7, 2.0), (-3.45, 1.6), "슬개인대가 비스듬해져\n계속 안쪽으로 당김", color=LIG, fs=10)
ax = axs[2]
lab(ax, (-0.2, 6.5), (-3.45, 7.0), "대퇴골", fs=10.5)
lab(ax, (1.1, 3.2), (-3.45, 3.2), "활차구\n(앞면의 홈)", color="#7a5a1e", fs=10.5)
lab(ax, (1.5, 3.25), (2.2, 5.2), "슬개골", color="#8a4b3a", fs=10.5)
lab(ax, (0.9, 6.0), (-3.45, 5.5), "대퇴사두근(넙다리네갈래근)\n→ 슬개골을 앞으로 당김", color="#a05555", fs=10)
lab(ax, (1.45, 2.0), (2.2, 2.3), "슬개인대", color=LIG, fs=10.5)
lab(ax, (1.35, 0.95), (2.2, 0.8), "경골 조면", fs=10)
lab(ax, (0.2, 0.2), (-3.45, 0.7), "경골", fs=10.5)
ax.text(-3.45, 8.0, "← 뒤   앞 →", fontsize=9.5, color="#555", va="center")
fig.suptitle("개 무릎(슬관절) 구조: 대퇴골 활차구의 위치와 내측 슬개골 탈구 (모식도, 비례 아님)", fontsize=14, weight="bold", y=0.98)
note = ("등급(Singleton): 1기 손으로 밀면 빠지지만 저절로 돌아옴 / 2기 무릎을 굽히거나 누르면 빠지고, 펴거나 손으로 되돌릴 때까지 머무름 / "
        "3기 대부분 빠져 있고 손으로 되돌리면 들어감 / 4기 늘 빠져 있고 손으로도 되돌릴 수 없음\n"
        "본 연구 모델: 슬개골과 활차구를 따로 모델링하지 않고, 슬관절(무릎) 신전 한계(수동 PROM 또는 보행 중 AROM)를 등급별로 낮추는 방식으로 표현함. 임상 실측은 Grade 0과 III만 사용.")
fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9.5, color="#333", linespacing=1.5)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig("results/figures/trochlea_diagram.png", dpi=170); print("saved")
