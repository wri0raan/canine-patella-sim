import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
OUT = "results/figures/thesis"
COL = dict(spine="#ffd166", pelvis="#ef476f", femur="#06d6a0", tibia="#4cc9f0", other="#c8b6ff")


def label(ax, xy, txt, xyt, col, fs=10):
    ax.annotate(txt, xy=xy, xytext=xyt, color="black", fontsize=fs, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none", alpha=0.92), arrowprops=dict(arrowstyle="-", color=col, lw=1.8, shrinkA=0, shrinkB=0))
    ax.plot(*xy, "o", ms=4, color=col)


im = Image.open("data/real/leo_xray/xr3.jpg").convert("RGB").resize((1687, 2000)); fig, ax = plt.subplots(figsize=(8.6, 10)); ax.imshow(im); ax.axis("off")
label(ax, (975, 300), "요추 (허리뼈)", (1330, 200), COL["spine"]); label(ax, (950, 560), "천추 (엉치뼈)", (1330, 330), COL["spine"])
label(ax, (1170, 520), "장골 (엉덩뼈)", (1450, 520), COL["pelvis"]); label(ax, (820, 480), "장골 (엉덩뼈)", (420, 470), COL["pelvis"])
label(ax, (985, 880), "폐쇄공", (1330, 990), COL["pelvis"]); label(ax, (770, 1010), "좌골 · 치골\n(골반 바닥)", (400, 1090), COL["pelvis"])
label(ax, (1075, 830), "고관절\n(대퇴골두와 관골구)", (1450, 760), COL["femur"]); label(ax, (745, 830), "고관절", (330, 760), COL["femur"])
label(ax, (1185, 825), "대전자", (1470, 880), COL["femur"], 9); label(ax, (655, 780), "대전자", (260, 840), COL["femur"], 9)
label(ax, (1215, 1100), "대퇴골 (넙다리뼈)", (1500, 1180), COL["femur"]); label(ax, (590, 1100), "대퇴골 (넙다리뼈)", (180, 1180), COL["femur"])
label(ax, (1225, 1440), "무릎 (슬관절)\n슬개골은 대퇴골 먼 쪽에 겹쳐 보임", (1475, 1520), COL["femur"], 9); label(ax, (560, 1420), "무릎 (슬관절)", (240, 1330), COL["femur"])
label(ax, (1175, 1650), "경골 (정강이뼈)", (1480, 1720), COL["tibia"]); label(ax, (560, 1650), "경골 (정강이뼈)", (190, 1720), COL["tibia"])
label(ax, (1245, 1710), "비골 (종아리뼈)", (1480, 1830), COL["tibia"], 9); label(ax, (600, 1730), "비골 (종아리뼈)", (190, 1830), COL["tibia"], 9)
label(ax, (880, 1200), "꼬리뼈 (미추)", (1010, 1260), COL["spine"], 9)
ax.annotate("", xy=(745, 1345), xytext=(585, 1345), arrowprops=dict(arrowstyle="-|>", color="white", lw=3)); ax.text(665, 1305, "내측 탈구 방향\n(몸 중심 쪽)", color="white", fontsize=9, ha="center", fontweight="bold", bbox=dict(fc="black", ec="none", alpha=0.6, pad=2))
ax.plot([735, 555], [820, 1430], "--", color="white", lw=1.4); ax.plot([545, 590], [1470, 1850], "--", color="white", lw=1.4); ax.text(470, 1000, "대퇴골 축", color="white", fontsize=9, rotation=75); ax.text(615, 1600, "경골 축", color="white", fontsize=9, rotation=84)
ax.set_title("Raeo 골반과 뒷다리 방사선 사진(정면, 배-등 방향)의 뼈 이름 도식", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/F18_xray_VD_bones.png", dpi=150); plt.close(); print("F18")

im4 = Image.open("data/real/leo_xray/xr4.jpg").convert("RGB"); H = {"hip": (1244, 585), "stifle": (927, 1256), "plateau": (945, 1343), "hock": (1343, 2071), "tip": (1425, 1965), "pat": (806, 1200)}
fig, ax = plt.subplots(figsize=(8.4, 10.4)); ax.imshow(im4); ax.axis("off")
ax.plot([H["hip"][0], H["stifle"][0]], [H["hip"][1], H["stifle"][1]], "-", color=COL["femur"], lw=3, alpha=0.9); ax.plot([H["stifle"][0], H["hock"][0]], [H["stifle"][1], H["hock"][1]], "-", color=COL["tibia"], lw=3, alpha=0.9)
for k in ("hip", "stifle", "hock"): ax.plot(*H[k], "o", ms=11, mfc="none", mec="white", mew=2.5)
label(ax, (1244, 585), "고관절 중심\n(대퇴골두)", (1650, 560), COL["femur"]); label(ax, (1085, 920), "대퇴골 (넙다리뼈)", (1560, 960), COL["femur"])
label(ax, (822, 1212), "슬개골\n(무릎뼈)", (430, 1130), COL["femur"]); label(ax, (927, 1256), "슬관절 중심\n(대퇴골 먼 쪽 중심)", (420, 1380), COL["femur"])
label(ax, (1130, 1650), "경골 (정강이뼈)", (1560, 1560), COL["tibia"]); label(ax, (1010, 1560), "비골 (종아리뼈)", (480, 1620), COL["tibia"], 9)
label(ax, (1425, 1990), "종골 (뒤꿈치뼈) 끝\n= 피부 표식", (1640, 1850), COL["other"], 9); label(ax, (1343, 2071), "비절(발목관절) 중심", (1560, 2230), COL["other"])
label(ax, (1370, 2330), "중족골", (1700, 2380), COL["other"]); label(ax, (1300, 520), "골반", (1650, 420), COL["pelvis"]); label(ax, (500, 250), "요추", (350, 120), COL["spine"]); label(ax, (1700, 380), "꼬리뼈", (1800, 250), COL["spine"])
v1 = np.array(H["hip"]) - np.array(H["stifle"]); v2 = np.array(H["hock"]) - np.array(H["stifle"]); a1, a2 = np.degrees(np.arctan2(v1[1], v1[0])), np.degrees(np.arctan2(v2[1], v2[0])); ang = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
from matplotlib.patches import Arc
ax.add_patch(Arc(H["stifle"], 330, 330, theta1=min(a1, a2), theta2=max(a1, a2), color="yellow", lw=3)); ax.text(H["stifle"][0] + 230, H["stifle"][1] + 40, f"무릎 사이각\n약 {ang:.0f}도\n(이 자세의 값)", color="yellow", fontsize=10, fontweight="bold")
ax.text(60, 2400, "무릎 각도 = 엉덩이 기준점-무릎-비절 사이각 (영상), 대퇴골-경골 사이각 (모델의 관절각)", color="white", fontsize=9, bbox=dict(fc="black", ec="none", alpha=0.6))
ax.set_title("Raeo 뒷다리 방사선 사진(측면)의 뼈 이름과 관절 중심, 무릎 각도의 정의", fontsize=12); plt.tight_layout(); plt.savefig(f"{OUT}/F19_xray_lateral_bones.png", dpi=150); plt.close(); print("F19")
