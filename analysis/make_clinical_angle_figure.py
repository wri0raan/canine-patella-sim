import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
C1, C2, C3 = "#8e44ad", "#d35400", "#16a085"
FS = 1.35
fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 5.4), gridspec_kw=dict(width_ratios=[1.0, 1.2]))

P = {"crest": (945, 118), "troch": (1215, 535), "stifle": (927, 1256), "mall": (1343, 2071), "mt5": (1395, 2395)}
P = {k: (x / 1000, -y / 1000) for k, (x, y) in P.items()}
segs = [("crest", "troch", "#7f8c8d"), ("troch", "stifle", "#2c3e50"), ("stifle", "mall", "#2c3e50"), ("mall", "mt5", "#2c3e50")]
for p, q, col in segs: a.plot([P[p][0], P[q][0]], [P[p][1], P[q][1]], "-", color=col, lw=5, solid_capstyle="round", zorder=2)
for k, (x, y) in P.items(): a.plot(x, y, "o", ms=9, mfc="white", mec="black", mew=2, zorder=4)


def arc(vertex, p1, p2, col, r, tag):
    v, u1, u2 = np.array(P[vertex]), np.array(P[p1]) - np.array(P[vertex]), np.array(P[p2]) - np.array(P[vertex])
    t1, t2 = np.degrees(np.arctan2(u1[1], u1[0])), np.degrees(np.arctan2(u2[1], u2[0]))
    lo, hi = sorted((t1, t2))
    if hi - lo > 180: lo, hi = hi, lo + 360
    a.add_patch(Arc(v, 2 * r, 2 * r, theta1=lo, theta2=hi, color=col, lw=3, zorder=3))
    mid = np.radians((lo + hi) / 2)
    a.text(v[0] + (r + 0.15) * np.cos(mid), v[1] + (r + 0.15) * np.sin(mid), tag, color=col, fontsize=15 * FS * 0.8, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=col, lw=2))


arc("troch", "crest", "stifle", C1, 0.30, "①"); arc("stifle", "troch", "mall", C2, 0.36, "②"); arc("mall", "stifle", "mt5", C3, 0.30, "③")
lab = {"crest": ("장골능", (-0.12, 0.0), "right"), "troch": ("대전자", (0.14, 0.12), "left"), "stifle": ("외측 대퇴경골\n관절(무릎)", (-0.16, 0.0), "right"), "mall": ("외측 복사뼈\n(발목)", (0.17, 0.0), "left"), "mt5": ("제5 중족골", (0.15, -0.02), "left")}
for k, (t, (dx, dy), ha) in lab.items(): a.text(P[k][0] + dx, P[k][1] + dy, t, fontsize=10.5 * FS, ha=ha, va="center")
a.text(P["crest"][0] + 0.02, P["crest"][1] + 0.30, "골반", color="#7f8c8d", fontsize=10 * FS, ha="center")
a.set_aspect("equal"); a.set_xlim(-0.45, 2.0); a.set_ylim(-2.95, 0.6); a.axis("off")
a.set_title("(a) 마커 다섯 곳(○)과 세 관절 각도", fontsize=12 * FS, loc="left")
a.text(-0.4, -2.75, "①고관절 ②무릎 ③비절(발목)\n각도 = 이웃한 두 분절의 사이각", fontsize=9.5 * FS, color="#333", va="center")

b.set_xlim(88, 186); b.set_ylim(-0.1, 4.3); b.set_yticks([]); b.spines[["top", "right", "left"]].set_visible(False)
b.set_xticks(range(90, 181, 30)); b.tick_params(labelsize=10 * FS); b.set_xlabel("무릎 각도 (도), 오른쪽일수록 더 폄", fontsize=10.5 * FS)
b.set_title("(b) ② 무릎 각도의 눈금과 범위", fontsize=12 * FS, loc="left")
rows = ((3.2, "정상군", "#2980b9", 121.8, 167.1, 162.9), (1.5, "Grade III", "#e67e22", 96.88, 125.1, 149.7))
for y, name, col, lo, hi, prom in rows:
    b.barh(y, hi - lo, left=lo, height=0.5, color=col, alpha=0.9, zorder=2)
    b.text(lo, y + 0.42, f"{lo:.1f}", ha="center", fontsize=10 * FS, color=col, fontweight="bold"); b.text(hi, y + 0.42, f"{hi:.1f}", ha="center", fontsize=10 * FS, color=col, fontweight="bold")
    b.text(88.5, y + 0.78, name, ha="left", va="center", fontsize=11.5 * FS, fontweight="bold", color=col)
    b.plot(prom, y, "D", color="black", ms=8, zorder=3)
    b.text(prom, (y - 0.5) if name == "정상군" else (y + 0.42), f"{prom:.1f}", ha="center", va="top" if name == "정상군" else "bottom", fontsize=10 * FS)
b.annotate("", xy=(149.7, 1.5), xytext=(125.1, 1.5), arrowprops=dict(arrowstyle="<->", color="#a04000", lw=1.6, shrinkA=0, shrinkB=6))
b.text(137.4, 0.95, "쓰지 않는\n약 25도", ha="center", va="center", fontsize=10 * FS, color="#a04000")
b.plot([180, 180], [0.7, 3.9], "--", color="#888", lw=1.2); b.text(179, 4.05, "완전 신전", ha="right", fontsize=9.5 * FS, color="#666")
b.text(88.5, 0.15, "막대: 능동 가동범위(왼쪽 끝 최대 굴곡, 오른쪽 끝 최대 신전)\n◆: 수동 최대 신전(사람이 끝까지 편 한계)", fontsize=9 * FS, color="#333", ha="left", va="center")
plt.tight_layout(); plt.savefig("results/figures/thesis/F23_clinical_angles.png", dpi=190); plt.close(); print("F23")
