import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image
plt.rcParams["font.family"] = "Malgun Gothic"
im = Image.open("data/real/leo_xray/xr4.jpg").convert("RGB")
X0, X1, Y0, Y1 = 640, 1980, 40, 2488
P = {"crest": (945, 118), "hip": (1244, 585), "stifle": (927, 1256), "hock": (1343, 2071), "mt5": (1395, 2395)}
COL = {"hip": "#c39bd3", "knee": "#ff9f43", "hock": "#1dd1a1"}
fig, ax = plt.subplots(figsize=(4.6, 8.3)); ax.imshow(im.crop((X0, Y0, X1, Y1)), extent=(X0, X1, Y1, Y0)); ax.axis("off")
for p, q in (("crest", "hip"), ("hip", "stifle"), ("stifle", "hock"), ("hock", "mt5")):
    ax.plot([P[p][0], P[q][0]], [P[p][1], P[q][1]], "-", color="white", lw=2.2, alpha=0.9)


def arc(vertex, p1, p2, col, r, tag, name, off):
    v = np.array(P[vertex], float); u1 = np.array(P[p1], float) - v; u2 = np.array(P[p2], float) - v
    t1, t2 = np.degrees(np.arctan2(u1[1], u1[0])), np.degrees(np.arctan2(u2[1], u2[0]))
    lo, hi = sorted((t1, t2))
    if hi - lo > 180: lo, hi = hi, lo + 360
    ax.add_patch(Arc(v, 2 * r, 2 * r, theta1=lo, theta2=hi, color=col, lw=4))
    mid = np.radians((lo + hi) / 2); c = v + (r + 85) * np.array([np.cos(mid), np.sin(mid)]) + np.array(off)
    ax.text(c[0], c[1], tag, color=col, fontsize=17, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.14", fc="black", ec=col, lw=2.2))
    ax.text(c[0], c[1] + 120, name, color="white", fontsize=11, ha="center", va="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))


arc("hip", "crest", "stifle", COL["hip"], 190, "①", "고관절", (-110, 0))
arc("stifle", "hip", "hock", COL["knee"], 210, "②", "무릎", (-95, 0))
arc("hock", "stifle", "mt5", COL["hock"], 190, "③", "비절", (-100, 20))
for k in ("hip", "stifle", "hock"): ax.plot(*P[k], "o", ms=12, mfc="none", mec="white", mew=2.6)
ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0)
plt.subplots_adjust(0, 0, 1, 1); plt.savefig("results/figures/thesis/F24_xray_angles.png", dpi=170); plt.close(); print("F24")
