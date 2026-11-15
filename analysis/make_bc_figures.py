import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
OUT = "results/figures/thesis"
NAMES = ["tail", "pelvis", "hipL", "stifL", "hockL", "mtpL", "toeL", "hipR", "stifR", "hockR", "mtpR", "toeR", "elbR", "carpR", "pawR"]


def load(run):
    z = np.load(f"results/bcA_joints_{run}.npz"); return z["joints"], list(z["frames"])


def img(run, f): return Image.open(f"results/bcA_frames_{run}/f{f:04d}.png").convert("RGB")


def knee_angle(J, f_i):
    P = {n: J[f_i, i] for i, n in enumerate(NAMES)}; u = P["hipR"] - P["stifR"]; v = P["hockR"] - P["stifR"]
    return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))), P


def draw(ax, J, f_i, crop, labels=False, fs=9):
    ang, P = knee_angle(J, f_i); x0, y0, x1, y1 = crop; sh = np.array([x0, y0])
    pts = [P["hipR"], P["stifR"], P["hockR"], P["mtpR"], P["toeR"]]; cols = ["#06d6a0", "#4cc9f0", "#c8b6ff", "#ffd166"]
    for a_, b_, c in zip(pts[:-1], pts[1:], cols): ax.plot([a_[0] - x0, b_[0] - x0], [a_[1] - y0, b_[1] - y0], "-", color=c, lw=3.2, solid_capstyle="round")
    for q in pts: ax.plot(q[0] - x0, q[1] - y0, "o", ms=6, mfc="white", mec="black", mew=1.2)
    st = P["stifR"] - sh; u = P["hipR"] - P["stifR"]; v = P["hockR"] - P["stifR"]; a1, a2 = np.degrees(np.arctan2(u[1], u[0])), np.degrees(np.arctan2(v[1], v[0]))
    lo, hi = sorted([a1, a2]);
    if hi - lo > 180: lo, hi = hi, lo + 360
    ax.add_patch(Arc(st, 70, 70, theta1=lo, theta2=hi, color="yellow", lw=2.5)); ax.text(st[0] - 85, st[1] + 4, f"{ang:.0f}°", color="yellow", fontsize=fs + 2, fontweight="bold", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1))
    if labels:
        T = lambda n, dx, dy, t, c: ax.annotate(t, xy=P[n] - sh, xytext=P[n] - sh + np.array([dx, dy]), fontsize=fs, fontweight="bold", color="black", ha="center", bbox=dict(boxstyle="round,pad=0.2", fc=c, ec="none", alpha=0.92), arrowprops=dict(arrowstyle="-", color=c, lw=1.6))
        T("hipR", -150, -70, "고관절 (엉덩관절)", "#06d6a0"); T("stifR", -105, -48, "무릎 (슬관절)", "#4cc9f0"); T("hockR", 120, -40, "비절 (발목관절)", "#c8b6ff"); T("toeR", 105, 14, "발", "#ffd166"); T("tail", 40, -110, "꼬리 뿌리", "#ef476f")
        mid = 0.5 * (P["hipR"] + P["stifR"]) - sh; ax.annotate("대퇴골", xy=mid, xytext=mid + np.array([95, -20]), fontsize=fs, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="#06d6a0", ec="none", alpha=0.92), arrowprops=dict(arrowstyle="-", color="#06d6a0", lw=1.6))
        mid = 0.5 * (P["stifR"] + P["hockR"]) - sh; ax.annotate("경골 (정강이뼈)", xy=mid, xytext=mid + np.array([-100, 20]), fontsize=fs, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="#4cc9f0", ec="none", alpha=0.92), arrowprops=dict(arrowstyle="-", color="#4cc9f0", lw=1.6))
        mid = 0.5 * (P["hockR"] + P["mtpR"]) - sh; ax.annotate("중족골", xy=mid, xytext=mid + np.array([95, 22]), fontsize=fs, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="#c8b6ff", ec="none", alpha=0.92), arrowprops=dict(arrowstyle="-", color="#c8b6ff", lw=1.6))
    if labels: ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
    return ang


J, fr = load("fitR3"); k = int(min(range(40, 130), key=lambda i: knee_angle(J, i)[0])); f = fr[k]; im = img("fitR3", f); crop = (60, 20, 700, 430)
fig, ax = plt.subplots(figsize=(9.5, 6.2)); ax.imshow(im.crop(crop)); ax.axis("off"); draw(ax, J, k, crop, labels=True, fs=10)
ax.set_title("보더콜리 피부를 입힌 물리 모델(정상 걸음)과 뒷다리 뼈의 도식", fontsize=12); plt.tight_layout(); plt.savefig(f"{OUT}/F20_bc_bones.png", dpi=150); plt.close(); print("F20")

runs = (("fitR3", "정상"), ("fitR3_G2", "등급 2"), ("fitR3_G3", "등급 3")); nph = 6; t0 = 90
fig, axs = plt.subplots(3, nph, figsize=(3.0 * nph, 7.2)); crop = (150, 100, 560, 400)
for r, (run, lab) in enumerate(runs):
    J, fr = load(run); angs = []
    for c in range(nph):
        k = t0 + c * 3; a_ = axs[r, c]; a_.imshow(img(run, fr[k]).crop(crop)); a_.axis("off"); ang = draw(a_, J, k, crop, fs=8); angs.append(ang)
        if r == 0: a_.set_title(f"{(c * 3) / 30:.2f}초", fontsize=9)
    axs[r, 0].text(-0.04, 0.5, f"{lab}\n무릎 각도 {min(angs):.0f}~{max(angs):.0f}°", transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=11, fontweight="bold")
fig.suptitle("한 걸음 안에서 본 무릎 각도(노란 호): 정상, 등급 2, 등급 3 (보더콜리 피부, 뼈대는 물리 모델)", fontsize=12); plt.tight_layout(rect=(0.06, 0, 1, 0.96)); plt.savefig(f"{OUT}/F21_bc_grades_strip.png", dpi=130); plt.close(); print("F21")
