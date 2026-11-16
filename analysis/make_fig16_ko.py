import argparse, subprocess, sys, os, json
import numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from analysis.leo_judge_lib import constants, clip_stifle, posterior, lpv, unit, ang

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("out"); ap.add_argument("--baseline", default=""); ap.add_argument("--truth", default=""); ap.add_argument("--title", default=""); ap.add_argument("--frame", type=float, default=0.88); a = ap.parse_args()
FF = imageio_ffmpeg.get_ffmpeg_exe(); C = constants(runset=os.environ.get("RUNSET", "R3")); c = C["c_rump"]
cg = np.load(f"data/real/leo5/work/c{a.tag}_crop.npz"); fz = np.load(f"data/real/leo5/work/fused/{a.tag}_tracks.npz"); ix = {n: i for i, n in enumerate(fz["names"])}; ins = np.nonzero(fz["inside"])[0]; a0, b0 = int(ins[0]), int(ins[-1]) + 1
T = len(cg["x0"]); size = float(cg["size"]); sc = size / cg["side"]
body = fz["body"][a0:b0]; be, bb = lpv(body[:, ix["back_end"], :2]), lpv(body[:, ix["back_base"], :2]); ub = unit(be - bb)
legs = {k: np.stack([lpv(fz[k][a0:b0][:, i, :2]) for i in range(3)], 1) for k in ("hindA", "hindB")}
Tv = {k: float(np.median(np.linalg.norm(v[:, 0] - v[:, 1], axis=1))) for k, v in legs.items()}
rump = {k: be + c * Tv[k] * ub for k in legs}; A = {k: ang(rump[k], legs[k][:, 0], legs[k][:, 1]) for k in legs}
x, win, lg = clip_stifle(a.tag, c)
mus_abs = {"normal": C["mu_normal"], "abnormal": C["mu_normal"] + C["d_abnormal"], "severe": C["mu_normal"] + C["d_severe"]}; sd_abs = float(np.sqrt(4.5 ** 2 + C["sd_sim"] ** 2 + C["sd_c"] ** 2 + C["sd_tracker"] ** 2)); p_abs = posterior(x, mus_abs, sd_abs)
p_hyb = None
if a.baseline:
    B = [clip_stifle(t, c)[0] for t in a.baseline.split(",")]; mb = float(np.mean(B)); n = len(B); sdw = max(float(np.std(B, ddof=1)), 4.0); mus_h = {"normal": mb, "abnormal": mb + C["d_abnormal"], "severe": mb + C["d_severe"]}; p_hyb = posterior(x, mus_h, float(np.sqrt(sdw ** 2 * (1 + 1 / n) + 4.0)))
NM = {"normal": "정상", "abnormal": "이상(2기 수준)", "severe": "심함(3기 수준)"}
mild = lambda p: p["normal"] + p["abnormal"]

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FT = lambda s: ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", s)
iw = a0 + int(round(a.frame * 120)); i_sel = a0 + int(round(a.frame * 120))
p_dec = subprocess.Popen([FF, "-v", "error", "-i", f"data/real/leo5/work/c{a.tag}.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
for i in range(i_sel + 1):
    b = p_dec.stdout.read(int(size) * int(size) * 3)
p_dec.kill(); CS = 720; k = CS / size
fr = Image.fromarray(np.frombuffer(b, np.uint8).reshape(int(size), int(size), 3).copy()).resize((CS, CS), Image.LANCZOS); dr = ImageDraw.Draw(fr)
j = i_sel - a0; org = np.array([cg["x0"][i_sel], cg["y0"][i_sel]]); pt = lambda p: tuple((np.asarray(p) - org) * sc[i_sel] * k)
for key, col in (("hindA", (214, 39, 40)), ("hindB", (31, 119, 180))):
    P = [pt(rump[key][j]), pt(legs[key][j, 0]), pt(legs[key][j, 1]), pt(legs[key][j, 2])]; dr.line(P, fill=col, width=7)
    for q in P[1:]: dr.ellipse([q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9], outline=col, width=5)
    dr.text((P[1][0] + (14 if key == "hindA" else -118), P[1][1] - 44), f"{A[key][j]:.0f}\u00b0", fill=col, font=FT(40))
q = pt(rump["hindA"][j]); dr.ellipse([q[0] - 11, q[1] - 11, q[0] + 11, q[1] + 11], outline=(44, 160, 44), width=6); dr.text((q[0] + 14, q[1] - 40), "\uaf2c\ub9ac \ubfcc\ub9ac", fill=(44, 160, 44), font=FT(34))
fig = plt.figure(figsize=(6.6, 3.9))
axp = fig.add_axes([0.005, 0.12, 0.43, 0.727]); axp.imshow(fr); axp.axis("off"); axp.text(0.02, 0.97, "(a)", transform=axp.transAxes, fontsize=12, fontweight="bold", va="top", color="white", bbox=dict(fc="black", ec="none", alpha=0.6, pad=2))
ax = fig.add_axes([0.55, 0.70, 0.31, 0.25]); tt = np.arange(b0 - a0) / 120.0
for kk, col in (("hindA", "#d62728"), ("hindB", "#1f77b4")): ax.plot(tt, A[kk], color=col, lw=1.1)
for kk, col, lab, dy in (("normal", "#2ca02c", "정상", 4.5), ("abnormal", "#ff7f0e", "2기", -4.0), ("severe", "#d62728", "3기", 0.0)):
    ax.axhspan(mus_abs[kk] - sd_abs, mus_abs[kk] + sd_abs, color=col, alpha=0.14); ax.axhline(mus_abs[kk], color=col, ls="--", lw=0.8); ax.text(tt[-1] * 1.03, mus_abs[kk] + dy, f"{lab} {mus_abs[kk]:.0f}", va="center", fontsize=8, color=col)
ax.axhline(x, color="k", lw=0.9); ax.axvline((i_sel - a0) / 120.0, color="#555", lw=1.0); ax.set_xlim(0, tt[-1]); ax.set_ylim(75, 180)
ax.set_xlabel("시간 (초)", fontsize=8.5, labelpad=1); ax.set_ylabel("무릎 각도 (도)", fontsize=8.5, labelpad=4); ax.tick_params(labelsize=8, pad=1.5); ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.27, 1.06, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold")
axt = fig.add_axes([0.45, 0.0, 0.55, 0.55]); axt.axis("off")
mild = lambda p: p["normal"] + p["abnormal"]
fin = p_hyb or p_abs; verdict = "경증" if mild(fin) >= fin["severe"] else "중증"
def block(y0, title, p):
    axt.text(0.0, y0, title, fontsize=9.3, fontweight="bold", color="#222", va="top")
    axt.text(0.04, y0 - 0.115, f"정상 {p['normal']:.2f} · 2기 수준 {p['abnormal']:.2f} · 3기 수준 {p['severe']:.2f}", fontsize=9, color="#222", va="top")
    axt.text(0.04, y0 - 0.225, f"경증 {mild(p):.2f} / 중증 {p['severe']:.2f}", fontsize=9.3, fontweight="bold", color="#145a9c", va="top")
axt.text(0.0, 1.0, f"구간 평균 무릎 각도 {x:.1f}도", fontsize=10, fontweight="bold", color="#222", va="top")
block(0.86, "① 모델 값만 사용", p_abs)
if p_hyb: block(0.50, "② 같은 개의 다른 검사 영상 1~4 기준", p_hyb)
axt.add_patch(plt.Rectangle((0.0, 0.0), 0.99, 0.14, fc="#e1eefa", ec="none")); axt.text(0.02, 0.07, f"최종 판별: {verdict}" + (f" (실제 {a.truth})" if a.truth else ""), fontsize=10.5, fontweight="bold", color="#0a3270", va="center")
fig.savefig(a.out, dpi=240); print("wrote", a.out)
