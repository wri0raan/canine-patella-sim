import argparse, subprocess, sys, os, json
import numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from analysis.leo_judge_lib import constants, clip_stifle, posterior, lpv, unit, ang

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("out"); ap.add_argument("--baseline", default=""); ap.add_argument("--truth", default=""); ap.add_argument("--title", default=""); a = ap.parse_args()
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
PW, PH, FS = 880, 720, 30.0; plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
fig = plt.figure(figsize=(PW / 100, 330 / 100), dpi=100); ax = fig.add_axes([0.09, 0.2, 0.80, 0.68]); tt = np.arange(b0 - a0) / 120.0
for k, col, lab in (("hindA", "#d62728", "뒷다리 A"), ("hindB", "#1f77b4", "뒷다리 B")): ax.plot(tt, A[k], color=col, lw=1.6, label=lab)
for k, col, lab in (("normal", "#2ca02c", "정상"), ("abnormal", "#ff7f0e", "등급 2"), ("severe", "#d62728", "등급 3")):
    ax.axhspan(mus_abs[k] - sd_abs, mus_abs[k] + sd_abs, color=col, alpha=0.13); ax.axhline(mus_abs[k], color=col, ls="--", lw=1.0); ax.text(tt[-1] * 1.005, mus_abs[k], f"{lab} {mus_abs[k]:.0f}", va="center", fontsize=8, color=col)
ax.axhline(x, color="k", lw=1.2); ax.set_xlim(0, tt[-1] * 1.0); ax.set_xlabel("시간 (초)"); ax.set_ylabel("무릎 각도 (도)"); ax.set_title("무릎 각도(영상)와 모델이 예측한 등급별 평균", fontsize=11); ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.canvas.draw(); curve = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); bbx = ax.get_window_extent(); X0, X1 = bbx.x0, bbx.x1; plt.close(fig)
FT = lambda s: ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", s); FR = lambda s: ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", s)
panel0 = Image.new("RGB", (PW, PH), (250, 250, 248)); panel0.paste(curve, (0, 0)); d = ImageDraw.Draw(panel0); y = 345
d.text((20, y), "판정 결과", fill=(20, 20, 20), font=FT(22)); y += 36
d.text((20, y), f"구간 평균 무릎 각도 {x:.1f}도 (두 다리 {lg[0]:.1f} / {lg[1]:.1f}), 온전히 보인 구간 {win:.1f}초", fill=(40, 40, 40), font=FR(17)); y += 32
d.text((20, y), "① 기준 영상 없이 모델 값만:", fill=(20, 20, 20), font=FT(17)); y += 26
d.text((36, y), "  ".join(f"{NM[k]} {v:.2f}" for k, v in p_abs.items()), fill=(40, 40, 40), font=FR(17)); y += 26
d.text((36, y), f"경증(정상~2기) {mild(p_abs):.2f} / 중증(3기 이상) {p_abs['severe']:.2f}", fill=(20, 90, 160), font=FT(17)); y += 34
if p_hyb:
    d.text((20, y), f"② 같은 개의 정상 {len(a.baseline.split(','))}구간을 기준으로:", fill=(20, 20, 20), font=FT(17)); y += 26
    d.text((36, y), "  ".join(f"{NM[k]} {v:.2f}" for k, v in p_hyb.items()), fill=(40, 40, 40), font=FR(17)); y += 26
    d.text((36, y), f"경증 {mild(p_hyb):.2f} / 중증 {p_hyb['severe']:.2f}", fill=(20, 90, 160), font=FT(17)); y += 34
fin = p_hyb or p_abs; verdict = "경증 (정상~2기 수준)" if mild(fin) >= fin["severe"] else "중증 (3기 이상 수준)"
d.rectangle([20, y, PW - 20, y + 46], fill=(225, 238, 250)); d.text((32, y + 8), f"최종 판별: {verdict}   ({'정답 ' + a.truth if a.truth else ''})", fill=(10, 50, 110), font=FT(24)); y += 62
d.text((20, y), "※ 정상과 2기는 구분할 수 없음(구간 평균이 약 5도 흩어지고 등급 2의 변화는 약 6도). 중증은 구분 가능.", fill=(110, 110, 110), font=FR(14))
CS = PH; out_w = CS + PW
p_dec = subprocess.Popen([FF, "-v", "error", "-i", f"data/real/leo5/work/c{a.tag}.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
p_enc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{PH}", "-r", str(int(FS)), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
for i in range(T):
    b = p_dec.stdout.read(int(size) * int(size) * 3)
    if len(b) < size * size * 3: break
    fr = Image.fromarray(np.frombuffer(b, np.uint8).reshape(int(size), int(size), 3).copy()).resize((CS, CS), Image.LANCZOS); dr = ImageDraw.Draw(fr); k = CS / size
    if a0 <= i < b0:
        j = i - a0; org = np.array([cg["x0"][i], cg["y0"][i]])
        pt = lambda p: tuple((np.asarray(p) - org) * sc[i] * k)
        for key, col in (("hindA", (214, 39, 40)), ("hindB", (31, 119, 180))):
            P = [pt(rump[key][j]), pt(legs[key][j, 0]), pt(legs[key][j, 1]), pt(legs[key][j, 2])]; dr.line(P, fill=col, width=4)
            for q in P[1:]: dr.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=col, width=3)
            dr.text((P[1][0] + 8, P[1][1] - 22), f"{A[key][j]:.0f}°", fill=col, font=FT(20))
        q = pt(rump["hindA"][j]); dr.ellipse([q[0] - 8, q[1] - 8, q[0] + 8, q[1] + 8], outline=(44, 160, 44), width=4); dr.text((q[0] + 10, q[1] - 24), "꼬리 뿌리(추정)", fill=(44, 160, 44), font=FT(16))
    else: dr.text((16, 16), "(개가 온전히 보이지 않는 구간: 판정에서 제외)", fill=(200, 60, 60), font=FT(18))
    pn = panel0.copy(); dp = ImageDraw.Draw(pn)
    if a0 <= i < b0:
        xp = X0 + (X1 - X0) * ((i - a0) / 120.0) / tt[-1]; dp.line([(xp, 40), (xp, 290)], fill=(60, 60, 60), width=2)
    dr.rectangle([0, CS - 38, CS, CS], fill=(0, 0, 0)); dr.text((12, CS - 32), (a.title or f"{a.tag}") + "  |  4배 느리게", fill=(255, 255, 255), font=FR(18))
    canvas = Image.new("RGB", (out_w, PH)); canvas.paste(fr, (0, 0)); canvas.paste(pn, (CS, 0)); p_enc.stdin.write(canvas.tobytes())
p_enc.stdin.close(); p_enc.wait(); print("wrote", a.out)
