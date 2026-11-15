import argparse, json, sys
from pathlib import Path
import imageio, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from analysis.leo_segments import smooth_series

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--start", type=float, default=43.73); ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--label", default="leo (실측 7 kg)"); ap.add_argument("--est", default="results/leo_estimates.json"); ap.add_argument("--fps_out", type=int, default=60)
ap.add_argument("--video", default="data/real/leo2/leo_1080p120.mp4"); ap.add_argument("--kp", default="data/real/leo2/work/leo_kp.npz")
ap.add_argument("--tibia_m", type=float, default=0.127); ap.add_argument("--follow", type=float, default=0.1); ap.add_argument("--cw", type=int, default=720)
a = ap.parse_args()
z = np.load(a.kp, allow_pickle=True); KP = z["kp"]; names = list(z["names"]); ix = {n: i for i, n in enumerate(names)}
f0 = int(a.start * 120); n = int(a.seconds * 120); S = KP[f0:f0 + n]
P = lambda nm: smooth_series(S[:, ix[nm], :2], S[:, ix[nm], 2], thr=0.3, win=9); C = lambda nm: S[:, ix[nm], 2]
facing = np.sign(np.median(P("nose")[:, 0] - P("tail_base")[:, 0])) or 1.0
def best(side_pair):
    return max(side_pair, key=lambda s: sum(np.median(C(f"{s[0]}_{k}")) for k in s[1]))
hind_sides = [("back_left", ("thai", "knee", "paw")), ("back_right", ("thai", "knee", "paw"))]
fore_sides = [("front_left", ("thai", "knee", "paw")), ("front_right", ("thai", "knee", "paw"))]
hs, fs = best(hind_sides)[0], best(fore_sides)[0]
def tibia_px():
    v = []
    for s in ("left", "right"):
        m = (C(f"back_{s}_thai") > 0.5) & (C(f"back_{s}_knee") > 0.5)
        if m.sum() > 10: v.append(np.median(np.linalg.norm(P(f"back_{s}_thai")[m] - P(f"back_{s}_knee")[m], axis=1)))
    return float(np.mean(v))
mpp = a.tibia_m / tibia_px()
def angle(p, q, r):
    u = p - q; v = r - q; c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9); return np.degrees(np.arccos(np.clip(c, -1, 1)))
carpus = angle(P(f"{fs}_thai"), P(f"{fs}_knee"), P(f"{fs}_paw")); carpus_c = np.minimum(C(f"{fs}_thai"), np.minimum(C(f"{fs}_knee"), C(f"{fs}_paw")))
hock = angle(P(f"{hs}_thai"), P(f"{hs}_knee"), P(f"{hs}_paw")); hock_c = np.minimum(C(f"{hs}_thai"), np.minimum(C(f"{hs}_knee"), C(f"{hs}_paw")))
from scipy.signal import savgol_filter
ctr_b = 0.5 * (P("back_base") + P("back_end")); fvec = (P("nose") - P("tail_base")).mean(0); fvec /= np.linalg.norm(fvec)
proj = ctr_b @ fvec; tt = np.arange(len(proj)) / 120; speed_mean = float(abs(np.polyfit(tt, proj, 1)[0]) * mpp)
from analysis.leo_segments import analyse_segment
seg_out, _td = analyse_segment(KP, ix, f0, f0 + n - 1, 120); stride_hz = float(seg_out["stride_hz"])
est = json.load(open(a.est, encoding="utf-8")) if Path(a.est).exists() else None
W, H = 960, 720; PW = 480; font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 24); fsm = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
a.ch = a.cw * 3 // 4
r = imageio.get_reader(a.video); frames = []; ctr = None
LINES = [("back_base", "back_end", (255, 220, 0)), ("neck_base", "nose", (0, 220, 255)), ("tail_base", "tail_end", (255, 120, 255)),
         (f"{fs}_thai", f"{fs}_knee", (0, 255, 120)), (f"{fs}_knee", f"{fs}_paw", (0, 255, 120)), (f"{hs}_thai", f"{hs}_knee", (255, 90, 90)), (f"{hs}_knee", f"{hs}_paw", (255, 90, 90))]
PTS = {nm: P(nm) for nm in {l[0] for l in LINES} | {l[1] for l in LINES}}
step = 2
for k in range(0, n, step):
    fr = Image.fromarray(r.get_data(f0 + k)); cx, cy = np.median([PTS[nm][k] for nm in ("back_base", "back_end", f"{hs}_paw", f"{fs}_paw")], axis=0)
    ctr = np.array([cx, cy]) if ctr is None else (1 - a.follow) * ctr + a.follow * np.array([cx, cy])
    x0 = int(np.clip(ctr[0] - a.cw / 2, 0, 1920 - a.cw)); y0 = int(np.clip(ctr[1] - a.ch * 0.55, 0, 1080 - a.ch)); sc = W / a.cw
    im = fr.crop((x0, y0, x0 + a.cw, y0 + a.ch)).resize((W, H)); dr = ImageDraw.Draw(im)
    for p, q, col in LINES:
        a_, b_ = (PTS[p][k] - [x0, y0]) * sc, (PTS[q][k] - [x0, y0]) * sc; dr.line([tuple(a_), tuple(b_)], fill=col, width=5)
        for pt in (a_, b_): dr.ellipse([pt[0] - 6, pt[1] - 6, pt[0] + 6, pt[1] + 6], fill=col)
    dr.text((14, 10), a.label, font=font, fill=(255, 255, 0))
    panel = Image.new("RGB", (PW, H), (24, 26, 32)); pd = ImageDraw.Draw(panel); yy = 14
    def line(t, f=fsm, col=(235, 235, 235)):
        global yy
        pd.text((18, yy), t, font=f, fill=col); yy += 30 if f is fsm else 36
    line("영상 분석 (측면, 0.5배속)", font, (255, 220, 0)); yy += 4
    line(f"평균 속도  {speed_mean:.2f} m/s   (환산: 정강이 {a.tibia_m * 100:.1f} cm)")
    line(f"걸음 주기  {stride_hz:.2f} Hz  ({1 / stride_hz:.2f} 초)")
    line("걸음 속도  " + ("느림" if speed_mean < 0.6 else "보통" if speed_mean <= 1.0 else "빠름") + "  (0.6 미만 느림, 1.0 초과 빠름)", col=(200, 200, 255))
    line(f"앞다리 손목 각도  {carpus[k]:.0f}°" + ("" if carpus_c[k] > 0.4 else "  (신뢰 낮음)"), col=(120, 255, 170))
    line(f"뒷다리 비절 각도  {hock[k]:.0f}°" + ("" if hock_c[k] > 0.4 else "  (신뢰 낮음)"), col=(255, 140, 140))
    top, hgt = yy + 8, 190; pd.rectangle([18, top, PW - 18, top + hgt], outline=(90, 90, 100))
    for arr, col in ((carpus, (0, 255, 120)), (hock, (255, 90, 90))):
        lo, hi = 60, 180; xs = np.linspace(20, PW - 20, len(arr)); ys = top + hgt - (np.clip(arr, lo, hi) - lo) / (hi - lo) * hgt
        pd.line(list(zip(xs, ys)), fill=col, width=2)
    xk = 20 + (PW - 40) * k / len(carpus); pd.line([(xk, top), (xk, top + hgt)], fill=(255, 255, 255), width=1)
    yy = top + hgt + 14
    pd.text((18, yy), "각도: 이미지 평면의 사이각(180°=곧음)", font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15), fill=(160, 160, 170)); yy += 30
    line("무릎 판정 (시뮬레이션 기반)" + (" - 참고값" if est and est.get("knee") else ""), font, (255, 220, 0)); yy += 2
    if est and est.get("knee"):
        line(f"체중  {est['mass_kg']:.1f} kg  (입력값)")
        line(f"무릎 상태  {est['knee_ko']}", f=fsm, col=(255, 140, 140) if est["knee"] != "normal" else (120, 255, 170))
        line(f"정상 {100 * est['p_normal']:.0f} % / 이상 {100 * est['p_abnormal']:.0f} % / 심함 {100 * est['p_severe']:.0f} %")
        fn = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15); note = est.get("note", "모델 상대값, 임상 진단이 아님")
        for i in range(0, len(note), 29):
            pd.text((18, yy), note[i:i + 29], font=fn, fill=(160, 160, 170)); yy += 20
    else:
        line("판정 불가" if est else "(식별 실험 결과 대기)", col=(160, 160, 170))
        if est:
            fn = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15); note = est.get("note", "")
            for i in range(0, len(note), 29): pd.text((18, yy), note[i:i + 29], font=fn, fill=(160, 160, 170)); yy += 20
    frames.append(np.array(Image.fromarray(np.hstack([np.array(im), np.array(panel)]))))
r.close(); imageio.mimsave(a.out, frames, fps=a.fps_out, macro_block_size=1)
print("wrote", a.out, len(frames), "frames | scale %.5f m/px | stride %.2f Hz | speed %.2f m/s | hind side %s fore side %s" % (mpp, stride_hz, speed_mean, hs, fs))
