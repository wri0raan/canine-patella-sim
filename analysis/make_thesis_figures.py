import sys, os, json, glob, subprocess
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from analysis.leo_judge_lib import constants, clip_stifle, posterior, lpv, unit, ang
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
OUT = "results/figures/thesis"; os.makedirs(OUT, exist_ok=True); FF = imageio_ffmpeg.get_ffmpeg_exe()
FT = lambda s: ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", s)
EN = os.environ.get("FIG_LANG") == "en"; T = lambda ko, en: en if EN else ko
if EN: OUT = "docs/readme/en"; os.makedirs(OUT, exist_ok=True)
want = set(sys.argv[1:]); on = lambda k: not want or k in want
C = constants(runset="R3"); CR = C["c_rump"]


def frame(path, idx, size):
    r = subprocess.run([FF, "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    return Image.fromarray(np.frombuffer(r.stdout, np.uint8).reshape(size, size, 3).copy())


def tracks(tag):
    cg = np.load(f"data/real/leo5/work/c{tag}_crop.npz"); fz = np.load(f"data/real/leo5/work/fused/{tag}_tracks.npz"); return cg, fz, {n: i for i, n in enumerate(fz["names"])}


def to_crop(p, cg, i):
    return tuple((np.asarray(p) - np.array([cg["x0"][i], cg["y0"][i]])) * float(cg["size"]) / cg["side"][i])


if on("F01"):
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off"); ax.set_xlim(0, 12.6); ax.set_ylim(0, 4.2)
    def box(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#444", lw=1.2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=12)
    def arr(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color="#444", lw=1.5))
    ax.text(0.2, 3.95, "영상", fontsize=13, fontweight="bold", color="#1f4e79"); ax.text(0.2, 1.62, "모델", fontsize=13, fontweight="bold", color="#7a3e00")
    A = [("걷기 영상\n(옆모습)", 0.2), ("관절 추적\n(무릎 각도 추출)", 3.3), ("영상의\n무릎 각도", 6.4)]
    for t, x in A: box(x, 2.65, 2.5, 1.05, t, "#dcebf8")
    for i in range(2): arr((A[i][1] + 2.58, 3.17), (A[i + 1][1] - 0.08, 3.17))
    Bx = [("영상과 X-ray로\n물리 모델 제작", 0.2), ("정상·등급별\n걸음 합성", 3.3), ("모델의 등급별\n무릎 각도 예측", 6.4)]
    for t, x in Bx: box(x, 0.35, 2.5, 1.05, t, "#fbe5d0")
    for i in range(2): arr((Bx[i][1] + 2.58, 0.87), (Bx[i + 1][1] - 0.08, 0.87))
    box(9.9, 1.5, 2.5, 1.2, "같은 정의로 비교\n경증 / 중증 구분", "#e2f0d9")
    arr((8.98, 3.0), (10.6, 2.78)); arr((8.98, 1.05), (10.6, 1.42))
    plt.tight_layout(); plt.savefig(f"{OUT}/F01_pipeline.png", dpi=200); plt.close(); print("F01")

if on("F02"):
    cg, fz, ix = tracks("a"); z = np.load("data/real/leo5/work/hrnet_w32/a_kp.npz"); zi = {n: i for i, n in enumerate(z["names"])}; idxs = [205, 222, 240, 258]
    fig, axs = plt.subplots(2, len(idxs), figsize=(3.3 * len(idxs), 6.4))
    for c, f in enumerate(idxs):
        im = frame("data/real/leo5/work/ca.mp4", f, 1024); x0, y0 = 40, 330
        for r in range(2):
            axs[r, c].imshow(im.crop((x0, y0, x0 + 700, y0 + 560))); axs[r, c].axis("off")
        for side, col in (("left", "#d62728"), ("right", "#1f77b4")):
            P = np.array([z["kp"][f, zi[f"back_{side}_{k}"], :2] for k in ("thai", "knee", "paw")]) - [x0, y0]; axs[0, c].plot(P[:, 0], P[:, 1], "-o", color=col, ms=4, lw=2)
        for key, col in (("hindA", "#d62728"), ("hindB", "#1f77b4")):
            P = np.array([to_crop(fz[key][f, i, :2], cg, f) for i in range(3)]) - [x0, y0]; axs[1, c].plot(P[:, 0], P[:, 1], "-o", color=col, ms=4, lw=2)
        axs[0, c].set_title(f"프레임 {f}", fontsize=10)
    axs[0, 0].text(-0.02, 0.5, "포즈 모델의\n좌/우 이름", transform=axs[0, 0].transAxes, ha="right", va="center", fontsize=10); axs[1, 0].text(-0.02, 0.5, "연속성으로\n이어 붙인 것", transform=axs[1, 0].transAxes, ha="right", va="center", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/F02_identity_fix.png", dpi=150, bbox_inches="tight"); plt.close(); print("F02")

if on("F03"):
    L = {p.replace("\\", "/").split("/")[-1][:-5]: json.load(open(p))["pts"] for p in glob.glob("data/real/leo5/label/user/labels/*.json")}
    cent = json.load(open("data/real/leo5/label/centres.json")) if os.path.exists("data/real/leo5/label/centres.json") else {}
    pack = json.load(open("data/real/leo5/label/pack/frames.json")); meta = {p["id"]: p for p in pack}
    ids = ["a_245", "a_335", "b_212", "b_332"]; names = {"tail": T("꼬리 뿌리", "tail base"), "hip": T("엉덩이", "rump"), "nstif": T("무릎", "stifle"), "nhock": T("뒤꿈치", "hock"), "npaw": T("발", "paw"), "fstif": T("무릎", "stifle"), "fhock": T("뒤꿈치", "hock"), "fpaw": T("발", "paw")}
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
    import base64, io
    for a_, i in zip(axs, ids):
        m = meta[i]; im = Image.open(io.BytesIO(base64.b64decode(m["jpg_b64"]))); a_.imshow(im); a_.axis("off")
        for k, v in L[i].items():
            if not v: continue
            X, Y = (v[0] - m["x0"]) * m["z"], (v[1] - m["y0"]) * m["z"]; col = "#2ca02c" if k in ("tail", "hip") else ("#d62728" if k.startswith("n") else "#1f77b4")
            a_.plot(X, Y, "o", mfc="none", mec=col, ms=9, mew=2); a_.text(X + 8, Y - 8, names[k], color="white", fontsize=7.5, bbox=dict(fc=col, ec="none", pad=1))
        a_.set_title(T(f"{'보정용 구간 a' if i[0] == 'a' else '시험용 구간 b'}, 프레임 {i[2:]}", f"{'calibration clip a' if i[0] == 'a' else 'test clip b'}, frame {i[2:]}"), fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/F03_marks_examples.png", dpi=150); plt.close(); print("F03")

if on("F04"):
    R = pd.read_csv("results/ident_results/U_label_error.csv"); R = R[(R.leg != "body") & (R["clip"] == "b")]; R["e"] = R.err_px / 119.3 * 100
    G = R.groupby(["joint", "model"]).e.median().unstack(); G = G.loc[["paw", "hock", "stif"]][["hrnet_w32", "resnet_50", "fused"]]
    fig, ax = plt.subplots(figsize=(7, 4.2)); w = 0.26
    for k, (m, lab, col) in enumerate((("hrnet_w32", "hrnet_w32", "#9ecae1"), ("resnet_50", "resnet_50", "#6baed6"), ("fused", "합친 추적", "#08519c"))):
        v = G[m].values; ax.bar(np.arange(3) + (k - 1) * w, v, w, label=lab, color=col)
        for i, t in enumerate(v): ax.text(i + (k - 1) * w, t + 1, f"{t:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["발", "뒤꿈치", "무릎"]); ax.set_ylabel("위치 오차 (정강이 길이의 %)"); ax.set_title("표시한 점 대비 추적 오차 (시험용 구간 b, 중앙값)"); ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/F04_tracking_error.png", dpi=150); plt.close(); print("F04")

if on("F05"):
    A = pd.read_csv("results/ident_results/V_convert_angles.csv"); A = A[A.model == "fused"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for a_, (j, nm) in zip(axs, (("stif", "무릎"), ("hock", "뒤꿈치(비절)"))):
        for cl, col, lab in (("a", "#bbbbbb", "구간 a (보정용)"), ("b", "#d62728", "구간 b (시험용)")):
            g = A[A["clip"] == cl]; a_.scatter(g[f"truth_{j}_conv"], g[f"track_{j}"], s=22, color=col, label=lab)
        gb = A[A["clip"] == "b"]; r = np.corrcoef(gb[f"track_{j}"], gb[f"truth_{j}_conv"])[0, 1]; lo, hi = 60, 180; a_.plot([lo, hi], [lo, hi], "k--", lw=0.8); a_.set_xlim(lo, hi); a_.set_ylim(lo, hi)
        a_.set_xlabel(f"표시한 점(관절 중심 환산)의 {nm} 각도 (도)"); a_.set_ylabel("추적한 각도 (도)"); a_.set_title(f"{nm}: 시험용 구간 b 상관 {r:.2f}"); a_.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/F05_track_vs_marked_angle.png", dpi=150); plt.close(); print("F05")

if on("F06"):
    LM = {"xr4": dict(tip=(1425, 1965), hock=(1343, 2071), pat=(806, 1200), stifle=(927, 1256)), "xr6": dict(tip=(1470, 2110), hock=(1388, 2200), pat=(758, 1385), stifle=(894, 1400))}
    fig, axs = plt.subplots(2, 2, figsize=(10, 9.4))
    for r, (k, v) in enumerate(LM.items()):
        im = Image.open(f"data/real/leo_xray/{k}.jpg").convert("RGB"); h = v["hock"]; s = v["stifle"]
        a1 = axs[r, 0]; a1.imshow(im.crop((h[0] - 260, h[1] - 230, h[0] + 260, h[1] + 230)), extent=(h[0] - 260, h[0] + 260, h[1] + 230, h[1] - 230)); a1.axis("off")
        a1.plot(*v["tip"], "o", mfc="none", mec="#ffcc00", ms=12, mew=2); a1.plot(*v["hock"], "x", color="#00e5ff", ms=12, mew=2.5); a1.annotate("", xy=v["hock"], xytext=v["tip"], arrowprops=dict(arrowstyle="->", color="#ffcc00", lw=2))
        a1.text(v["tip"][0] + 14, v["tip"][1] - 8, "뒤꿈치뼈 끝(피부 표식)", color="#ffcc00", fontsize=9); a1.text(v["hock"][0] - 250, v["hock"][1] + 40, "발목관절 중심", color="#00e5ff", fontsize=9); a1.set_title(f"{'왼쪽' if k == 'xr4' else '오른쪽'} 뒷다리: 뒤꿈치", fontsize=10)
        a2 = axs[r, 1]; a2.imshow(im.crop((s[0] - 230, s[1] - 230, s[0] + 230, s[1] + 230)), extent=(s[0] - 230, s[0] + 230, s[1] + 230, s[1] - 230)); a2.axis("off")
        a2.plot(*v["pat"], "o", mfc="none", mec="#ffcc00", ms=12, mew=2); a2.plot(*v["stifle"], "x", color="#00e5ff", ms=12, mew=2.5); a2.annotate("", xy=v["stifle"], xytext=v["pat"], arrowprops=dict(arrowstyle="->", color="#ffcc00", lw=2))
        a2.text(v["pat"][0] - 10, v["pat"][1] - 20, "무릎뼈 앞면", color="#ffcc00", fontsize=9); a2.text(v["stifle"][0] + 16, v["stifle"][1] + 40, "무릎관절 중심", color="#00e5ff", fontsize=9); a2.set_title(f"{'왼쪽' if k == 'xr4' else '오른쪽'} 뒷다리: 무릎", fontsize=10)
    fig.text(0.5, 0.005, "뒤꿈치뼈 끝은 관절 중심보다 정강이 길이의 약 12 % 위, 10 % 뒤, 무릎뼈 앞면은 약 16 % 앞 (두 다리 평균)", ha="center", fontsize=10)
    plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig(f"{OUT}/F06_xray_offsets.png", dpi=150); plt.close(); print("F06")

if on("F07"):
    W = pd.read_csv("results/ident_results/W_sim_vs_video.csv"); fig, ax = plt.subplots(figsize=(9, 4.6))
    simA = W[(W.source == "sim fitA") & (W.paw == "mtp_end")]
    labs, vals, cols = [], [], []
    for rump, lab, col in (("Ca_1", "모델 fitA\n(기존 기준점)", "#bdbdbd"), ("xray_rump", "모델 fitA\n(방사선 기준점)", "#5b8def")): labs.append(lab); vals.append(float(simA[simA.rump == rump].stif_mean.iloc[0])); cols.append(col)
    vid = W[W.source.str.startswith("video") & (W.rump == "tail_est(c_fit)")]; vb = W[W.source.str.startswith("video") & (W.rump == "back_end")]
    labs.append("영상\n(back_end 그대로)"); vals.append(float(vb.stif_mean.mean())); cols.append("#fdd0a2"); labs.append("영상\n(꼬리 뿌리 추정)"); vals.append(float(vid.stif_mean.mean())); cols.append("#f08a24")
    ax.bar(labs, vals, color=cols, width=0.6)
    tops = {2: float(vb.stif_mean.max()), 3: float(vid.stif_mean.max())}
    for i, v in enumerate(vals): ax.text(i, max(v, tops.get(i, v)) + 1.8, f"{v:.0f}", ha="center", fontsize=11)
    ax.errorbar([3], [vid.stif_mean.mean()], yerr=[[vid.stif_mean.mean() - vid.stif_mean.min()], [vid.stif_mean.max() - vid.stif_mean.mean()]], color="k", capsize=5); ax.errorbar([2], [vb.stif_mean.mean()], yerr=[[vb.stif_mean.mean() - vb.stif_mean.min()], [vb.stif_mean.max() - vb.stif_mean.mean()]], color="k", capsize=5)
    ax.set_ylim(80, 175); ax.set_ylabel("정상 걸음의 무릎 각도 평균 (도)"); ax.set_title("엉덩이 기준점의 정의를 맞추면 모델과 영상의 평균 무릎 각도가 가까워진다 (막대 선: 구간 a, b 두 다리의 범위)"); ax.title.set_fontsize(10); plt.tight_layout(); plt.savefig(f"{OUT}/F07_rump_definition.png", dpi=150); plt.close(); print("F07")

if on("F08"):
    tags = ["a", "b", "l2a", "l2b", "t7725"]; lab = {"a": "클립 a", "b": "클립 b", "l2a": "이전 촬영 1", "l2b": "이전 촬영 2", "t7725": "IMG_7725"}
    X = {t: clip_stifle(t, CR) for t in tags}; mus = {"정상": C["mu_normal"], "등급 2": C["mu_normal"] + C["d_abnormal"], "등급 3": C["mu_normal"] + C["d_severe"]}; sd = float(np.sqrt(4.5 ** 2 + C["sd_sim"] ** 2 + C["sd_c"] ** 2 + C["sd_tracker"] ** 2))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (k, m), col in zip(mus.items(), ("#2ca02c", "#ff7f0e", "#d62728")): ax.axhspan(m - sd, m + sd, color=col, alpha=0.15); ax.axhline(m, color=col, ls="--", lw=1.2); ax.text(4.6, m, f"모델 예측 {k} {m:.0f}도", va="center", fontsize=9, color=col, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    for i, t in enumerate(tags):
        ax.plot([i, i], X[t][2], color="#999", lw=1); ax.plot(i, X[t][0], "ko", ms=8); ax.plot([i] * 2, X[t][2], "_", color="#555", ms=10)
    ax.set_xticks(range(5)); ax.set_xticklabels([lab[t] for t in tags]); ax.set_xlim(-0.5, 5.7); ax.set_ylabel("구간 평균 무릎 각도 (도)"); ax.set_title("정상으로 아는 영상 5구간의 무릎 각도와 모델의 등급별 예측 (띠: ±판정 폭 %.1f도)" % sd, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/F08_judge_summary.png", dpi=150); plt.close(); print("F08")

if on("F09"):
    from analysis.virtual_markers import sim_full
    from analysis.identify_angles import lp
    from analysis.leo_judge_lib import CAUD, DORS
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for run, lab, col in (("fitR3", "정상", "#2ca02c"), ("fitR3_G2", "등급 2", "#ff7f0e"), ("fitR3_G3", "등급 3", "#d62728")):
        path = f"results/ident/{run}.npz"; L, axv, fps = sim_full(path); z = np.load(path, allow_pickle=True); names = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; ix = {k: i for i, k in enumerate(names)}; xz = lambda k: P[:, ix[k]][:, [0, 2]]
        hip, st, hk = xz("upper_leg_L"), xz("lower_leg_L"), xz("foot_L"); dd = np.stack([-axv[:, 1], axv[:, 0]], 1); dd = dd if dd[:, 1].mean() > 0 else -dd; Tm = np.median(np.linalg.norm(st - hk, axis=1))
        a_ = ang(lp(hip + Tm * (CAUD * axv + DORS * dd), fps), lp(st, fps), lp(hk, fps)); t = np.arange(len(a_)) / fps; m = (t > 2) & (t < 6); ax.plot(t[m] - 2, a_[m], color=col, lw=1.8, label=f"{lab} (평균 {a_[t > 2].mean():.0f}도)")
    ax.set_xlabel("시간 (초)"); ax.set_ylabel("무릎 각도 (도)"); ax.set_title("모델(fitR3)이 만든 등급별 걸음의 무릎 각도"); ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/F09_model_grades.png", dpi=150); plt.close(); print("F09")

if on("F10"):
    vp = "results/videos/MODEL_grades_fitR3.mp4"; probe = subprocess.run([FF, "-i", vp], capture_output=True, text=True).stderr; import re
    W_, H_ = [int(v) for v in re.search(r"Video:.*?, (\d{3,5})x(\d{3,5})", probe).groups()]; times = [1.0, 2.5, 4.0]
    fig, axs = plt.subplots(len(times), 1, figsize=(12, 2.6 * len(times)))
    for a_, t in zip(axs, times):
        r = subprocess.run([FF, "-v", "error", "-ss", str(t), "-i", vp, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True); im = np.frombuffer(r.stdout, np.uint8).reshape(H_, W_, 3); a_.imshow(im); a_.axis("off")
        a_.set_title(f"영상 {t:.1f}초 지점 (왼쪽부터 정상, 등급 2, 등급 3)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/F10_model_snapshots.png", dpi=140); plt.close(); print("F10")

for n, (tag, title, true) in enumerate((("a", "검사 영상 1", "정상"), ("b", "검사 영상 2", "정상"), ("l2a", "검사 영상 3", "정상"), ("l2b", "검사 영상 4", "정상"), ("t7725", "검사 영상 5", "정상"))):
    key = f"F{11 + n}"
    if not on(key): continue
    cg, fz, ix = tracks(tag); ins = np.nonzero(fz["inside"])[0]; a0, b0 = int(ins[0]), int(ins[-1]) + 1
    body = fz["body"][a0:b0]; be, bb = lpv(body[:, ix["back_end"], :2]), lpv(body[:, ix["back_base"], :2]); ub = unit(be - bb)
    legs = {k: np.stack([lpv(fz[k][a0:b0][:, i, :2]) for i in range(3)], 1) for k in ("hindA", "hindB")}; Tv = {k: float(np.median(np.linalg.norm(v[:, 0] - v[:, 1], axis=1))) for k, v in legs.items()}
    rump = {k: be + CR * Tv[k] * ub for k in legs}; A = {k: ang(rump[k], legs[k][:, 0], legs[k][:, 1]) for k in legs}; sel = np.linspace(a0 + 6, b0 - 8, 6).astype(int); x_, win, lg = clip_stifle(tag, CR)
    fig, axs = plt.subplots(2, 3, figsize=(9, 5.1))
    for a_, i in zip(axs.flat, sel):
        j = i - a0; im = frame(f"data/real/leo5/work/c{tag}.mp4", i, 1024); a_.imshow(im.crop((60, 260, 960, 960))); a_.axis("off")
        for k, col in (("hindA", "#d62728"), ("hindB", "#1f77b4")):
            P = [to_crop(rump[k][j], cg, i)] + [to_crop(legs[k][j, q], cg, i) for q in range(3)]; P = np.array(P) - [60, 260]; a_.plot(P[:, 0], P[:, 1], "-o", color=col, lw=2.2, ms=4); a_.text(P[1, 0] + 8, P[1, 1] - 12, f"{A[k][j]:.0f}°", color=col, fontsize=13, fontweight="bold")
        a_.set_title(f"{(i - a0) / 120:.2f}초", fontsize=12)
    fig.suptitle(f"{title}: 구간 평균 무릎 각도 {x_:.1f}도 (두 다리 {lg[0]:.1f}, {lg[1]:.1f}), 온전히 보인 구간 {win:.1f}초", fontsize=14); plt.tight_layout(); plt.savefig(f"{OUT}/{key}_scenes_{tag}.png", dpi=130, bbox_inches="tight", pad_inches=0.06); plt.close(); print(key)
